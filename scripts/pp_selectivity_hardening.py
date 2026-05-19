"""Hardening checks for the PP selectivity model:

1. Triple-fold validation: train(<=2022), tune threshold(2023), evaluate(2024)
   - eliminates the 'tuned threshold on the test set' confound
2. T-1 min execution: train model can use openings (more data), but the
   backtest must use prices at T-1 (when a real trader would actually bet)
3. Per-league breakdown of the held-out 2024 backtest

If the strategy survives all three, it's a real candidate for live trading.
If it breaks on any one, we've found the artefact.
"""
from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.ingestion.storage import read_balls
from src.logging_setup import configure_logging

logger = logging.getLogger(__name__)


def main() -> int:
    configure_logging()

    # --- Load saved per-runner snapshot table -------------------------------
    df = pd.read_parquet("data/processed/pp_time_anchors.parquet")
    logger.info("base rows: %d", len(df))

    # --- Load each league's balls/matches once ------------------------------
    leagues = ["ipl", "bbl", "psl", "cpl", "ntb"]
    all_balls = []
    league_by_match = {}
    for league in leagues:
        d = Path("data/processed") if league == "ipl" else Path(f"data/processed/{league}")
        b = read_balls(d)
        if not b.empty:
            all_balls.append(b)
        m = pd.read_parquet(d / "matches.parquet")
        for mid in m["match_id"].astype(str).unique():
            league_by_match[mid] = league
    balls = pd.concat(all_balls, ignore_index=True)

    # --- Per-match metadata: season, venue, batting team per innings --------
    season_map = (
        balls[["match_id", "season"]].drop_duplicates("match_id")
        .set_index("match_id")["season"].astype(int).to_dict()
    )
    venue_map = (
        balls[["match_id", "venue"]].drop_duplicates("match_id").set_index("match_id")["venue"].to_dict()
    )
    bat_team_map = {}
    for (mid, inn), g in balls.groupby(["match_id", "innings"]):
        bat_team_map[(mid, int(inn))] = g["batting_team"].iloc[0]

    # --- Leak-free team PP rates and venue par from earlier ----------------
    pp_rows = []
    for (mid, innings), g in balls.groupby(["match_id", "innings"]):
        g = g.sort_values(["over", "ball", "is_legal_delivery"], ascending=[True, True, False]).reset_index(drop=True)
        legal_idx = np.where(g["is_legal_delivery"].values)[0]
        if len(legal_idx) < 36:
            continue
        cut_36 = legal_idx[35] + 1
        pp_rows.append({
            "match_id": mid, "innings": int(innings),
            "season": int(g["season"].iloc[0]),
            "batting_team": g["batting_team"].iloc[0],
            "bowling_team": g["bowling_team"].iloc[0],
            "venue": g["venue"].iloc[0],
            "pp_total": int(g.iloc[:cut_36]["runs_total"].sum()),
        })
    pp_df = pd.DataFrame(pp_rows)
    seasons = sorted(pp_df["season"].unique())
    bat_pp, bowl_pp, ven_par = {}, {}, {}
    for s in seasons:
        prior = pp_df[pp_df["season"] < s]
        if prior.empty:
            continue
        for t, m in prior.groupby("batting_team")["pp_total"].mean().items():
            bat_pp[(str(t), int(s))] = float(m)
        for t, m in prior.groupby("bowling_team")["pp_total"].mean().items():
            bowl_pp[(str(t), int(s))] = float(m)
        for v, m in prior.groupby("venue")["pp_total"].mean().items():
            ven_par[(str(v), int(s))] = float(m)

    LEAGUE_DEFAULT = 50.0

    # --- Feature construction ----------------------------------------------
    df["season"] = df["match_id"].astype(str).map(season_map)
    df["venue"] = df["match_id"].astype(str).map(venue_map)
    df["batting_team"] = df.apply(lambda r: bat_team_map.get((r["match_id"], int(r["innings"])), ""), axis=1)
    # bowling team = whichever team in match isn't this innings's batting team
    df["bowling_team"] = df.apply(
        lambda r: next(
            (t for (m, inn), t in bat_team_map.items()
             if m == r["match_id"] and inn != int(r["innings"])),
            "",
        ),
        axis=1,
    )
    df["bat_pp_prior"]  = df.apply(lambda r: bat_pp.get((r["batting_team"], int(r["season"])), LEAGUE_DEFAULT), axis=1)
    df["bowl_pp_prior"] = df.apply(lambda r: bowl_pp.get((r["bowling_team"], int(r["season"])), LEAGUE_DEFAULT), axis=1)
    df["venue_par"] = df.apply(lambda r: ven_par.get((r["venue"], int(r["season"])), LEAGUE_DEFAULT), axis=1)
    df["league"] = df["match_id"].astype(str).map(league_by_match)

    # Prices
    df = df.dropna(subset=["first_ltp"])
    df = df[df["first_ltp"] > 1.0].copy()
    df["implied_open"] = 1.0 / df["first_ltp"]
    df["implied_t1"] = np.where(
        df["ltp_t_minus_1"].notna() & (df["ltp_t_minus_1"] > 1.0),
        1.0 / df["ltp_t_minus_1"], np.nan,
    )

    df["x_minus_par"]  = df["threshold_X"] - df["venue_par"]
    df["x_minus_bat"]  = df["threshold_X"] - df["bat_pp_prior"]
    df["x_minus_bowl"] = df["threshold_X"] - df["bowl_pp_prior"]
    for L in leagues:
        df[f"is_{L}"] = (df["league"] == L).astype(int)

    feature_cols = [
        "threshold_X", "implied_open",
        "bat_pp_prior", "bowl_pp_prior", "venue_par",
        "x_minus_par", "x_minus_bat", "x_minus_bowl",
        "innings",
    ] + [f"is_{L}" for L in leagues]

    df = df.dropna(subset=["season", "actual_over_X"] + feature_cols)
    logger.info("model dataset after feature joins: %d rows", len(df))

    # --- Three-way split: train / validate-threshold / test ----------------
    seasons_sorted = sorted(df["season"].unique())
    if len(seasons_sorted) < 3:
        logger.error("need at least 3 seasons for triple-fold; got %s", seasons_sorted)
        return 1
    test_season = seasons_sorted[-1]
    val_season  = seasons_sorted[-2]
    train_df = df[df["season"] <  val_season].copy()
    val_df   = df[df["season"] == val_season].copy()
    test_df  = df[df["season"] == test_season].copy()
    logger.info("train (<%s): %d   validate (%s): %d   test (%s): %d",
                val_season, len(train_df), val_season, len(val_df), test_season, len(test_df))

    # --- Train --------------------------------------------------------------
    x_tr = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_tr = train_df["actual_over_X"].to_numpy(dtype=np.float32)
    x_va = val_df[feature_cols].to_numpy(dtype=np.float32)
    y_va = val_df["actual_over_X"].to_numpy(dtype=np.float32)
    gbm = lgb.train(
        params={
            "objective": "binary", "metric": "binary_logloss",
            "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 100,
            "verbose": -1, "seed": 17, "deterministic": True,
        },
        train_set=lgb.Dataset(x_tr, label=y_tr, feature_name=feature_cols),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(x_va, label=y_va, feature_name=feature_cols)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # --- Tune threshold on validation (using OPENING prices for trade pricing) ----
    val_df = val_df.assign(model_p=gbm.predict(x_va))
    val_df["edge_open"] = val_df["model_p"] - val_df["implied_open"]
    candidates = [-0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08, -0.10]
    val_ll_market = log_loss(y_va, np.clip(val_df["implied_open"], 1e-3, 1-1e-3))
    val_ll_model  = log_loss(y_va, np.clip(val_df["model_p"],     1e-3, 1-1e-3))
    print(f"\n=== Validation ({val_season}) log-loss: market={val_ll_market:.4f}  model={val_ll_model:.4f} ===")

    print(f"\n=== Validation: ROI vs edge threshold (OPENING prices for execution) ===")
    best_thresh = None
    best_roi = -np.inf
    for th in candidates:
        sub = val_df[(val_df["edge_open"] < th) &
                     (val_df["implied_open"] >= 0.10) & (val_df["implied_open"] <= 0.90)]
        if len(sub) < 20:
            print(f"  edge<{th}: n={len(sub):3d}  too small")
            continue
        pnl = backtest_lays(sub, price_col="implied_open")["pnl"].sum()
        roi = pnl / (len(sub) * 100.0)
        print(f"  edge<{th}: n={len(sub):3d}  pnl={pnl:+.0f}  ROI={roi:+.2%}")
        if roi > best_roi:
            best_roi = roi; best_thresh = th
    if best_thresh is None:
        logger.error("no viable threshold on validation; aborting")
        return 1
    print(f"\nBest validation threshold: edge < {best_thresh}  (val ROI {best_roi:+.2%})")

    # --- Apply to test (2024) -- OPENING PRICES ---------------------------
    x_te = test_df[feature_cols].to_numpy(dtype=np.float32)
    test_df = test_df.assign(model_p=gbm.predict(x_te))
    test_df["edge_open"] = test_df["model_p"] - test_df["implied_open"]
    test_df["edge_t1"]   = test_df["model_p"] - test_df["implied_t1"]

    test_ll_market = log_loss(test_df["actual_over_X"],
                              np.clip(test_df["implied_open"], 1e-3, 1-1e-3))
    test_ll_model  = log_loss(test_df["actual_over_X"],
                              np.clip(test_df["model_p"],     1e-3, 1-1e-3))
    print(f"\n=== TEST ({test_season}) log-loss: market={test_ll_market:.4f}  model={test_ll_model:.4f} ===")

    print(f"\n=== TEST (OPENING price for execution, threshold from validation) ===")
    sub_open = test_df[(test_df["edge_open"] < best_thresh) &
                       (test_df["implied_open"] >= 0.10) & (test_df["implied_open"] <= 0.90)].copy()
    print(f"All leagues: n={len(sub_open)}")
    if len(sub_open) > 0:
        bt = backtest_lays(sub_open, price_col="implied_open")
        print_stats(bt)
        print("\n  Per-league:")
        for league in leagues:
            ls = sub_open[sub_open["league"] == league]
            if ls.empty: continue
            bts = backtest_lays(ls, price_col="implied_open")
            n = len(bts); pnl = bts["pnl"].sum()
            print(f"    {league.upper():5s} n={n:3d}  pnl={pnl:+7.0f}  ROI={pnl/(n*100):+.2%}")

    print(f"\n=== TEST (T-1 MIN price for execution, threshold from validation) ===")
    # Same model edge filter, but use T-1 prices for actual trade execution
    # Drop rows where T-1 price is missing (can't trade there)
    sub_t1 = test_df.dropna(subset=["implied_t1"])
    sub_t1 = sub_t1[(sub_t1["edge_t1"] < best_thresh) &
                    (sub_t1["implied_t1"] >= 0.10) & (sub_t1["implied_t1"] <= 0.90)].copy()
    print(f"All leagues: n={len(sub_t1)} (rows with T-1 price)")
    if len(sub_t1) > 0:
        bt = backtest_lays(sub_t1, price_col="implied_t1")
        print_stats(bt)
        print("\n  Per-league:")
        for league in leagues:
            ls = sub_t1[sub_t1["league"] == league]
            if ls.empty: continue
            bts = backtest_lays(ls, price_col="implied_t1")
            n = len(bts); pnl = bts["pnl"].sum()
            print(f"    {league.upper():5s} n={n:3d}  pnl={pnl:+7.0f}  ROI={pnl/(n*100):+.2%}")
    return 0


def backtest_lays(df: pd.DataFrame, *, price_col: str, stake: float = 100.0, commission: float = 0.05) -> pd.DataFrame:
    df = df.copy()
    price = 1.0 / df[price_col]
    won = df["actual_over_X"] == 0
    df["pnl"] = np.where(won, stake * (1 - commission), -stake * (price - 1.0))
    return df


def print_stats(trades: pd.DataFrame, stake: float = 100.0) -> None:
    n = len(trades)
    if n == 0:
        print("  no trades"); return
    pnl = trades["pnl"].sum()
    wr = (trades["pnl"] > 0).mean()
    aw = trades.loc[trades["pnl"] > 0, "pnl"].mean() if (trades["pnl"] > 0).any() else 0
    al = trades.loc[trades["pnl"] < 0, "pnl"].mean() if (trades["pnl"] < 0).any() else 0
    print(f"  n={n}  win_rate={wr:.3f}  pnl={pnl:+.0f}  avg_win={aw:+.1f}  avg_loss={al:+.1f}  ROI={pnl/(n*stake):+.2%}")


if __name__ == "__main__":
    raise SystemExit(main())

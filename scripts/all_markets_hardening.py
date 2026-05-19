"""Hardening pipeline for all 'X or more' style cricket markets.

Reads data/processed/all_innings_markets.parquet (which has 6_over, 10_over,
and full_innings rows) and runs the same triple-fold validation +
T-1 execution + per-league analysis on each market type.

Compares ROI per market_type so we know where the strongest tradeable edge
lives.
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


def build_team_stats(balls: pd.DataFrame, target_balls: int) -> tuple[dict, dict, dict]:
    """Per-(team, season) batting/bowling rolling totals at `target_balls` legal balls.
    Leak-free (uses STRICTLY prior seasons)."""
    rows = []
    for (mid, innings), g in balls.groupby(["match_id", "innings"]):
        g = g.sort_values(["over", "ball", "is_legal_delivery"], ascending=[True, True, False]).reset_index(drop=True)
        legal_idx = np.where(g["is_legal_delivery"].values)[0]
        if target_balls == 120:
            total = int(g["runs_total"].sum())  # final innings total even if short
            if len(legal_idx) < 30:
                continue
        else:
            if len(legal_idx) < target_balls:
                continue
            cut = legal_idx[target_balls - 1] + 1
            total = int(g.iloc[:cut]["runs_total"].sum())
        rows.append({
            "season": int(g["season"].iloc[0]),
            "batting_team": g["batting_team"].iloc[0],
            "bowling_team": g["bowling_team"].iloc[0],
            "venue": g["venue"].iloc[0],
            "total": total,
        })
    pp_df = pd.DataFrame(rows)
    bat, bowl, ven = {}, {}, {}
    if pp_df.empty:
        return bat, bowl, ven
    for s in sorted(pp_df["season"].unique()):
        prior = pp_df[pp_df["season"] < s]
        if prior.empty:
            continue
        for t, m in prior.groupby("batting_team")["total"].mean().items():
            bat[(str(t), int(s))] = float(m)
        for t, m in prior.groupby("bowling_team")["total"].mean().items():
            bowl[(str(t), int(s))] = float(m)
        for v, m in prior.groupby("venue")["total"].mean().items():
            ven[(str(v), int(s))] = float(m)
    return bat, bowl, ven


def run_market_type(
    df: pd.DataFrame,
    market_type: str,
    target_balls: int,
    balls: pd.DataFrame,
    bat_team_map: dict,
    league_by_match: dict,
    season_map: dict,
    venue_map: dict,
    leagues: list[str],
):
    sub = df[df["market_type"] == market_type].copy()
    if sub.empty:
        print(f"\n=== {market_type} ===  (no rows)")
        return None

    # Build leak-free prior stats specific to this market's target_balls
    bat_pp, bowl_pp, ven_par = build_team_stats(balls, target_balls)
    DEFAULT = float(np.mean([v for v in ven_par.values()])) if ven_par else 50.0

    sub["season"] = sub["match_id"].astype(str).map(season_map)
    sub["venue"]  = sub["match_id"].astype(str).map(venue_map)
    sub["batting_team"] = sub.apply(
        lambda r: bat_team_map.get((r["match_id"], int(r["innings"])), ""), axis=1
    )
    sub["bowling_team"] = sub.apply(
        lambda r: next(
            (t for (m, inn), t in bat_team_map.items()
             if m == r["match_id"] and inn != int(r["innings"])),
            "",
        ), axis=1,
    )
    sub["bat_prior"]  = sub.apply(lambda r: bat_pp.get((r["batting_team"], int(r["season"])), DEFAULT), axis=1)
    sub["bowl_prior"] = sub.apply(lambda r: bowl_pp.get((r["bowling_team"], int(r["season"])), DEFAULT), axis=1)
    sub["venue_par"]  = sub.apply(lambda r: ven_par.get((r["venue"], int(r["season"])), DEFAULT), axis=1)

    sub = sub.dropna(subset=["first_ltp"])
    sub = sub[sub["first_ltp"] > 1.0].copy()
    sub["implied_open"] = 1.0 / sub["first_ltp"]
    sub["implied_t1"]   = np.where(sub["ltp_t_minus_1"].notna() & (sub["ltp_t_minus_1"] > 1.0),
                                    1.0 / sub["ltp_t_minus_1"], np.nan)
    sub["x_minus_par"]  = sub["threshold_X"] - sub["venue_par"]
    sub["x_minus_bat"]  = sub["threshold_X"] - sub["bat_prior"]
    sub["x_minus_bowl"] = sub["threshold_X"] - sub["bowl_prior"]
    for L in leagues:
        sub[f"is_{L}"] = (sub["league"] == L).astype(int)

    features = [
        "threshold_X", "implied_open",
        "bat_prior", "bowl_prior", "venue_par",
        "x_minus_par", "x_minus_bat", "x_minus_bowl",
        "innings",
    ] + [f"is_{L}" for L in leagues]
    sub = sub.dropna(subset=["season", "actual_over_X"] + features)
    if sub.empty:
        print(f"\n=== {market_type} ===  (no rows after dropna)")
        return None

    seasons_sorted = sorted(sub["season"].unique())
    if len(seasons_sorted) < 3:
        print(f"\n=== {market_type} ===  only {len(seasons_sorted)} seasons; need 3 for triple-fold")
        return None
    test_season = seasons_sorted[-1]
    val_season  = seasons_sorted[-2]
    train_df = sub[sub["season"] <  val_season].copy()
    val_df   = sub[sub["season"] == val_season].copy()
    test_df  = sub[sub["season"] == test_season].copy()

    x_tr = train_df[features].to_numpy(dtype=np.float32)
    y_tr = train_df["actual_over_X"].to_numpy(dtype=np.float32)
    x_va = val_df[features].to_numpy(dtype=np.float32)
    y_va = val_df["actual_over_X"].to_numpy(dtype=np.float32)
    gbm = lgb.train(
        params={
            "objective": "binary", "metric": "binary_logloss",
            "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 100,
            "verbose": -1, "seed": 17, "deterministic": True,
        },
        train_set=lgb.Dataset(x_tr, label=y_tr, feature_name=features),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(x_va, label=y_va, feature_name=features)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    val_df["model_p"] = gbm.predict(x_va)
    val_df["edge_open"] = val_df["model_p"] - val_df["implied_open"]
    val_ll_market = log_loss(y_va, np.clip(val_df["implied_open"], 1e-3, 1-1e-3))
    val_ll_model  = log_loss(y_va, np.clip(val_df["model_p"],     1e-3, 1-1e-3))

    candidates = [-0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08, -0.10]
    best_th, best_roi = None, -np.inf
    val_results = []
    for th in candidates:
        s = val_df[(val_df["edge_open"] < th) &
                   (val_df["implied_open"] >= 0.10) & (val_df["implied_open"] <= 0.90)]
        if len(s) < 20:
            continue
        bt = backtest_lays(s, price_col="implied_open")
        pnl, n = bt["pnl"].sum(), len(s)
        roi = pnl / (n * 100.0)
        val_results.append((th, n, pnl, roi))
        if roi > best_roi:
            best_th, best_roi = th, roi
    if best_th is None:
        print(f"\n=== {market_type} ===  no viable validation threshold")
        return None

    # Apply best threshold to test
    x_te = test_df[features].to_numpy(dtype=np.float32)
    test_df["model_p"] = gbm.predict(x_te)
    test_df["edge_open"] = test_df["model_p"] - test_df["implied_open"]
    test_df["edge_t1"]   = test_df["model_p"] - test_df["implied_t1"]
    test_ll_market = log_loss(test_df["actual_over_X"], np.clip(test_df["implied_open"], 1e-3, 1-1e-3))
    test_ll_model  = log_loss(test_df["actual_over_X"], np.clip(test_df["model_p"],     1e-3, 1-1e-3))

    so = test_df[(test_df["edge_open"] < best_th) &
                 (test_df["implied_open"] >= 0.10) & (test_df["implied_open"] <= 0.90)]
    bt_open = backtest_lays(so, price_col="implied_open") if len(so) > 0 else pd.DataFrame()

    s_t1 = test_df.dropna(subset=["implied_t1"])
    s_t1 = s_t1[(s_t1["edge_t1"] < best_th) &
                (s_t1["implied_t1"] >= 0.10) & (s_t1["implied_t1"] <= 0.90)]
    bt_t1 = backtest_lays(s_t1, price_col="implied_t1") if len(s_t1) > 0 else pd.DataFrame()

    per_league_open = []
    for L in leagues:
        ls = so[so["league"] == L]
        if ls.empty: continue
        b = backtest_lays(ls, price_col="implied_open")
        per_league_open.append((L, len(b), b["pnl"].sum(), b["pnl"].sum()/(len(b)*100)))

    return {
        "market_type": market_type,
        "n_rows": len(sub),
        "train": len(train_df), "val": len(val_df), "test": len(test_df),
        "val_season": val_season, "test_season": test_season,
        "val_ll_market": val_ll_market, "val_ll_model": val_ll_model,
        "test_ll_market": test_ll_market, "test_ll_model": test_ll_model,
        "best_threshold": best_th,
        "val_results": val_results,
        "bt_open": bt_open, "bt_t1": bt_t1,
        "per_league_open": per_league_open,
    }


def backtest_lays(df: pd.DataFrame, *, price_col: str, stake: float = 100.0, commission: float = 0.05) -> pd.DataFrame:
    df = df.copy()
    price = 1.0 / df[price_col]
    won = df["actual_over_X"] == 0
    df["pnl"] = np.where(won, stake * (1 - commission), -stake * (price - 1.0))
    return df


def main() -> int:
    configure_logging()
    df = pd.read_parquet("data/processed/all_innings_markets.parquet")
    logger.info("loaded %d rows: %s", len(df), df["market_type"].value_counts().to_dict())

    leagues = ["ipl", "bbl", "psl", "cpl", "ntb"]
    all_balls = []
    for league in leagues:
        d = Path("data/processed") if league == "ipl" else Path(f"data/processed/{league}")
        b = read_balls(d)
        if not b.empty: all_balls.append(b)
    balls = pd.concat(all_balls, ignore_index=True)

    bat_team_map = {}
    for (mid, inn), g in balls.groupby(["match_id", "innings"]):
        bat_team_map[(mid, int(inn))] = g["batting_team"].iloc[0]
    season_map = balls[["match_id", "season"]].drop_duplicates("match_id").set_index("match_id")["season"].astype(int).to_dict()
    venue_map = balls[["match_id", "venue"]].drop_duplicates("match_id").set_index("match_id")["venue"].to_dict()
    league_by_match = {}
    for league in leagues:
        d = Path("data/processed") if league == "ipl" else Path(f"data/processed/{league}")
        m = pd.read_parquet(d / "matches.parquet")
        for mid in m["match_id"].astype(str).unique():
            league_by_match[mid] = league

    market_specs = [("6_over", 36), ("10_over", 60), ("full_innings", 120)]
    results = []
    for mt, balls_target in market_specs:
        r = run_market_type(df, mt, balls_target, balls, bat_team_map,
                            league_by_match, season_map, venue_map, leagues)
        if r is None: continue
        results.append(r)

        print(f"\n========== {mt} ==========")
        print(f"  rows: {r['n_rows']} (train={r['train']}, val={r['val']}, test={r['test']})")
        print(f"  validation log-loss: market={r['val_ll_market']:.4f}  model={r['val_ll_model']:.4f}")
        print(f"  test log-loss:       market={r['test_ll_market']:.4f}  model={r['test_ll_model']:.4f}")
        print(f"\n  Validation thresholds (val season {r['val_season']}):")
        for th, n, pnl, roi in r["val_results"]:
            mark = "  <- best" if th == r["best_threshold"] else ""
            print(f"    edge<{th}: n={n:3d}  pnl={pnl:+.0f}  ROI={roi:+.2%}{mark}")
        print(f"\n  TEST (season {r['test_season']}), threshold = {r['best_threshold']}:")
        if not r["bt_open"].empty:
            bt = r["bt_open"]
            print(f"    OPEN execution: n={len(bt):3d}  win%={(bt['pnl']>0).mean():.3f}  pnl={bt['pnl'].sum():+.0f}  ROI={bt['pnl'].sum()/(len(bt)*100):+.2%}")
        if not r["bt_t1"].empty:
            bt = r["bt_t1"]
            print(f"    T-1  execution: n={len(bt):3d}  win%={(bt['pnl']>0).mean():.3f}  pnl={bt['pnl'].sum():+.0f}  ROI={bt['pnl'].sum()/(len(bt)*100):+.2%}")
        if r["per_league_open"]:
            print(f"    Per-league (OPEN):")
            for L, n, pnl, roi in r["per_league_open"]:
                print(f"      {L.upper():5s} n={n:3d}  pnl={pnl:+7.0f}  ROI={roi:+.2%}")

    print("\n\n===== SUMMARY =====")
    print(f"{'market':<15} {'val_n':>5} {'val_roi':>9} {'best_th':>8} {'test_n_open':>12} {'test_roi_open':>14} {'test_n_t1':>10} {'test_roi_t1':>13}")
    for r in results:
        val_winning = [v for v in r["val_results"] if v[0] == r["best_threshold"]]
        val_n, val_roi = (val_winning[0][1], val_winning[0][3]) if val_winning else (0, 0.0)
        bt_o, bt_t = r["bt_open"], r["bt_t1"]
        n_o = len(bt_o); roi_o = bt_o["pnl"].sum()/(n_o*100) if n_o else 0.0
        n_t = len(bt_t); roi_t = bt_t["pnl"].sum()/(n_t*100) if n_t else 0.0
        print(f"{r['market_type']:<15} {val_n:>5d} {val_roi:>+9.2%} {r['best_threshold']:>+8.2f} {n_o:>12d} {roi_o:>+14.2%} {n_t:>10d} {roi_t:>+13.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

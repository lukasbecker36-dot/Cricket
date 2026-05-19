"""Selectivity model on top of the PP threshold mispricing signal.

Base signal: cross-league mid-bin overpricing of '~7%%'. This model tries to
identify WHICH mid-bin runners are most overpriced using team batting/bowling
PP rates, venue PP par, league, and the market's own implied probability.

Train walk-forward by season. Backtest a 'lay only the highest model edge'
strategy against a 'lay every mid-bin runner' baseline.
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


def innings_batting_team(balls: pd.DataFrame) -> dict[tuple[str, int], str]:
    """For each (match_id, innings), the batting team's name."""
    sub = balls[["match_id", "innings", "batting_team"]].drop_duplicates(["match_id", "innings"])
    return {(row.match_id, int(row.innings)): row.batting_team for row in sub.itertuples(index=False)}


def venue_for_match(balls: pd.DataFrame) -> dict[str, str]:
    return balls[["match_id", "venue"]].drop_duplicates("match_id").set_index("match_id")["venue"].to_dict()


def season_for_match(balls: pd.DataFrame) -> dict[str, int]:
    return (
        balls[["match_id", "season"]].drop_duplicates("match_id")
        .set_index("match_id")["season"].astype(int).to_dict()
    )


def team_pp_rates(balls: pd.DataFrame) -> tuple[dict, dict]:
    """For each (team, season): rolling average PP total scored / conceded
    using STRICTLY PRIOR seasons. Leak-free.
    """
    rows = []
    for (mid, innings), g in balls.groupby(["match_id", "innings"]):
        g = g.sort_values(["over", "ball", "is_legal_delivery"], ascending=[True, True, False]).reset_index(drop=True)
        legal_idx = np.where(g["is_legal_delivery"].values)[0]
        if len(legal_idx) < 36:
            continue
        cut_36 = legal_idx[35] + 1
        rows.append({
            "match_id": mid,
            "innings": int(innings),
            "season": int(g["season"].iloc[0]),
            "batting_team": g["batting_team"].iloc[0],
            "bowling_team": g["bowling_team"].iloc[0],
            "pp_total": int(g.iloc[:cut_36]["runs_total"].sum()),
        })
    pp_df = pd.DataFrame(rows)
    # Per-team per-season aggregates from prior seasons
    seasons = sorted(pp_df["season"].unique())
    bat_avg: dict[tuple[str, int], float] = {}
    bowl_avg: dict[tuple[str, int], float] = {}
    for s in seasons:
        prior = pp_df[pp_df["season"] < s]
        if prior.empty:
            continue
        for team, m in prior.groupby("batting_team")["pp_total"].mean().items():
            bat_avg[(str(team), int(s))] = float(m)
        for team, m in prior.groupby("bowling_team")["pp_total"].mean().items():
            bowl_avg[(str(team), int(s))] = float(m)
    return bat_avg, bowl_avg


def venue_par_dict(balls: pd.DataFrame) -> dict[tuple[str, int], float]:
    rows = []
    for (mid, innings), g in balls.groupby(["match_id", "innings"]):
        g = g.sort_values(["over", "ball", "is_legal_delivery"], ascending=[True, True, False]).reset_index(drop=True)
        legal_idx = np.where(g["is_legal_delivery"].values)[0]
        if len(legal_idx) < 36:
            continue
        cut_36 = legal_idx[35] + 1
        rows.append({
            "venue": g["venue"].iloc[0],
            "season": int(g["season"].iloc[0]),
            "pp_total": int(g.iloc[:cut_36]["runs_total"].sum()),
        })
    pp_df = pd.DataFrame(rows)
    seasons = sorted(pp_df["season"].unique())
    par: dict[tuple[str, int], float] = {}
    for s in seasons:
        prior = pp_df[pp_df["season"] < s]
        if prior.empty:
            continue
        for venue, m in prior.groupby("venue")["pp_total"].mean().items():
            par[(str(venue), int(s))] = float(m)
    return par


def main() -> int:
    configure_logging()
    df = pd.read_parquet("data/processed/pp_time_anchors.parquet")
    logger.info("loaded %d runner-level rows", len(df))

    leagues = ["ipl", "bbl", "psl", "cpl", "ntb"]
    all_balls = []
    for league in leagues:
        d = Path("data/processed") if league == "ipl" else Path(f"data/processed/{league}")
        b = read_balls(d)
        if not b.empty:
            all_balls.append(b)
    balls = pd.concat(all_balls, ignore_index=True)

    logger.info("computing per-match metadata (batting/bowling team, venue, season)")
    bat_team = innings_batting_team(balls)
    venue_map = venue_for_match(balls)
    season_map = season_for_match(balls)

    logger.info("computing leak-free team PP rates")
    bat_pp, bowl_pp = team_pp_rates(balls)
    logger.info("computing leak-free venue PP par")
    venue_par = venue_par_dict(balls)

    LEAGUE_DEFAULT_PP = 50.0

    def lookup_bat(team: str, season: int) -> float:
        return bat_pp.get((team, season), LEAGUE_DEFAULT_PP)
    def lookup_bowl(team: str, season: int) -> float:
        return bowl_pp.get((team, season), LEAGUE_DEFAULT_PP)
    def lookup_venue(venue: str, season: int) -> float:
        return venue_par.get((venue, season), LEAGUE_DEFAULT_PP)

    df["season"] = df["match_id"].astype(str).map(season_map)
    df["venue"] = df["match_id"].astype(str).map(venue_map)
    df["batting_team"] = df.apply(
        lambda r: bat_team.get((r["match_id"], int(r["innings"])), ""), axis=1
    )
    df["bowling_team"] = df.apply(
        lambda r: next(
            (t for (m, inn), t in bat_team.items() if m == r["match_id"] and inn != int(r["innings"])),
            "",
        ), axis=1,
    )
    df["bat_pp_prior"] = df.apply(lambda r: lookup_bat(r["batting_team"], int(r["season"])), axis=1)
    df["bowl_pp_prior"] = df.apply(lambda r: lookup_bowl(r["bowling_team"], int(r["season"])), axis=1)
    df["venue_par"] = df.apply(lambda r: lookup_venue(r["venue"], int(r["season"])), axis=1)

    # Use opening price (largest sample) for training; T-1 for evaluation
    df = df.dropna(subset=["first_ltp"])
    df = df[df["first_ltp"] > 1.0].copy()
    df["implied_open"] = 1.0 / df["first_ltp"]
    df["implied_t1"] = np.where(df["ltp_t_minus_1"].notna() & (df["ltp_t_minus_1"] > 1.0),
                                 1.0 / df["ltp_t_minus_1"], np.nan)

    df["x_minus_par"] = df["threshold_X"] - df["venue_par"]
    df["x_minus_bat"] = df["threshold_X"] - df["bat_pp_prior"]
    df["x_minus_bowl"] = df["threshold_X"] - df["bowl_pp_prior"]
    for L in leagues:
        df[f"is_{L}"] = (df["match_id"].astype(str).map(
            {mid: lg for lg, dirpath in [
                (lg, Path("data/processed") if lg=="ipl" else Path(f"data/processed/{lg}"))
                for lg in leagues
            ] for mid in pd.read_parquet(dirpath / "matches.parquet")["match_id"].astype(str)}
        ) == L).astype(int)

    feature_cols = [
        "threshold_X", "implied_open",
        "bat_pp_prior", "bowl_pp_prior", "venue_par",
        "x_minus_par", "x_minus_bat", "x_minus_bowl",
        "innings",
        "is_ipl", "is_bbl", "is_psl", "is_cpl", "is_ntb",
    ]

    df = df.dropna(subset=["season", "actual_over_X"] + feature_cols)
    logger.info("model dataset: %d rows", len(df))

    # Walk-forward: hold out the 2 most recent seasons per league as test
    seasons_sorted = sorted(df["season"].unique())
    cutoff = seasons_sorted[-2]
    train = df[df["season"] < cutoff].copy()
    test = df[df["season"] >= cutoff].copy()
    logger.info("train=%d, test=%d, cutoff_season=%s", len(train), len(test), cutoff)

    x_train = train[feature_cols].to_numpy(dtype=np.float32)
    y_train = train["actual_over_X"].to_numpy(dtype=np.float32)
    x_test = test[feature_cols].to_numpy(dtype=np.float32)
    y_test = test["actual_over_X"].to_numpy(dtype=np.float32)

    gbm = lgb.train(
        params={
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 100,
            "verbose": -1,
            "seed": 17,
            "deterministic": True,
        },
        train_set=lgb.Dataset(x_train, label=y_train, feature_name=feature_cols),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(x_test, label=y_test, feature_name=feature_cols)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    test = test.assign(model_p=gbm.predict(x_test))
    market_ll = log_loss(y_test, np.clip(test["implied_open"], 1e-3, 1 - 1e-3))
    model_ll = log_loss(y_test, np.clip(test["model_p"], 1e-3, 1 - 1e-3))
    logger.info("test log-loss   market: %.4f   model: %.4f   (lower is better)",
                market_ll, model_ll)

    test["edge"] = test["model_p"] - test["implied_open"]

    # Baseline: lay every mid-bin runner uniformly
    mid = test[(test["implied_open"] >= 0.40) & (test["implied_open"] < 0.50)].copy()
    print(f"\n=== Baseline: uniform lay mid-bin (implied 0.40-0.50), n={len(mid)} ===")
    pnl_baseline = backtest_lays(mid)
    print_stats(pnl_baseline)

    # Selective: lay mid-bin runners where model edge < -0.03 (model says less likely than market)
    for thresh in [-0.02, -0.04, -0.06, -0.08]:
        selective = mid[mid["edge"] < thresh].copy()
        print(f"\n=== Selective: mid-bin + model_edge < {thresh}, n={len(selective)} ===")
        if selective.empty:
            print("  (no trades)")
            continue
        pnl = backtest_lays(selective)
        print_stats(pnl)

    # Cross-bin selective (across full implied range, just model edge cutoff)
    for thresh in [-0.05, -0.10]:
        sub = test[test["edge"] < thresh].copy()
        # Restrict to plausible-liquidity prices (0.10 - 0.90)
        sub = sub[(sub["implied_open"] >= 0.10) & (sub["implied_open"] <= 0.90)]
        print(f"\n=== All-bins selective: model_edge < {thresh}, implied 0.10-0.90, n={len(sub)} ===")
        if sub.empty:
            print("  (no trades)")
            continue
        pnl = backtest_lays(sub)
        print_stats(pnl)
    return 0


def backtest_lays(df: pd.DataFrame, stake: float = 100.0, commission: float = 0.05) -> pd.DataFrame:
    """Each row: lay '>= X' runner at decimal odds 1/implied_open, stake `stake`.
    Win stake*(1-commission) if NOT (actual >= X); lose stake*(price-1) if it IS.
    """
    df = df.copy()
    price = 1.0 / df["implied_open"]
    won = df["actual_over_X"] == 0  # lay wins when 'X or more' does NOT happen
    df["pnl"] = np.where(won, stake * (1 - commission), -stake * (price - 1.0))
    return df


def print_stats(trades: pd.DataFrame, stake: float = 100.0) -> None:
    n = len(trades)
    if n == 0:
        print("  no trades")
        return
    pnl = trades["pnl"].sum()
    win_rate = (trades["pnl"] > 0).mean()
    avg_win = trades.loc[trades["pnl"] > 0, "pnl"].mean() if (trades["pnl"] > 0).any() else 0
    avg_loss = trades.loc[trades["pnl"] < 0, "pnl"].mean() if (trades["pnl"] < 0).any() else 0
    print(f"  n={n}  win_rate={win_rate:.3f}  total_pnl={pnl:+.0f}  avg_win={avg_win:+.1f}  avg_loss={avg_loss:+.1f}")
    print(f"  ROI per stake = {pnl / (n * stake) * 100:+.2f}%")


if __name__ == "__main__":
    raise SystemExit(main())

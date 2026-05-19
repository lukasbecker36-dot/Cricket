"""P&L estimates for the full_innings selectivity strategy.

Re-runs the triple-fold validation for the full_innings market, captures
every test-set trade, and simulates two bankroll managers:
  - Flat stake (£10 / £25 / £100 per trade)
  - Fractional Kelly (full / half / quarter)

Reports total P&L, hit rate, max drawdown, bankroll path, and the
distribution of stake sizes under Kelly.
"""
from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.ingestion.storage import read_balls
from src.logging_setup import configure_logging

logger = logging.getLogger(__name__)


COMMISSION = 0.05


def build_team_stats_full_innings(balls: pd.DataFrame):
    rows = []
    for (mid, innings), g in balls.groupby(["match_id", "innings"]):
        g = g.sort_values(["over", "ball", "is_legal_delivery"], ascending=[True, True, False]).reset_index(drop=True)
        legal_idx = np.where(g["is_legal_delivery"].values)[0]
        if len(legal_idx) < 30:
            continue
        total = int(g["runs_total"].sum())
        rows.append({
            "season": int(g["season"].iloc[0]),
            "batting_team": g["batting_team"].iloc[0],
            "bowling_team": g["bowling_team"].iloc[0],
            "venue": g["venue"].iloc[0],
            "total": total,
        })
    pp_df = pd.DataFrame(rows)
    bat, bowl, ven = {}, {}, {}
    for s in sorted(pp_df["season"].unique()):
        prior = pp_df[pp_df["season"] < s]
        if prior.empty: continue
        for t, m in prior.groupby("batting_team")["total"].mean().items():
            bat[(str(t), int(s))] = float(m)
        for t, m in prior.groupby("bowling_team")["total"].mean().items():
            bowl[(str(t), int(s))] = float(m)
        for v, m in prior.groupby("venue")["total"].mean().items():
            ven[(str(v), int(s))] = float(m)
    return bat, bowl, ven


def lay_outcome(stake: float, market_implied: float, won: bool, commission: float = COMMISSION) -> float:
    """Return PnL for a lay trade. stake = the matched stake (= max win)."""
    price = 1.0 / market_implied
    if won:
        return stake * (1.0 - commission)
    return -stake * (price - 1.0)


def kelly_fraction_lay(p_model: float, market_implied: float, commission: float = COMMISSION) -> float:
    """Optimal Kelly fraction of bankroll to stake on a lay.

    For a lay at price P with stake S:
      Win  (prob 1-p_true): profit S*(1-c)
      Lose (prob p_true):   loss   S*(P-1)
    Kelly: f* = ((1-p)*a - p*b) / (a*b), where a=1-c, b=P-1.
    Returns 0 if no edge.
    """
    if market_implied <= 0 or market_implied >= 1:
        return 0.0
    price = 1.0 / market_implied
    a = 1.0 - commission   # profit factor on win
    b = price - 1.0        # loss factor on lose
    if b <= 0:
        return 0.0
    p = float(p_model)
    f = ((1.0 - p) * a - p * b) / (a * b)
    return max(0.0, f)


def simulate_flat(trades: pd.DataFrame, stake: float) -> dict:
    bankroll = 1000.0
    history = [bankroll]
    pnls = []
    for r in trades.itertuples(index=False):
        pnl = lay_outcome(stake, float(r.market_implied), bool(r.won))
        bankroll += pnl
        history.append(bankroll)
        pnls.append(pnl)
    hist = np.array(history)
    drawdown = float((np.maximum.accumulate(hist) - hist).max())
    return {
        "n": len(trades),
        "total_pnl": float(sum(pnls)),
        "win_rate": float(np.mean([p > 0 for p in pnls])),
        "final_bankroll": float(bankroll),
        "max_dd": drawdown,
        "history": hist,
    }


def simulate_kelly(trades: pd.DataFrame, fraction: float, *, cap: float = 0.05, start: float = 1000.0) -> dict:
    """fraction: 1.0 = full Kelly, 0.25 = quarter, etc. cap = max stake fraction of bankroll."""
    bankroll = start
    history = [bankroll]
    pnls = []
    stakes = []
    for r in trades.itertuples(index=False):
        f_star = kelly_fraction_lay(float(r.model_p), float(r.market_implied))
        f = min(f_star * fraction, cap)
        if f <= 0 or bankroll <= 0:
            history.append(bankroll)
            continue
        stake = f * bankroll
        pnl = lay_outcome(stake, float(r.market_implied), bool(r.won))
        bankroll += pnl
        history.append(bankroll)
        pnls.append(pnl)
        stakes.append(stake)
    hist = np.array(history)
    drawdown = float((np.maximum.accumulate(hist) - hist).max())
    return {
        "n": len(pnls),
        "total_pnl": float(sum(pnls)),
        "win_rate": float(np.mean([p > 0 for p in pnls])) if pnls else 0.0,
        "final_bankroll": float(bankroll),
        "max_dd": drawdown,
        "mean_stake": float(np.mean(stakes)) if stakes else 0.0,
        "median_stake": float(np.median(stakes)) if stakes else 0.0,
        "max_stake": float(max(stakes)) if stakes else 0.0,
        "history": hist,
    }


def main() -> int:
    configure_logging()
    df = pd.read_parquet("data/processed/all_innings_markets.parquet")
    df = df[df["market_type"] == "full_innings"].copy()
    logger.info("full_innings rows: %d", len(df))

    leagues = ["ipl", "bbl", "psl", "cpl", "ntb"]
    all_balls = []
    for league in leagues:
        d = Path("data/processed") if league == "ipl" else Path(f"data/processed/{league}")
        b = read_balls(d)
        if not b.empty: all_balls.append(b)
    balls = pd.concat(all_balls, ignore_index=True)
    season_map = balls[["match_id", "season"]].drop_duplicates("match_id").set_index("match_id")["season"].astype(int).to_dict()
    venue_map = balls[["match_id", "venue"]].drop_duplicates("match_id").set_index("match_id")["venue"].to_dict()
    bat_team_map = {(mid, int(inn)): g["batting_team"].iloc[0]
                    for (mid, inn), g in balls.groupby(["match_id", "innings"])}

    bat_pp, bowl_pp, ven_par = build_team_stats_full_innings(balls)
    DEFAULT = float(np.mean(list(ven_par.values()))) if ven_par else 150.0

    df["season"] = df["match_id"].astype(str).map(season_map)
    df["venue"]  = df["match_id"].astype(str).map(venue_map)
    df["batting_team"] = df.apply(lambda r: bat_team_map.get((r["match_id"], int(r["innings"])), ""), axis=1)
    df["bowling_team"] = df.apply(
        lambda r: next(
            (t for (m, inn), t in bat_team_map.items() if m == r["match_id"] and inn != int(r["innings"])), ""), axis=1)
    df["bat_prior"]  = df.apply(lambda r: bat_pp.get((r["batting_team"], int(r["season"])), DEFAULT), axis=1)
    df["bowl_prior"] = df.apply(lambda r: bowl_pp.get((r["bowling_team"], int(r["season"])), DEFAULT), axis=1)
    df["venue_par"]  = df.apply(lambda r: ven_par.get((r["venue"], int(r["season"])), DEFAULT), axis=1)
    df = df.dropna(subset=["first_ltp"])
    df = df[df["first_ltp"] > 1.0].copy()
    df["implied_open"] = 1.0 / df["first_ltp"]
    df["implied_t1"]   = np.where(df["ltp_t_minus_1"].notna() & (df["ltp_t_minus_1"] > 1.0),
                                   1.0 / df["ltp_t_minus_1"], np.nan)
    df["x_minus_par"]  = df["threshold_X"] - df["venue_par"]
    df["x_minus_bat"]  = df["threshold_X"] - df["bat_prior"]
    df["x_minus_bowl"] = df["threshold_X"] - df["bowl_prior"]
    for L in leagues:
        df[f"is_{L}"] = (df["league"] == L).astype(int)

    features = [
        "threshold_X", "implied_open",
        "bat_prior", "bowl_prior", "venue_par",
        "x_minus_par", "x_minus_bat", "x_minus_bowl",
        "innings",
    ] + [f"is_{L}" for L in leagues]
    df = df.dropna(subset=["season", "actual_over_X"] + features)

    seasons_sorted = sorted(df["season"].unique())
    val_season = seasons_sorted[-2]
    test_season = seasons_sorted[-1]
    train_df = df[df["season"] <  val_season].copy()
    val_df   = df[df["season"] == val_season].copy()
    test_df  = df[df["season"] == test_season].copy()
    logger.info("train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df))

    gbm = lgb.train(
        params={
            "objective": "binary", "metric": "binary_logloss",
            "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 100,
            "verbose": -1, "seed": 17, "deterministic": True,
        },
        train_set=lgb.Dataset(train_df[features].to_numpy(dtype=np.float32),
                              label=train_df["actual_over_X"].to_numpy(dtype=np.float32),
                              feature_name=features),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(val_df[features].to_numpy(dtype=np.float32),
                                label=val_df["actual_over_X"].to_numpy(dtype=np.float32),
                                feature_name=features)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    THRESHOLD = -0.03  # validated separately
    test_df["model_p"] = gbm.predict(test_df[features].to_numpy(dtype=np.float32))
    test_df["edge_open"] = test_df["model_p"] - test_df["implied_open"]

    # Trade selection (OPEN-price selection, same as hardening script)
    trades = test_df[(test_df["edge_open"] < THRESHOLD) &
                     (test_df["implied_open"] >= 0.10) & (test_df["implied_open"] <= 0.90)].copy()
    trades = trades.rename(columns={"implied_open": "market_implied"})
    trades["won"] = trades["actual_over_X"] == 0  # lay wins when 'X or more' does NOT happen
    logger.info("selected trades: %d", len(trades))

    # --- Flat-stake P&L at different sizes -----------------------------
    print(f"\n===== Flat-stake P&L on {len(trades)} test-set trades (season {test_season}) =====")
    print(f"{'stake':>8} {'total_pnl':>10} {'win_rate':>9} {'max_dd':>9} {'final_bankroll':>16}")
    for stake in [5.0, 10.0, 25.0, 50.0, 100.0]:
        r = simulate_flat(trades, stake=stake)
        print(f"  £{stake:>5.0f} £{r['total_pnl']:>+8.2f} {r['win_rate']:>9.3f} £{r['max_dd']:>7.2f}  £{r['final_bankroll']:>+13.2f}")

    # --- Kelly sizing -----------------------------------------------------
    print(f"\n===== Kelly sizing P&L (starting bankroll £1000) =====")
    print(f"{'kelly_mult':>12} {'cap%':>5} {'total_pnl':>11} {'mean_stake':>11} {'max_stake':>10} {'max_dd':>9} {'final':>11}")
    for mult, cap in [(1.0, 0.05), (0.5, 0.05), (0.25, 0.05), (0.1, 0.05), (0.25, 0.02)]:
        r = simulate_kelly(trades, fraction=mult, cap=cap)
        print(f"  {mult:>10.2f}x  {cap*100:>4.1f}%  £{r['total_pnl']:>+8.2f} £{r['mean_stake']:>+9.2f} £{r['max_stake']:>+8.2f} £{r['max_dd']:>+7.2f}  £{r['final_bankroll']:>+9.2f}")

    # Kelly distribution analysis
    f_stars = [kelly_fraction_lay(float(r.model_p), float(r.market_implied)) for r in trades.itertuples(index=False)]
    f_stars = np.array(f_stars)
    print(f"\nKelly fraction distribution (per-trade f*):")
    print(f"  mean: {f_stars.mean():.4f}   median: {np.median(f_stars):.4f}   max: {f_stars.max():.4f}")
    print(f"  trades where f* >= 0.05 (cap-binding): {(f_stars >= 0.05).sum()} / {len(f_stars)}")
    print(f"  trades where f* >= 0.10:               {(f_stars >= 0.10).sum()} / {len(f_stars)}")
    print(f"  trades where f* >= 0.20:               {(f_stars >= 0.20).sum()} / {len(f_stars)}")

    # Trade distribution
    print(f"\nMarket implied probability distribution of selected trades:")
    print(f"  mean: {trades['market_implied'].mean():.3f}   median: {trades['market_implied'].median():.3f}")
    print(f"  decile breakdown:")
    print(trades["market_implied"].describe(percentiles=[.1,.25,.5,.75,.9]).round(3).to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

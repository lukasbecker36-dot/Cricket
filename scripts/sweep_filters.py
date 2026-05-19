"""Sweep edge threshold and phase filter for one or more leagues.

Reads pre-saved aligned predictions (output of eval_external or
train_and_evaluate when Betfair data is available) and runs the backtest
across combinations of (edge_threshold, ball_index range). Prints a grid
of ROI / pnl / n_trades per league.

Usage:
    python -m scripts.sweep_filters --leagues ipl bbl psl cpl ntb
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import BacktestConfig, Config
from src.logging_setup import configure_logging
from src.validation.backtest import run_backtest

logger = logging.getLogger(__name__)


THRESHOLDS = [0.05, 0.075, 0.10, 0.15, 0.20]
PHASE_FILTERS = [
    # (name, ball_min, ball_max)  inclusive
    ("full       ",   6, 120),
    ("skip-PP    ",  36, 120),
    ("skip-late  ",   6,  90),
    ("middle     ",  36,  90),
    ("middle-only",  42,  78),
]


def predictions_path(league: str) -> Path:
    if league == "ipl":
        return Path("data/processed/eval_predictions.parquet")
    return Path(f"data/processed/{league}/eval_predictions.parquet")


def sweep_one(league: str, cfg: Config) -> pd.DataFrame:
    path = predictions_path(league)
    if not path.exists():
        logger.warning("%s: %s missing -- run eval_external --league %s first",
                       league.upper(), path, league)
        return pd.DataFrame()
    df = pd.read_parquet(path).reset_index(drop=True)
    rows = []
    for fname, b_min, b_max in PHASE_FILTERS:
        sub = df[(df["ball_index"] >= b_min) & (df["ball_index"] <= b_max)].reset_index(drop=True)
        if sub.empty:
            continue
        market_implied = sub["market_prob"].astype(float)
        for thr in THRESHOLDS:
            bt_cfg = BacktestConfig(
                commission=cfg.backtest.commission,
                slippage_ticks=cfg.backtest.slippage_ticks,
                decision_lag_seconds=cfg.backtest.decision_lag_seconds,
                edge_threshold=thr,
                min_liquidity_gbp=cfg.backtest.min_liquidity_gbp,
            )
            res = run_backtest(sub, bt_cfg, market_implied=market_implied)
            rows.append({
                "league": league,
                "filter": fname.strip(),
                "ball_range": f"{b_min}-{b_max}",
                "thr": thr,
                "n_trades": res.n_trades,
                "pnl": round(res.total_pnl, 0),
                "roi": round(res.roi, 4),
                "max_dd": round(res.max_drawdown, 0),
            })
    return pd.DataFrame(rows)


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--leagues", nargs="+", default=["ipl", "bbl", "psl", "cpl", "ntb"],
    )
    args = parser.parse_args()
    cfg = Config.load(args.config)

    all_rows: list[pd.DataFrame] = []
    for league in args.leagues:
        df = sweep_one(league, cfg)
        if df.empty:
            continue
        all_rows.append(df)
        print(f"\n=== {league.upper()} ===")
        print(df[["filter", "ball_range", "thr", "n_trades", "pnl", "roi"]].to_string(index=False))

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        out_path = Path("data/processed/sweep_results.parquet")
        combined.to_parquet(out_path, index=False)
        logger.info("wrote sweep results: %s (%d rows)", out_path, len(combined))

        # Pivot: rows = (filter, thr), columns = league, values = roi
        print("\n=== ROI grid (rows=filter@thr, cols=league) ===")
        pivot = combined.pivot_table(
            index=["filter", "thr"], columns="league", values="roi", aggfunc="first",
        )
        print(pivot.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

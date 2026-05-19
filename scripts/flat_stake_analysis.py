"""Decompose flat-stake performance per league: hit rate, win/loss sizes, EV.

For each league's saved predictions, runs the same flat-stake backtest the
project uses, then computes:
  - hit_rate: fraction of trades that won
  - avg_win, avg_loss: mean PnL on winning/losing trades
  - ev_per_trade: average PnL across all trades
  - ROI: ev_per_trade / stake

Aggregates across leagues to answer: are losses catastrophic, or is the
edge just too small to matter?
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.logging_setup import configure_logging
from src.validation.backtest import run_backtest

logger = logging.getLogger(__name__)


def analyse_league(league: str, cfg: Config, stake: float) -> dict:
    path = (
        Path("data/processed/eval_predictions.parquet") if league == "ipl"
        else Path(f"data/processed/{league}/eval_predictions.parquet")
    )
    if not path.exists():
        return {}
    df = pd.read_parquet(path).reset_index(drop=True)
    market_implied = df["market_prob"].astype(float).reset_index(drop=True)
    bt = run_backtest(df, cfg.backtest, market_implied=market_implied, stake=stake)
    if bt.trades.empty:
        return {"league": league.upper(), "n": 0}
    wins = bt.trades[bt.trades["pnl"] > 0]
    losses = bt.trades[bt.trades["pnl"] < 0]
    return {
        "league": league.upper(),
        "n": len(bt.trades),
        "hit_rate": len(wins) / len(bt.trades),
        "avg_win": wins["pnl"].mean() if len(wins) else 0.0,
        "avg_loss": losses["pnl"].mean() if len(losses) else 0.0,  # negative
        "ev_per_trade": bt.trades["pnl"].mean(),
        "roi": bt.roi,
        "total_pnl": bt.total_pnl,
        "max_dd": bt.max_drawdown,
    }


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--leagues", nargs="+", default=["ipl", "bbl", "psl", "cpl", "ntb"])
    parser.add_argument("--stake", type=float, default=100.0)
    args = parser.parse_args()
    cfg = Config.load("configs/default.yaml")

    rows = [analyse_league(L, cfg, args.stake) for L in args.leagues]
    rows = [r for r in rows if r]
    df = pd.DataFrame(rows)
    print("\n=== Flat-stake decomposition per league ===")
    print(df[[
        "league", "n", "hit_rate", "avg_win", "avg_loss",
        "ev_per_trade", "roi", "total_pnl"
    ]].round(3).to_string(index=False))

    # Aggregate across leagues
    totals = {
        "n_trades": int(df["n"].sum()),
        "total_pnl": float(df["total_pnl"].sum()),
        "total_stake": float(df["n"].sum() * args.stake),
    }
    totals["aggregate_roi"] = totals["total_pnl"] / totals["total_stake"]

    # Aggregate excluding IPL (which has the suspected alignment leak)
    not_ipl = df[df["league"] != "IPL"]
    totals_ex_ipl = {
        "n_trades": int(not_ipl["n"].sum()),
        "total_pnl": float(not_ipl["total_pnl"].sum()),
        "total_stake": float(not_ipl["n"].sum() * args.stake),
    }
    totals_ex_ipl["aggregate_roi"] = (
        totals_ex_ipl["total_pnl"] / totals_ex_ipl["total_stake"]
        if totals_ex_ipl["total_stake"] else 0.0
    )

    print("\n=== Aggregate ===")
    print(f"All 5 leagues:        n={totals['n_trades']:,} pnl={totals['total_pnl']:+.0f} "
          f"ROI={totals['aggregate_roi']*100:+.2f}%")
    print(f"Excluding IPL (leak): n={totals_ex_ipl['n_trades']:,} pnl={totals_ex_ipl['total_pnl']:+.0f} "
          f"ROI={totals_ex_ipl['aggregate_roi']*100:+.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

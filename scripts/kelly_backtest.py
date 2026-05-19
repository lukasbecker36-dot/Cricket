"""Kelly-criterion sizing applied to model edges over market price.

For each (model_p, market_p) pair we compute the Kelly fraction:
  edge = model_p * price - 1      (for a back, when model > market)
  f*   = edge / (price - 1)

If model > market we back at the chasing-team price; if model < market we lay.
We simulate a compounding bankroll. Settlement uses the match outcome (y);
commission is applied to winnings; slippage worsens the price by `slippage_ticks`.

Kelly is profitable only when the model has edge over the market. This script
DEMONSTRATES that: leagues where flat-stake ROI was positive (IPL) compound to
gains; leagues where flat-stake was negative (NTB) compound to losses.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.logging_setup import configure_logging
from src.validation.backtest import apply_slippage, prob_to_price

logger = logging.getLogger(__name__)


def kelly_fraction(model_p: float, market_p: float, side: str) -> float:
    """Optimal Kelly fraction for a back (model_p > market_p) or lay (<)."""
    market_p = max(min(market_p, 0.999), 0.001)
    model_p = max(min(model_p, 0.999), 0.001)
    if side == "back":
        decimal_odds = 1.0 / market_p
        edge = model_p * decimal_odds - 1.0
        if edge <= 0:
            return 0.0
        return edge / (decimal_odds - 1.0)
    # lay: bet against the chasing team. Equivalent to backing the other team
    # at decimal odds 1 / (1 - market_p). Model belief in 'other team wins' = 1 - model_p.
    p_other_model = 1.0 - model_p
    p_other_market = 1.0 - market_p
    decimal_odds = 1.0 / p_other_market
    edge = p_other_model * decimal_odds - 1.0
    if edge <= 0:
        return 0.0
    return edge / (decimal_odds - 1.0)


def simulate(
    predictions: pd.DataFrame,
    *,
    kelly_fraction_cap: float,
    kelly_multiplier: float,  # 1.0 = full Kelly, 0.5 = half-Kelly (less volatile, more realistic)
    starting_bankroll: float,
    commission: float,
    slippage_ticks: int,
    min_edge: float,
) -> dict:
    """Run a compounding Kelly sim. predictions has columns p, market_prob, y."""
    bankroll = starting_bankroll
    history = [bankroll]
    n_trades = 0
    n_wins = 0
    for row in predictions.itertuples(index=False):
        model_p = float(row.p)
        market_p = float(row.market_prob)
        if pd.isna(market_p):
            history.append(bankroll); continue
        edge = model_p - market_p
        if abs(edge) < min_edge:
            history.append(bankroll); continue
        side = "back" if edge > 0 else "lay"
        f = kelly_fraction(model_p, market_p, side)
        if f <= 0:
            history.append(bankroll); continue
        f = min(f * kelly_multiplier, kelly_fraction_cap)
        stake = f * bankroll
        chasing_price = prob_to_price(market_p)
        if side == "back":
            exec_price = apply_slippage(chasing_price, "back", slippage_ticks)
            won = int(row.y) == 1
            gross = stake * (exec_price - 1.0) if won else -stake
        else:
            other_price = 1.0 / max(1.0 - market_p, 0.001)
            exec_price = apply_slippage(other_price, "back", slippage_ticks)
            won = int(row.y) == 0
            gross = stake * (exec_price - 1.0) if won else -stake
        if gross > 0:
            gross *= 1.0 - commission
        bankroll += gross
        history.append(bankroll)
        n_trades += 1
        if gross > 0:
            n_wins += 1
        if bankroll <= 0:
            logger.warning("bankroll went to zero")
            break
    hist = np.array(history)
    drawdown = (np.maximum.accumulate(hist) - hist).max()
    return {
        "n_trades": n_trades,
        "n_wins": n_wins,
        "hit_rate": n_wins / max(n_trades, 1),
        "starting_bankroll": starting_bankroll,
        "final_bankroll": float(bankroll),
        "return": float(bankroll / starting_bankroll - 1.0),
        "max_drawdown": float(drawdown),
    }


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--leagues", nargs="+", default=["ipl", "bbl", "psl", "cpl", "ntb"])
    parser.add_argument("--starting-bankroll", type=float, default=1000.0)
    parser.add_argument("--kelly-multiplier", type=float, default=0.25,
                        help="0.25 = quarter Kelly (standard; full Kelly is impractical)")
    parser.add_argument("--kelly-cap", type=float, default=0.05,
                        help="hard cap on any single bet as fraction of bankroll")
    parser.add_argument("--commission", type=float, default=0.05)
    parser.add_argument("--slippage-ticks", type=int, default=1)
    parser.add_argument("--min-edge", type=float, default=0.05)
    args = parser.parse_args()

    rows = []
    for league in args.leagues:
        path = (
            Path("data/processed/eval_predictions.parquet") if league == "ipl"
            else Path(f"data/processed/{league}/eval_predictions.parquet")
        )
        if not path.exists():
            logger.warning("%s: no predictions parquet -- run eval_external/train_and_evaluate first", league)
            continue
        df = pd.read_parquet(path)
        res = simulate(
            df,
            kelly_fraction_cap=args.kelly_cap,
            kelly_multiplier=args.kelly_multiplier,
            starting_bankroll=args.starting_bankroll,
            commission=args.commission,
            slippage_ticks=args.slippage_ticks,
            min_edge=args.min_edge,
        )
        res["league"] = league.upper()
        rows.append(res)
        logger.info(
            "%-5s trades=%d hit=%.1f%% start=%.0f final=%.0f return=%+.1f%% maxDD=%.0f",
            league.upper(), res["n_trades"], res["hit_rate"]*100,
            res["starting_bankroll"], res["final_bankroll"],
            res["return"]*100, res["max_drawdown"],
        )

    print(f"\n=== Kelly backtest summary "
          f"(quarter-Kelly, cap {args.kelly_cap*100:.0f}% per bet, "
          f"min edge {args.min_edge}, {args.commission*100:.0f}% commission) ===")
    print(pd.DataFrame(rows)[
        ["league", "n_trades", "hit_rate", "starting_bankroll", "final_bankroll", "return", "max_drawdown"]
    ].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

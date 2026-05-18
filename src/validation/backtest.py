"""Backtest with realistic costs. Per CLAUDE.md: a backtest without these is not trustworthy.

We don't have real Betfair tick data yet; for v1 we *simulate* a counterfactual market
that prices the true win probability with random noise. This is a placeholder that
demonstrates the cost model — replace `simulate_market` with real Betfair price data
once available. The cost accounting itself is real.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import BacktestConfig

logger = logging.getLogger(__name__)


def betfair_tick_size(price: float) -> float:
    """Betfair price ladder."""
    if price < 2.0:
        return 0.01
    if price < 3.0:
        return 0.02
    if price < 4.0:
        return 0.05
    if price < 6.0:
        return 0.1
    if price < 10.0:
        return 0.2
    if price < 20.0:
        return 0.5
    if price < 30.0:
        return 1.0
    if price < 50.0:
        return 2.0
    if price < 100.0:
        return 5.0
    return 10.0


def prob_to_price(p: float) -> float:
    p = max(min(p, 0.999), 0.001)
    return 1.0 / p


def apply_slippage(price: float, side: str, ticks: int) -> float:
    """Worsen the price by `ticks` ticks. 'back' worse = lower price; 'lay' worse = higher."""
    out = price
    for _ in range(ticks):
        tick = betfair_tick_size(out)
        out = out - tick if side == "back" else out + tick
    return max(out, 1.01)


@dataclass
class BacktestResult:
    trades: pd.DataFrame  # match_id, ball_index, side, model_p, market_p, price, pnl
    total_pnl: float
    roi: float  # pnl / total_stake
    max_drawdown: float
    n_trades: int


def simulate_market(
    predictions: pd.DataFrame, rng: np.random.Generator, noise_std: float = 0.05
) -> pd.Series:
    """Placeholder market prices. Treats market as a noisy version of the true label,
    NOT the model — so any model edge over this is fictional. Replace with real Betfair
    history before drawing any conclusions about strategy ROI."""
    truth = predictions["y"].to_numpy(dtype=float)
    noise = rng.normal(0.0, noise_std, size=len(truth))
    implied = np.clip(truth * 0.85 + 0.075 + noise, 0.02, 0.98)
    return pd.Series(implied, index=predictions.index)


def run_backtest(
    predictions: pd.DataFrame,
    cfg: BacktestConfig,
    rng: np.random.Generator | None = None,
    stake: float = 100.0,
    market_implied: pd.Series | None = None,
) -> BacktestResult:
    """Apply edge threshold, place a flat-stake trade, settle with commission + slippage."""
    if rng is None:
        rng = np.random.default_rng(0)
    df = predictions.copy()
    df["market_p"] = (
        market_implied.values if market_implied is not None else simulate_market(df, rng).values
    )
    df["edge"] = df["p"] - df["market_p"]

    trades = []
    for row in df.itertuples(index=False):
        edge = float(row.edge)
        if abs(edge) < cfg.edge_threshold:
            continue
        side = "back" if edge > 0 else "lay"
        market_price = prob_to_price(float(row.market_p))
        exec_price = apply_slippage(market_price, side, cfg.slippage_ticks)
        won = int(row.y) == 1
        if side == "back":
            gross = stake * (exec_price - 1.0) if won else -stake
        else:
            # liability if lay loses (= true wins); profit = stake if true loses
            gross = stake if not won else -stake * (exec_price - 1.0)
        commission = max(gross, 0.0) * cfg.commission
        pnl = gross - commission
        trades.append(
            {
                "match_id": row.match_id,
                "ball_index": row.ball_index,
                "side": side,
                "model_p": float(row.p),
                "market_p": float(row.market_p),
                "exec_price": exec_price,
                "pnl": pnl,
            }
        )

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return BacktestResult(trades_df, 0.0, 0.0, 0.0, 0)

    equity = trades_df["pnl"].cumsum()
    drawdown = (equity.cummax() - equity).max()
    total_pnl = float(equity.iloc[-1])
    total_stake = stake * len(trades_df)
    roi = total_pnl / total_stake if total_stake else 0.0
    return BacktestResult(
        trades=trades_df,
        total_pnl=total_pnl,
        roi=roi,
        max_drawdown=float(drawdown),
        n_trades=len(trades_df),
    )

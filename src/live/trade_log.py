"""Persistent trade log for the match bot.

One row per logged trade. Trades start unsettled; once the match's phase
scores are known, settle() fills in the outcome and P&L. Stored as parquet
so it survives restarts and can be analysed later.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path("data/trades.parquet")
COMMISSION = 0.05


@dataclass
class Trade:
    logged_at_utc: str
    match_id: str
    event: str
    innings: int                 # 1 or 2
    phase: str                   # '6 over' / '10 over' / '15 over' / '20 over (full)'
    side: str                    # 'over' / 'under'
    line: float                  # the Betfair line the user traded
    odds: float                  # decimal odds matched
    stake: float
    model_par: float | None = None    # model's fair line at log time (context)
    settled: bool = False
    actual_total: float | None = None  # phase score at settlement
    won: bool | None = None
    pnl: float | None = None


PHASE_BALLS = {"6 over": 36, "10 over": 60, "15 over": 90, "20 over (full)": 120}


def _load(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=list(Trade.__annotations__.keys()))


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def log_trade(trade: Trade, path: Path = DEFAULT_PATH) -> None:
    if not trade.logged_at_utc:
        trade.logged_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    df = _load(path)
    df = pd.concat([df, pd.DataFrame([asdict(trade)])], ignore_index=True)
    _save(df, path)
    logger.info("logged trade: %s inn%d %s %s @ %.2f £%.0f",
                trade.event, trade.innings, trade.phase, trade.side, trade.odds, trade.stake)


def settle_match(match_id: str, phase_scores: dict[tuple[int, str], float],
                 path: Path = DEFAULT_PATH) -> list[Trade]:
    """Settle all unsettled trades for a match.

    phase_scores: {(innings, phase): actual_total_at_phase_boundary}
    Returns the trades that were settled this call.
    """
    df = _load(path)
    if df.empty:
        return []
    settled_now = []
    for i, row in df[(df["match_id"] == str(match_id)) & (~df["settled"].astype(bool))].iterrows():
        key = (int(row["innings"]), row["phase"])
        actual = phase_scores.get(key)
        if actual is None:
            continue
        won = (actual > row["line"]) if row["side"] == "over" else (actual < row["line"])
        # push (actual == line) shouldn't happen with .5 lines; treat as loss-of-stake void
        if actual == row["line"]:
            pnl = 0.0
            won = None
        else:
            pnl = row["stake"] * (row["odds"] - 1) * (1 - COMMISSION) if won else -row["stake"]
        df.at[i, "settled"] = True
        df.at[i, "actual_total"] = float(actual)
        df.at[i, "won"] = won
        df.at[i, "pnl"] = float(pnl)
        settled_now.append(Trade(**{k: df.at[i, k] for k in Trade.__annotations__}))
    _save(df, path)
    return settled_now


def summary(path: Path = DEFAULT_PATH) -> str:
    df = _load(path)
    if df.empty:
        return "No trades logged yet."
    settled = df[df["settled"].astype(bool)]
    n = len(df)
    n_settled = len(settled)
    pnl = settled["pnl"].sum() if n_settled else 0.0
    wins = int((settled["won"] == True).sum())  # noqa: E712
    staked = settled["stake"].sum() if n_settled else 0.0
    roi = (pnl / staked) if staked else 0.0
    lines = [
        f"*Trade log* — {n} trades ({n_settled} settled, {n - n_settled} open)",
        f"Settled P&L: £{pnl:+.2f}  ({wins}/{n_settled} won, ROI {roi:+.1%})",
    ]
    return "\n".join(lines)

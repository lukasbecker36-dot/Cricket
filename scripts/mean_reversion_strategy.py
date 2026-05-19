"""Mean-reversion strategy: lay a heavy favourite that emerged during innings 1.

Hypothesis: an early wicket in innings 1 can produce an overreaction in the
market, pushing one team's price below ~1.4 (heavy favourite). If the price
later mean-reverts as the batting team recovers, a lay-then-close-out trade
captures the difference.

This is a pure market-behaviour strategy. The win-probability model is not
used.

For each match we have Betfair LTP ticks for. We:
  1. Identify the innings-2 start by gap detection (same as backtest).
  2. Bound the innings-1 entry window: [marketTime + warm_up, innings_2_start).
  3. Take the first innings-1 tick where any runner's LTP <= entry_max_price
     (provided enough game-time has passed since marketTime).
  4. Track that runner's LTP from then on within innings 1.
  5. Exit:
       a. first tick where runner's LTP >= exit_min_price, OR
       b. tick where runner's LTP <= stop_price (optional stop-loss), OR
       c. close at the last innings-1 tick if neither.
  6. Compute closing-out PnL with commission and tick slippage.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.ingestion.betfair_loader import implied_market_probability
from src.ingestion.market_alignment import detect_innings2_window, market_time_ms
from src.logging_setup import configure_logging
from src.validation.backtest import apply_slippage

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    match_id: str
    season: int
    entry_time_ms: int
    entry_price: float
    exit_time_ms: int
    exit_price: float
    triggered_exit: bool  # True = hit exit_min_price; False = closed at end of innings 1
    pnl: float


def closing_lay_pnl(
    entry_price: float, exit_price: float, stake: float, commission: float
) -> float:
    """Profit when laying at entry_price then matching back at exit_price (fully
    hedged close-out). Profit = stake * (1 - entry_price / exit_price); commission
    is taken only on profits."""
    profit = stake * (1.0 - entry_price / exit_price)
    if profit > 0:
        profit *= 1.0 - commission
    return profit


def run_on_match(
    ticks: pd.DataFrame,
    market_time_iso: str,
    *,
    entry_max_price: float,
    exit_min_price: float,
    stop_price: float | None,
    warmup_ms: int,
    stake: float,
    commission: float,
    slippage_ticks: int,
) -> Trade | None:
    if ticks.empty:
        return None
    # Locate innings break to bound innings 1
    sel_ids = list(ticks["selection_id"].unique())
    if len(sel_ids) < 2:
        return None
    mt = market_time_ms(market_time_iso)
    prob = implied_market_probability(ticks, int(sel_ids[0]))
    innings2_start, _ = detect_innings2_window(prob, mt)
    if innings2_start is None:
        return None

    earliest_entry = mt + warmup_ms
    inn1 = ticks[
        (ticks["pt_ms"] >= earliest_entry) & (ticks["pt_ms"] < innings2_start)
    ].sort_values("pt_ms")
    if inn1.empty:
        return None

    entry_candidates = inn1[inn1["ltp"] <= entry_max_price]
    if entry_candidates.empty:
        return None
    entry = entry_candidates.iloc[0]
    entry_time = int(entry["pt_ms"])
    entry_runner = int(entry["selection_id"])
    raw_entry_price = float(entry["ltp"])
    # Slippage worsens lay entry (higher price = more liability)
    entry_price = apply_slippage(raw_entry_price, "lay", slippage_ticks)

    after = ticks[
        (ticks["selection_id"] == entry_runner)
        & (ticks["pt_ms"] > entry_time)
        & (ticks["pt_ms"] < innings2_start)
    ].sort_values("pt_ms")

    exit_row = None
    triggered = False
    for row in after.itertuples(index=False):
        if stop_price is not None and row.ltp <= stop_price:
            exit_row = row
            break
        if row.ltp >= exit_min_price:
            exit_row = row
            triggered = True
            break
    if exit_row is None:
        if after.empty:
            return None
        exit_row = after.iloc[-1] if not isinstance(after, pd.DataFrame) else after.iloc[-1]

    raw_exit_price = float(exit_row.ltp if hasattr(exit_row, "ltp") else exit_row["ltp"])
    # Slippage worsens back close-out (lower price = worse for back)
    exit_price = apply_slippage(raw_exit_price, "back", slippage_ticks)
    exit_time = int(exit_row.pt_ms if hasattr(exit_row, "pt_ms") else exit_row["pt_ms"])

    pnl = closing_lay_pnl(entry_price, exit_price, stake, commission)
    return Trade(
        match_id="",  # caller fills
        season=0,     # caller fills
        entry_time_ms=entry_time,
        entry_price=entry_price,
        exit_time_ms=exit_time,
        exit_price=exit_price,
        triggered_exit=triggered,
        pnl=pnl,
    )


def run_on_league(
    league: str,
    *,
    entry_max_price: float,
    exit_min_price: float,
    stop_price: float | None,
    warmup_minutes: int,
    stake: float,
    commission: float,
    slippage_ticks: int,
) -> list[Trade]:
    cricsheet_dir = Path("data/processed") if league == "ipl" else Path(f"data/processed/{league}")
    betfair_dir = cricsheet_dir / "betfair"
    join_path = cricsheet_dir / "betfair_join.parquet"
    if not join_path.exists() or not betfair_dir.exists():
        logger.warning("%s: no Betfair data; skipping", league)
        return []
    join_df = pd.read_parquet(join_path)
    cricsheet = pd.read_parquet(cricsheet_dir / "matches.parquet").set_index("match_id")

    trades: list[Trade] = []
    for row in join_df.itertuples(index=False):
        match_id = row.cricsheet_match_id
        season_row = cricsheet.loc[match_id] if match_id in cricsheet.index else None
        if season_row is None:
            continue
        season = int(season_row["season"])
        ticks_path = betfair_dir / f"season={season}" / f"{match_id}_ticks.parquet"
        if not ticks_path.exists():
            continue
        ticks = pd.read_parquet(ticks_path)
        market_time = row.market_time
        t = run_on_match(
            ticks, market_time,
            entry_max_price=entry_max_price,
            exit_min_price=exit_min_price,
            stop_price=stop_price,
            warmup_ms=warmup_minutes * 60_000,
            stake=stake,
            commission=commission,
            slippage_ticks=slippage_ticks,
        )
        if t is not None:
            t.match_id = match_id
            t.season = season
            trades.append(t)
    return trades


def summarise(trades: list[Trade], stake: float) -> dict[str, float]:
    if not trades:
        return {"n": 0}
    pnl_total = sum(t.pnl for t in trades)
    hit = sum(1 for t in trades if t.triggered_exit)
    stake_total = stake * len(trades)
    return {
        "n": len(trades),
        "n_triggered": hit,
        "hit_rate": hit / len(trades),
        "pnl": round(pnl_total, 0),
        "roi": round(pnl_total / stake_total, 4) if stake_total else 0.0,
        "avg_pnl_per_trade": round(pnl_total / len(trades), 2),
    }


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--leagues", nargs="+", default=["ipl", "bbl", "psl", "cpl", "ntb"])
    parser.add_argument("--entry-max-price", type=float, default=1.40)
    parser.add_argument("--exit-min-price", type=float, default=1.70)
    parser.add_argument("--stop-price", type=float, default=None)
    parser.add_argument("--warmup-minutes", type=int, default=10)
    parser.add_argument("--stake", type=float, default=100.0)
    parser.add_argument("--commission", type=float, default=0.05)
    parser.add_argument("--slippage-ticks", type=int, default=1)
    args = parser.parse_args()

    rows = []
    for league in args.leagues:
        trades = run_on_league(
            league,
            entry_max_price=args.entry_max_price,
            exit_min_price=args.exit_min_price,
            stop_price=args.stop_price,
            warmup_minutes=args.warmup_minutes,
            stake=args.stake,
            commission=args.commission,
            slippage_ticks=args.slippage_ticks,
        )
        s = summarise(trades, args.stake)
        s["league"] = league.upper()
        rows.append(s)
        if s["n"]:
            logger.info(
                "%-5s n=%d hit_rate=%.2f%% pnl=%.0f roi=%.3f avg=%.2f",
                league.upper(), s["n"], s["hit_rate"]*100, s["pnl"], s["roi"], s["avg_pnl_per_trade"],
            )
        else:
            logger.info("%-5s no trades", league.upper())

    print("\n=== Mean-reversion strategy summary ===")
    print(f"Entry: lay if LTP <= {args.entry_max_price}")
    print(f"Exit:  back if LTP >= {args.exit_min_price}; else close at end of innings 1")
    if args.stop_price:
        print(f"Stop:  close if LTP <= {args.stop_price}")
    print(f"Costs: {args.commission*100:.0f}% commission on winnings, {args.slippage_ticks}-tick slippage each leg")
    print()
    summary_df = pd.DataFrame(rows)[["league", "n", "n_triggered", "hit_rate", "pnl", "roi", "avg_pnl_per_trade"]]
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

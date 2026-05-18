"""Align Betfair market prices to model per-ball predictions.

Cricsheet ball data has no wall-clock timestamps, so we approximate. For each
match we identify the chase window in real time (heuristic from marketTime
plus innings-1 and break durations) and linearly map ball indices into that
window, taking the prevailing market price at each.

This is rough but honest: the alignment error per ball is on the order of
seconds to a minute, well within the 5-second decision-lag tolerance for
moderate-confidence trades.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .betfair_loader import (
    CHASE_OFFSET_END_MIN,
    CHASE_OFFSET_START_MIN,
    implied_market_probability,
)

logger = logging.getLogger(__name__)

# Decision lag: model output at time T cannot trade on prices at time T (CLAUDE.md).
# Our ball-index -> wall-time map is linear and accurate only to within minutes,
# so a 5s positive lag is meaningless. We instead use a NEGATIVE lag (we look at
# prices that are 60s STALE relative to our projected wall-time) so any timing
# uncertainty biases us toward older prices, not future ones. This is the
# rigorous direction: if model edge survives looking at stale prices, the edge
# is more likely real than a timing artefact.
DECISION_LAG_MS = -60_000


def chasing_selection_id(
    balls: pd.DataFrame, match_id: str, runners: dict[int, str]
) -> int | None:
    """Identify the Betfair selection_id for the chasing team in a match."""
    inn2 = balls[(balls["match_id"] == match_id) & (balls["innings"] == 2)]
    if inn2.empty:
        return None
    chasing_team = str(inn2["batting_team"].iloc[0])
    for sel_id, name in runners.items():
        if name == chasing_team:
            return int(sel_id)
    # Fall back to a substring match (handles 'Royal Challengers Bangalore' vs 'Bengaluru')
    for sel_id, name in runners.items():
        if name and chasing_team and (
            chasing_team.lower().startswith(name.lower()[:10])
            or name.lower().startswith(chasing_team.lower()[:10])
        ):
            return int(sel_id)
    return None


def market_time_ms(market_time_iso: str) -> int:
    """Parse ISO 8601 market start time to unix milliseconds."""
    dt = datetime.fromisoformat(market_time_iso.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


# Maximum plausible innings-2 wall duration. Chases usually finish in ~90 min;
# slow chases or rain-interrupted matches can stretch to ~110. Anything past
# this from the detected start is treated as post-match / settled and ignored.
INNINGS2_MAX_DURATION_MIN = 110
# A gap shorter than this within the match window is NOT an innings break.
MIN_GAP_FOR_BREAK_MIN = 5


def detect_innings2_window(
    prob: pd.DataFrame, market_time_anchor_ms: int
) -> tuple[int, int] | tuple[None, None]:
    """Identify innings-2 start/end from the price-action gap.

    The innings break (~15-20 min) is typically the largest pause between
    consecutive ticks within the match wall window. We look in a wide window
    around marketTime, find the largest inter-tick gap, and treat the next
    tick as innings-2 start. The end is capped at start + 110 min to exclude
    post-settlement ticks where one side has converged to ~1.01.
    """
    if prob.empty:
        return None, None
    wide_start = market_time_anchor_ms - 30 * 60_000  # marketTime can be off
    wide_end = market_time_anchor_ms + 5 * 60 * 60_000
    w = prob[(prob["pt_ms"] >= wide_start) & (prob["pt_ms"] <= wide_end)].copy()
    if len(w) < 10:
        return None, None
    w = w.sort_values("pt_ms").reset_index(drop=True)
    diffs = w["pt_ms"].diff()
    biggest_idx = int(diffs.idxmax())
    biggest_gap_ms = float(diffs.iloc[biggest_idx])
    if biggest_gap_ms < MIN_GAP_FOR_BREAK_MIN * 60_000:
        return None, None
    start = int(w["pt_ms"].iloc[biggest_idx])
    end_cap = start + INNINGS2_MAX_DURATION_MIN * 60_000
    last_observed = int(w["pt_ms"].iloc[-1])
    end = min(end_cap, last_observed)
    return start, end


def chase_window_prices(
    ticks: pd.DataFrame,
    runners: dict[int, str],
    chasing_sel: int,
    market_time_iso: str,
) -> pd.DataFrame:
    """Return a (pt_ms, market_prob_chasing) frame for the chase window only.

    First tries gap-detection to anchor on the real innings break. Falls back
    to the heuristic marketTime + 115min..215min window when no clear break
    is detected (e.g. pre-2018 markets where tick density was lower).
    """
    prob = implied_market_probability(ticks, chasing_sel)
    if prob.empty:
        return prob
    mt = market_time_ms(market_time_iso)
    start, end = detect_innings2_window(prob, mt)
    if start is None:
        start = mt + CHASE_OFFSET_START_MIN * 60_000
        end = mt + CHASE_OFFSET_END_MIN * 60_000
    in_chase = prob[(prob["pt_ms"] >= start) & (prob["pt_ms"] <= end)].copy()
    return in_chase.reset_index(drop=True)


def market_prob_at_ball(
    chase_prices: pd.DataFrame, ball_index: int, total_balls: int = 120
) -> float | None:
    """Look up prevailing market_prob_chasing at the projected wall-time for ball_index.

    Uses chase_prices' observed first/last timestamps as the chase wall-time span,
    then linearly maps ball_index/total_balls into that span, applies the 5s
    decision lag, and returns the most recent price at or before that time.
    """
    if chase_prices.empty:
        return None
    t0 = int(chase_prices["pt_ms"].iloc[0])
    t1 = int(chase_prices["pt_ms"].iloc[-1])
    if t1 <= t0:
        return float(chase_prices["market_prob_chasing"].iloc[-1])
    frac = min(max(ball_index / total_balls, 0.0), 1.0)
    target = t0 + int(frac * (t1 - t0)) + DECISION_LAG_MS
    prior = chase_prices[chase_prices["pt_ms"] <= target]
    if prior.empty:
        return float(chase_prices["market_prob_chasing"].iloc[0])
    return float(prior["market_prob_chasing"].iloc[-1])


def build_market_lookup(
    predictions: pd.DataFrame,
    balls: pd.DataFrame,
    betfair_dir: Path,
    join_df: pd.DataFrame,
) -> pd.DataFrame:
    """For each (match_id, ball_index) in predictions, find the market_prob_chasing.

    Returns the predictions frame with an added 'market_prob' column. Rows for
    matches without Betfair coverage get NaN, to be filtered out downstream.
    """
    join_by_match = join_df.set_index("cricsheet_match_id")
    out_rows: list[float | None] = []

    cache: dict[str, pd.DataFrame] = {}
    for row in predictions.itertuples(index=False):
        match_id = row.match_id
        if match_id not in join_by_match.index:
            out_rows.append(None)
            continue
        if match_id not in cache:
            season = balls.loc[balls["match_id"] == match_id, "season"].iloc[0]
            ticks_path = betfair_dir / f"season={season}" / f"{match_id}_ticks.parquet"
            runners_path = betfair_dir / f"season={season}" / f"{match_id}_runners.json"
            if not ticks_path.exists() or not runners_path.exists():
                cache[match_id] = pd.DataFrame(columns=["pt_ms", "market_prob_chasing"])
            else:
                ticks = pd.read_parquet(ticks_path)
                runners = {int(k): v for k, v in json.loads(runners_path.read_text()).items()}
                chase_sel = chasing_selection_id(balls, match_id, runners)
                join_row = join_by_match.loc[match_id]
                market_time = (
                    join_row["market_time"] if isinstance(join_row, pd.Series)
                    else join_row["market_time"].iloc[0]
                )
                if chase_sel is None:
                    cache[match_id] = pd.DataFrame(columns=["pt_ms", "market_prob_chasing"])
                else:
                    cache[match_id] = chase_window_prices(ticks, runners, chase_sel, market_time)

        out_rows.append(market_prob_at_ball(cache[match_id], int(row.ball_index)))

    result = predictions.copy()
    result["market_prob"] = out_rows
    matched = result["market_prob"].notna().sum()
    logger.info(
        "aligned %d / %d prediction rows to Betfair prices",
        matched, len(result),
    )
    return result


_ = np  # imported for future use in interpolation refinements

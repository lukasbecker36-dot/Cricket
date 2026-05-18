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
    chase_prices: pd.DataFrame,
    ball_index: int,
    anchors: list[tuple[int, int]] | None = None,
    total_balls: int = 120,
) -> float | None:
    """Look up prevailing market_prob_chasing at the projected wall-time for ball_index.

    If `anchors` is provided, interpolates wall-time between (ball_index,
    wall_time_ms) anchor pairs -- typically derived from matching wicket events
    to detected price drops. Otherwise falls back to linear ball/total mapping
    across the chase window observed in `chase_prices`.

    Applies DECISION_LAG_MS (negative => use stale prices) before lookup.
    """
    if chase_prices.empty:
        return None
    if anchors is not None and len(anchors) >= 2:
        target = interpolate_ball_to_wall(ball_index, anchors)
    else:
        t0 = int(chase_prices["pt_ms"].iloc[0])
        t1 = int(chase_prices["pt_ms"].iloc[-1])
        if t1 <= t0:
            return float(chase_prices["market_prob_chasing"].iloc[-1])
        frac = min(max(ball_index / total_balls, 0.0), 1.0)
        target = t0 + int(frac * (t1 - t0))
    target += DECISION_LAG_MS
    prior = chase_prices[chase_prices["pt_ms"] <= target]
    if prior.empty:
        return float(chase_prices["market_prob_chasing"].iloc[0])
    return float(prior["market_prob_chasing"].iloc[-1])


# --- Wicket-event price-jump matching ------------------------------------

# A price drop must be at least this large within `WICKET_WINDOW_MS` to count
# as a likely wicket. Boundaries cause smaller drops; bowled/caught/lbw typically
# move the chasing team's prob by 4-15 percentage points within a minute.
WICKET_MIN_DROP = 0.04
WICKET_WINDOW_MS = 60_000
# A ball is considered to have happened during the "uncertain" period if the
# market prob is strictly between these. Outside this range the outcome is
# essentially settled and tick timestamps no longer track real ball events.
UNCERTAIN_LO = 0.05
UNCERTAIN_HI = 0.95


def wicket_ball_indices(balls: pd.DataFrame, match_id: str) -> list[int]:
    """Return ball_index values (replay_chase numbering) where wickets fell in inn 2."""
    inn2 = balls[(balls["match_id"] == match_id) & (balls["innings"] == 2)].sort_values(
        ["over", "ball", "is_legal_delivery"], ascending=[True, True, False]
    ).reset_index(drop=True)
    return [int(i) for i, row in enumerate(inn2.itertuples(index=False)) if bool(row.wicket)]


def detect_price_drops(
    chase_prices: pd.DataFrame,
    min_drop: float = WICKET_MIN_DROP,
    window_ms: int = WICKET_WINDOW_MS,
) -> list[int]:
    """Find timestamps of likely wicket-induced sudden drops in chasing-team prob.

    For each tick, compare its prob to the max prob in the prior `window_ms`.
    A drop of >= `min_drop` is a candidate. Candidates within `window_ms` of
    each other are clustered (same event); we keep the biggest in each cluster.
    """
    if chase_prices.empty:
        return []
    s = chase_prices.sort_values("pt_ms").reset_index(drop=True)
    times = s["pt_ms"].to_numpy()
    probs = s["market_prob_chasing"].to_numpy()
    candidates: list[tuple[float, int]] = []
    j = 0
    for i in range(1, len(s)):
        while j < i and times[j] < times[i] - window_ms:
            j += 1
        if j == i:
            continue
        max_prev = probs[j:i].max()
        drop = float(max_prev - probs[i])
        if drop >= min_drop:
            candidates.append((drop, int(times[i])))
    if not candidates:
        return []
    candidates.sort(key=lambda x: x[1])
    clustered: list[tuple[float, int]] = [candidates[0]]
    for drop, t in candidates[1:]:
        if t - clustered[-1][1] < window_ms:
            if drop > clustered[-1][0]:
                clustered[-1] = (drop, t)
        else:
            clustered.append((drop, t))
    return [t for _, t in clustered]


def find_chase_uncertain_end(chase_prices: pd.DataFrame) -> int | None:
    """Last tick whose prob is in the uncertain range; rough proxy for 'match still alive'."""
    if chase_prices.empty:
        return None
    uncertain = chase_prices[
        (chase_prices["market_prob_chasing"] > UNCERTAIN_LO)
        & (chase_prices["market_prob_chasing"] < UNCERTAIN_HI)
    ]
    if uncertain.empty:
        return int(chase_prices["pt_ms"].iloc[-1])
    return int(uncertain["pt_ms"].iloc[-1])


def build_anchors(
    wicket_indices: list[int],
    drop_times: list[int],
    chase_start_ms: int,
    chase_end_ms: int,
    last_ball_index: int = 120,
) -> list[tuple[int, int]] | None:
    """Pair wickets to price drops by chronological order, return sorted
    (ball_index, wall_time_ms) anchors. Returns None if the result isn't
    strictly monotonic in both axes (which means we matched wrong).

    Always brackets with the chase start (ball_index=0) and chase end
    (ball_index=last_ball_index).
    """
    n = min(len(wicket_indices), len(drop_times))
    pairs = list(zip(wicket_indices[:n], drop_times[:n]))
    anchors = [(0, chase_start_ms)] + pairs + [(last_ball_index, chase_end_ms)]
    for i in range(1, len(anchors)):
        if anchors[i][0] <= anchors[i - 1][0]:
            return None
        if anchors[i][1] <= anchors[i - 1][1]:
            return None
    return anchors


def interpolate_ball_to_wall(ball_index: int, anchors: list[tuple[int, int]]) -> int:
    """Linearly interpolate ball_index -> wall_time_ms between adjacent anchors."""
    for i in range(len(anchors) - 1):
        b0, t0 = anchors[i]
        b1, t1 = anchors[i + 1]
        if b0 <= ball_index <= b1:
            if b1 == b0:
                return t0
            frac = (ball_index - b0) / (b1 - b0)
            return int(t0 + frac * (t1 - t0))
    if ball_index <= anchors[0][0]:
        return anchors[0][1]
    return anchors[-1][1]


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

    # cache[match_id] = (chase_prices_df, anchors_list_or_None)
    cache: dict[str, tuple[pd.DataFrame, list[tuple[int, int]] | None]] = {}
    n_with_anchors = 0
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
                cache[match_id] = (pd.DataFrame(columns=["pt_ms", "market_prob_chasing"]), None)
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
                    cache[match_id] = (pd.DataFrame(columns=["pt_ms", "market_prob_chasing"]), None)
                else:
                    cp = chase_window_prices(ticks, runners, chase_sel, market_time)
                    anchors: list[tuple[int, int]] | None = None
                    if not cp.empty:
                        wickets = wicket_ball_indices(balls, match_id)
                        drops = detect_price_drops(cp)
                        chase_start = int(cp["pt_ms"].iloc[0])
                        chase_end = find_chase_uncertain_end(cp) or int(cp["pt_ms"].iloc[-1])
                        # Use the actual last ball index for this match as the
                        # end-anchor ball position. Cricsheet often has chases
                        # ending early (won with overs to spare), so 120 is wrong.
                        last_idx = int(predictions[predictions["match_id"] == match_id]["ball_index"].max())
                        anchors = build_anchors(
                            wickets, drops, chase_start, chase_end, last_ball_index=last_idx,
                        )
                        if anchors is not None:
                            n_with_anchors += 1
                    cache[match_id] = (cp, anchors)

        cp, anchors = cache[match_id]
        out_rows.append(market_prob_at_ball(cp, int(row.ball_index), anchors=anchors))

    result = predictions.copy()
    result["market_prob"] = out_rows
    matched = result["market_prob"].notna().sum()
    logger.info("anchored %d / %d matches via wicket-event matching", n_with_anchors, len(cache))
    logger.info(
        "aligned %d / %d prediction rows to Betfair prices",
        matched, len(result),
    )
    return result


_ = np  # imported for future use in interpolation refinements

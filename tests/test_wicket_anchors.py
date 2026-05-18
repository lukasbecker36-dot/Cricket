"""Wicket-event price-jump matching."""
import pandas as pd

from src.ingestion.market_alignment import (
    build_anchors,
    detect_price_drops,
    find_chase_uncertain_end,
    interpolate_ball_to_wall,
    wicket_ball_indices,
)


def test_wicket_ball_indices_extracts_from_balls():
    rows = []
    for i in range(12):
        rows.append({
            "match_id": "m", "season": 2024, "venue": "V",
            "innings": 2, "over": i // 6, "ball": (i % 6) + 1,
            "batting_team": "T1", "bowling_team": "T2",
            "striker": "A", "non_striker": "B", "bowler": "X",
            "runs_batter": 0, "runs_extras": 0, "runs_total": 0,
            "extras_kind": None,
            "wicket": i in (3, 9), "dismissal_kind": "bowled" if i in (3, 9) else None,
            "player_out": "A" if i in (3, 9) else None,
            "target": 200, "is_legal_delivery": True,
        })
    df = pd.DataFrame(rows)
    assert wicket_ball_indices(df, "m") == [3, 9]


def test_detect_price_drops_finds_sudden_drops():
    # Three small ticks ~0.5, then a 6pp drop to 0.44, then noise. One wicket event.
    times = [t * 1_000 for t in range(0, 200, 10)]  # 20 ticks, 10s apart
    probs = [0.50] * 10 + [0.44] * 10  # drop at index 10
    df = pd.DataFrame({"pt_ms": times, "market_prob_chasing": probs})
    drops = detect_price_drops(df, min_drop=0.04, window_ms=60_000)
    assert len(drops) == 1
    assert drops[0] == times[10]  # exactly the tick where it dropped


def test_detect_price_drops_clusters_close_events():
    # Two drops within 30s should collapse to one (same wicket re-pricing)
    times = [t * 1_000 for t in range(0, 200, 10)]
    probs = [0.6] * 5 + [0.55] * 5 + [0.50] * 10
    df = pd.DataFrame({"pt_ms": times, "market_prob_chasing": probs})
    drops = detect_price_drops(df, min_drop=0.04, window_ms=60_000)
    assert len(drops) == 1


def test_find_chase_uncertain_end_skips_settled_tail():
    df = pd.DataFrame({
        "pt_ms": [10, 20, 30, 40, 50],
        "market_prob_chasing": [0.5, 0.7, 0.96, 0.99, 0.99],
    })
    # Last uncertain tick is at t=20 (0.7); 0.96/0.99 are settled
    assert find_chase_uncertain_end(df) == 20


def test_build_anchors_pairs_chronologically():
    anchors = build_anchors(
        wicket_indices=[10, 30, 50],
        drop_times=[1000, 2000, 3000],
        chase_start_ms=0,
        chase_end_ms=4000,
        last_ball_index=120,
    )
    assert anchors == [(0, 0), (10, 1000), (30, 2000), (50, 3000), (120, 4000)]


def test_build_anchors_returns_none_on_non_monotonic():
    # Wickets in order but drop_times don't agree (would imply time going backward)
    anchors = build_anchors(
        wicket_indices=[10, 30],
        drop_times=[2000, 1500],
        chase_start_ms=0, chase_end_ms=3000, last_ball_index=120,
    )
    assert anchors is None


def test_build_anchors_truncates_to_shorter():
    # 3 wickets, 2 detected drops: pair first 2 only
    anchors = build_anchors(
        wicket_indices=[10, 30, 50],
        drop_times=[1000, 2000],
        chase_start_ms=0, chase_end_ms=4000, last_ball_index=120,
    )
    assert anchors == [(0, 0), (10, 1000), (30, 2000), (120, 4000)]


def test_interpolate_ball_to_wall_within_segment():
    anchors = [(0, 0), (60, 60_000), (120, 120_000)]
    # ball 30 -> halfway through first segment -> 30_000 ms
    assert interpolate_ball_to_wall(30, anchors) == 30_000
    # ball 90 -> halfway through second segment -> 90_000 ms
    assert interpolate_ball_to_wall(90, anchors) == 90_000


def test_interpolate_clamps_outside_range():
    anchors = [(10, 1000), (50, 5000)]
    assert interpolate_ball_to_wall(5, anchors) == 1000
    assert interpolate_ball_to_wall(100, anchors) == 5000

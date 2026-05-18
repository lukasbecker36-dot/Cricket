"""Pure-function tests for market alignment helpers."""
import pandas as pd

from src.ingestion.market_alignment import (
    chasing_selection_id,
    market_prob_at_ball,
    market_time_ms,
)


def test_chasing_selection_id_exact_match():
    balls = pd.DataFrame([
        {"match_id": "m1", "innings": 2, "batting_team": "Mumbai Indians"},
        {"match_id": "m1", "innings": 1, "batting_team": "Chennai Super Kings"},
    ])
    runners = {1234: "Mumbai Indians", 5678: "Chennai Super Kings"}
    assert chasing_selection_id(balls, "m1", runners) == 1234


def test_chasing_selection_id_substring_fallback():
    """Cricsheet says 'Royal Challengers Bangalore'; Betfair says 'Royal Challengers Bengaluru'."""
    balls = pd.DataFrame([
        {"match_id": "m1", "innings": 2, "batting_team": "Royal Challengers Bangalore"},
    ])
    runners = {1234: "Royal Challengers Bengaluru", 5678: "Sunrisers Hyderabad"}
    assert chasing_selection_id(balls, "m1", runners) == 1234


def test_chasing_selection_id_missing_returns_none():
    balls = pd.DataFrame(columns=["match_id", "innings", "batting_team"])
    assert chasing_selection_id(balls, "missing", {1: "A"}) is None


def test_market_time_ms_parses_isoformat():
    # 2024-04-15T14:00:00Z -> unix ms
    assert market_time_ms("2024-04-15T14:00:00Z") == 1_713_189_600_000


def test_market_prob_at_ball_empty_returns_none():
    assert market_prob_at_ball(pd.DataFrame(columns=["pt_ms", "market_prob_chasing"]), 0) is None


def test_market_prob_at_ball_first_and_last():
    chase = pd.DataFrame({
        "pt_ms": [1_000_000, 2_000_000, 3_000_000, 4_000_000],
        "market_prob_chasing": [0.5, 0.6, 0.7, 0.8],
    })
    # ball 0 -> first quartile of window
    p0 = market_prob_at_ball(chase, 0, total_balls=120)
    # ball 120 -> end of window
    p_end = market_prob_at_ball(chase, 120, total_balls=120)
    assert p0 == 0.5  # earliest tick (no prior, falls back to first)
    assert p_end == 0.8  # last tick


def test_market_prob_at_ball_midpoint():
    # Chase window 1_000_000 to 4_000_000 ms. Ball 60/120 -> target ~2_500_000 + 5s lag.
    chase = pd.DataFrame({
        "pt_ms": [1_000_000, 2_000_000, 3_000_000, 4_000_000],
        "market_prob_chasing": [0.5, 0.6, 0.7, 0.8],
    })
    p_mid = market_prob_at_ball(chase, 60, total_balls=120)
    # target = 2_505_000; prevailing price at or before is 0.6 (the 2_000_000 tick)
    assert p_mid == 0.6

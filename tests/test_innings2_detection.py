"""Innings-2 window detection from price-action gaps."""
import pandas as pd

from src.ingestion.market_alignment import detect_innings2_window


def make_ticks(timestamps_ms: list[int], probs: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"pt_ms": timestamps_ms, "market_prob_chasing": probs})


def test_no_clear_gap_returns_none():
    # 12 evenly-spaced ticks 1 min apart -> no big break
    ts = [60_000 * i for i in range(12)]
    df = make_ticks(ts, [0.5] * 12)
    s, e = detect_innings2_window(df, market_time_anchor_ms=0)
    assert s is None and e is None


def test_largest_gap_marks_innings2_start():
    # Innings 1: ticks at 0..9 min. Break: 9..30 min (21 min gap). Innings 2: 30..120 min.
    inn1 = list(range(0, 10 * 60_000, 60_000))
    inn2 = list(range(30 * 60_000, 120 * 60_000, 60_000))
    df = make_ticks(inn1 + inn2, [0.5] * len(inn1 + inn2))
    s, e = detect_innings2_window(df, market_time_anchor_ms=0)
    assert s == 30 * 60_000
    # End capped at start + 110 min if data goes further; here last is 119 min
    assert e == 119 * 60_000


def test_end_capped_at_max_duration():
    inn1 = list(range(0, 10 * 60_000, 60_000))
    # Innings 2 starts at 30 min, ticks go all the way to 200 min (post-settlement)
    inn2 = list(range(30 * 60_000, 200 * 60_000, 60_000))
    df = make_ticks(inn1 + inn2, [0.5] * len(inn1 + inn2))
    s, e = detect_innings2_window(df, market_time_anchor_ms=0)
    assert s == 30 * 60_000
    # 30 + 110 = 140 min cap
    assert e == 140 * 60_000


def test_empty_returns_none():
    s, e = detect_innings2_window(pd.DataFrame(columns=["pt_ms", "market_prob_chasing"]), 0)
    assert s is None and e is None

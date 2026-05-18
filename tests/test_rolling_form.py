"""Rolling-form must be leakage-free: a batter's recent SR for match M
must use only matches strictly before M.
"""
import pandas as pd

from src.features.rolling_form import compute_rolling_batter_form, recent_sr


def make_batter_match(match_id: str, date: str, striker: str, runs: int, balls: int) -> pd.DataFrame:
    """Build a tiny innings-2-only match with `balls` legal deliveries for `striker`."""
    return pd.DataFrame([
        {
            "match_id": match_id, "season": 2024, "venue": "V",
            "innings": 2, "over": i // 6, "ball": (i % 6) + 1,
            "batting_team": "T1", "bowling_team": "T2",
            "striker": striker, "non_striker": "B", "bowler": "X",
            "runs_batter": runs // balls if balls else 0, "runs_extras": 0,
            "runs_total": runs // balls if balls else 0,
            "extras_kind": None, "wicket": False, "dismissal_kind": None,
            "player_out": None, "target": 200, "is_legal_delivery": True,
        }
        for i in range(balls)
    ])


def test_recent_sr_uses_only_prior_matches():
    # Batter A: 30 runs off 30 balls in match 1 (SR 100), then plays match 2.
    m1 = make_batter_match("m1", "2024-04-01", "A", runs=30, balls=30)
    m2 = make_batter_match("m2", "2024-04-08", "A", runs=120, balls=30)  # explosive
    balls = pd.concat([m1, m2], ignore_index=True)
    matches = pd.DataFrame([
        {"match_id": "m1", "season": 2024, "date": "2024-04-01"},
        {"match_id": "m2", "season": 2024, "date": "2024-04-08"},
    ])

    form = compute_rolling_batter_form(balls, matches)
    # m1: no prior matches -> defaults
    assert form[("m1", "A")] == (0.0, 0.0)
    # m2: prior runs/balls = 30/30
    rr, rb = form[("m2", "A")]
    assert rr == 30.0
    assert rb == 30.0


def test_recent_sr_shrinkage_pulls_toward_league_mean():
    # No prior data -> recent_sr should default to league mean (130)
    assert recent_sr({}, "anywhere", "anyone") == 130.0
    # A batter with only 6 balls at SR 200 should be heavily shrunk
    form = {("m", "A"): (12.0, 6.0)}  # 12 runs off 6 = SR 200
    sr = recent_sr(form, "m", "A")
    # (12*100 + 60*130) / (6+60) = (1200 + 7800)/66 = 9000/66 = 136.4
    assert 130.0 < sr < 145.0


def test_recent_sr_with_full_window_matches_observed():
    # 100 runs off 100 balls -> observed SR = 100. With 60-ball prior at 130:
    # (100*100 + 60*130) / 160 = (10000+7800)/160 = 111.25
    form = {("m", "A"): (100.0, 100.0)}
    sr = recent_sr(form, "m", "A")
    assert abs(sr - 111.25) < 0.01


def test_rolling_window_caps_at_n_matches():
    # Three matches; window=2 should ignore the oldest when computing match 3's form
    m1 = make_batter_match("m1", "2024-04-01", "A", runs=0, balls=10)
    m2 = make_batter_match("m2", "2024-04-08", "A", runs=10, balls=10)
    m3 = make_batter_match("m3", "2024-04-15", "A", runs=20, balls=10)
    balls = pd.concat([m1, m2, m3], ignore_index=True)
    matches = pd.DataFrame([
        {"match_id": "m1", "season": 2024, "date": "2024-04-01"},
        {"match_id": "m2", "season": 2024, "date": "2024-04-08"},
        {"match_id": "m3", "season": 2024, "date": "2024-04-15"},
    ])
    form = compute_rolling_batter_form(balls, matches, window_matches=2)
    # m3's prior = m1 + m2 = 10 runs / 20 balls (window 2 = m1 and m2)
    rr, rb = form[("m3", "A")]
    assert rb == 20.0
    assert rr == 10.0

"""Boundary rate: shrunk toward league mean for thin samples."""
import pandas as pd

from src.features.player_quality import LEAGUE_BOUNDARY_RATE, PlayerStats, shrunk_boundary_rate


def stats(rate: float | None, balls: int) -> PlayerStats:
    return PlayerStats(
        batting_strike_rate=pd.Series(dtype=float),
        batting_balls_faced=pd.Series({"A": balls}) if balls else pd.Series(dtype=float),
        batting_boundary_rate=pd.Series({"A": rate}) if rate is not None else pd.Series(dtype=float),
        bowler_economy=pd.Series(dtype=float),
        bowler_balls=pd.Series(dtype=float),
        bowler_phase_runs={},
        bowler_phase_balls={},
        league_phase_economy={0: 8.0, 1: 8.0, 2: 8.0},
    )


def test_unknown_batter_defaults_to_league_mean():
    assert shrunk_boundary_rate(stats(None, 0), "anyone") == LEAGUE_BOUNDARY_RATE


def test_high_boundary_rate_with_thin_sample_pulled_to_prior():
    # 10 balls at 0.5 (super boundary-heavy); prior=60 at LEAGUE_BOUNDARY_RATE
    sr = shrunk_boundary_rate(stats(0.5, 10), "A")
    # (10*0.5 + 60*0.135) / 70 = (5 + 8.1)/70 = 0.187
    assert 0.135 < sr < 0.25


def test_dense_sample_close_to_observed():
    # 1200 balls at 0.20 boundary rate
    sr = shrunk_boundary_rate(stats(0.20, 1200), "A")
    # Should be close to 0.20 with small pull toward league mean
    assert 0.19 < sr < 0.21

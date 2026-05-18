import pandas as pd

from src.features.engineering import remaining_bowling_quality
from src.features.player_quality import LEAGUE_ECON_MEAN, PlayerStats


def stats_with(bowler_econs: dict[str, float], balls_each: int = 600) -> PlayerStats:
    """Build PlayerStats with given career economies for bowlers. balls_each large
    enough that shrinkage barely moves the value."""
    return PlayerStats(
        batting_strike_rate=pd.Series(dtype=float),
        batting_balls_faced=pd.Series(dtype=float),
        bowler_economy=pd.Series(bowler_econs),
        bowler_balls=pd.Series({b: balls_each for b in bowler_econs}),
    )


def test_remaining_econ_with_no_bowlers_used_defaults_to_league_mean():
    econ, best, n = remaining_bowling_quality({}, legal_balls_bowled=0, stats=stats_with({}))
    assert abs(econ - LEAGUE_ECON_MEAN) < 1e-6
    assert best == 0.0
    assert n == 0


def test_using_up_best_bowler_lifts_remaining_econ():
    # Bumrah-like: econ 6.0 over 600 balls. Other bowlers default to league mean.
    s = stats_with({"BUMRAH": 6.0})
    # Bumrah has bowled all 24 of his legal balls already
    used_all = {"BUMRAH": 24}
    used_none = {"BUMRAH": 0}
    econ_used, _, _ = remaining_bowling_quality(used_all, legal_balls_bowled=24, stats=s)
    econ_fresh, _, _ = remaining_bowling_quality(used_none, legal_balls_bowled=24, stats=s)
    # Fresh Bumrah pulls expected econ down; used-up Bumrah leaves only league-mean
    assert econ_fresh < econ_used


def test_best_bowler_overs_left_tracks_freshest_bowler():
    s = stats_with({"A": 8.0, "B": 8.0})
    # A has bowled 12 legal balls (2 overs); B has bowled 6 (1 over)
    used = {"A": 12, "B": 6}
    _, best, n = remaining_bowling_quality(used, legal_balls_bowled=18, stats=s)
    # B has the most left: 24 - 6 = 18 balls = 3 overs
    assert abs(best - 3.0) < 1e-6
    assert n == 2


def test_innings_complete_returns_league_mean():
    econ, _, _ = remaining_bowling_quality({}, legal_balls_bowled=120, stats=stats_with({}))
    assert abs(econ - LEAGUE_ECON_MEAN) < 1e-6

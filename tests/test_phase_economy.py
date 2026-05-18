"""Phase-specific bowler economy: a bowler with 5.0 in death and 9.0 in PP
should return distinct values when queried by phase."""
import pandas as pd

from src.features.player_quality import (
    LEAGUE_ECON_MEAN,
    PlayerStats,
    over_to_phase,
    phase_shrunk_economy,
)


def test_over_to_phase_boundaries():
    assert over_to_phase(0) == 0
    assert over_to_phase(5) == 0
    assert over_to_phase(6) == 1
    assert over_to_phase(14) == 1
    assert over_to_phase(15) == 2
    assert over_to_phase(19) == 2


def test_phase_economy_returns_league_for_unknown_bowler():
    stats = PlayerStats(
        batting_strike_rate=pd.Series(dtype=float),
        batting_balls_faced=pd.Series(dtype=float),
        batting_boundary_rate=pd.Series(dtype=float),
        bowler_economy=pd.Series(dtype=float),
        bowler_balls=pd.Series(dtype=float),
        bowler_phase_runs={},
        bowler_phase_balls={},
        league_phase_economy={0: 7.5, 1: 8.3, 2: 10.0},
    )
    assert phase_shrunk_economy(stats, "nobody", 0) == 7.5
    assert phase_shrunk_economy(stats, "nobody", 2) == 10.0


def test_phase_economy_distinguishes_phases_for_same_bowler():
    # Bowler with 1200 balls in PP at 5.0 econ, same volume in death at 9.0 econ
    pp_balls, death_balls = 1200, 1200
    pp_runs = int(5.0 * pp_balls / 6.0)
    death_runs = int(9.0 * death_balls / 6.0)
    stats = PlayerStats(
        batting_strike_rate=pd.Series(dtype=float),
        batting_balls_faced=pd.Series(dtype=float),
        batting_boundary_rate=pd.Series(dtype=float),
        bowler_economy=pd.Series(dtype=float),
        bowler_balls=pd.Series(dtype=float),
        bowler_phase_runs={("BUMRAH", 0): pp_runs, ("BUMRAH", 2): death_runs},
        bowler_phase_balls={("BUMRAH", 0): pp_balls, ("BUMRAH", 2): death_balls},
        league_phase_economy={0: 7.5, 1: 8.3, 2: 10.0},
    )
    pp_econ = phase_shrunk_economy(stats, "BUMRAH", 0)
    death_econ = phase_shrunk_economy(stats, "BUMRAH", 2)
    # With 1200 balls and a 30-ball prior, shrinkage barely moves the values
    assert abs(pp_econ - 5.06) < 0.1
    assert abs(death_econ - 9.02) < 0.1
    # And phase 1 (middle), which the bowler hasn't bowled, falls back to league
    assert phase_shrunk_economy(stats, "BUMRAH", 1) == 8.3

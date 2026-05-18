"""Bowler wicket rate per phase, with bowler-credited-only filtering."""
import pandas as pd

from src.features.player_quality import (
    BOWLER_DISMISSAL_KINDS,
    LEAGUE_WICKET_RATE,
    PlayerStats,
    compute_player_stats,
    phase_shrunk_wicket_rate,
)


def test_run_outs_excluded():
    assert "run out" not in BOWLER_DISMISSAL_KINDS
    assert "retired hurt" not in BOWLER_DISMISSAL_KINDS
    assert "bowled" in BOWLER_DISMISSAL_KINDS
    assert "caught" in BOWLER_DISMISSAL_KINDS


def test_phase_wicket_rate_unknown_bowler_returns_league():
    stats = PlayerStats(
        batting_strike_rate=pd.Series(dtype=float),
        batting_balls_faced=pd.Series(dtype=float),
        batting_boundary_rate=pd.Series(dtype=float),
        bowler_economy=pd.Series(dtype=float),
        bowler_balls=pd.Series(dtype=float),
        bowler_phase_runs={},
        bowler_phase_balls={},
        bowler_phase_wickets={},
        league_phase_economy={0: 8.0, 1: 8.0, 2: 8.0},
        league_phase_wicket_rate={0: 0.03, 1: 0.04, 2: 0.05},
    )
    assert phase_shrunk_wicket_rate(stats, "nobody", 2) == 0.05


def test_high_wicket_rate_bowler_distinct_from_league():
    # 1200 balls, 100 wickets in death -> 8.3% wicket rate; league 5%
    stats = PlayerStats(
        batting_strike_rate=pd.Series(dtype=float),
        batting_balls_faced=pd.Series(dtype=float),
        batting_boundary_rate=pd.Series(dtype=float),
        bowler_economy=pd.Series(dtype=float),
        bowler_balls=pd.Series(dtype=float),
        bowler_phase_runs={},
        bowler_phase_balls={("BUMRAH", 2): 1200},
        bowler_phase_wickets={("BUMRAH", 2): 100},
        league_phase_economy={0: 8.0, 1: 8.0, 2: 8.0},
        league_phase_wicket_rate={0: 0.03, 1: 0.04, 2: 0.05},
    )
    rate = phase_shrunk_wicket_rate(stats, "BUMRAH", 2)
    # observed 100/1200 = 0.0833. Prior 30 balls at 0.05.
    # (1200*0.0833 + 30*0.05)/(1230) = (100 + 1.5)/1230 = 0.0825
    assert 0.07 < rate < 0.09
    # PP fallback (not bowled there) -> league
    assert phase_shrunk_wicket_rate(stats, "BUMRAH", 0) == 0.03


def test_compute_player_stats_filters_run_outs():
    """If a wicket is a run-out, it should NOT count toward the bowler's tally."""
    rows = [
        # Bowler X: 12 legal balls, 1 caught (bowler-credited)
        *[
            {
                "match_id": "m1", "season": 1, "venue": "V", "innings": 1,
                "over": 0, "ball": i + 1, "batting_team": "T1", "bowling_team": "T2",
                "striker": "A", "non_striker": "B", "bowler": "X",
                "runs_batter": 0, "runs_extras": 0, "runs_total": 0,
                "extras_kind": None,
                "wicket": (i == 5), "dismissal_kind": ("caught" if i == 5 else None),
                "player_out": ("A" if i == 5 else None),
                "target": None, "is_legal_delivery": True,
            }
            for i in range(6)
        ],
        # Bowler Y: 6 legal balls, 1 run-out (NOT bowler-credited)
        *[
            {
                "match_id": "m1", "season": 1, "venue": "V", "innings": 1,
                "over": 1, "ball": i + 1, "batting_team": "T1", "bowling_team": "T2",
                "striker": "A", "non_striker": "B", "bowler": "Y",
                "runs_batter": 0, "runs_extras": 0, "runs_total": 0,
                "extras_kind": None,
                "wicket": (i == 5), "dismissal_kind": ("run out" if i == 5 else None),
                "player_out": ("B" if i == 5 else None),
                "target": None, "is_legal_delivery": True,
            }
            for i in range(6)
        ],
    ]
    balls = pd.DataFrame(rows)
    stats = compute_player_stats(balls, up_to_season=2)
    # X got 1 wicket in phase 0
    assert stats.bowler_phase_wickets.get(("X", 0), 0) == 1
    # Y had a run-out happen but it does NOT count
    assert stats.bowler_phase_wickets.get(("Y", 0), 0) == 0
    _ = LEAGUE_WICKET_RATE  # used elsewhere

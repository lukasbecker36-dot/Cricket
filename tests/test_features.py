import pandas as pd

from src.features.engineering import FEATURE_COLUMNS, features_from_state, venue_par_score
from src.features.player_quality import PlayerStats, shrunk_economy, shrunk_strike_rate
from src.features.state import ChaseState


def empty_stats():
    return PlayerStats(
        batting_strike_rate=pd.Series(dtype=float),
        batting_balls_faced=pd.Series(dtype=float),
        batting_boundary_rate=pd.Series(dtype=float),
        bowler_economy=pd.Series(dtype=float),
        bowler_balls=pd.Series(dtype=float),
        bowler_phase_runs={},
        bowler_phase_balls={},
        league_phase_economy={0: 8.0, 1: 8.0, 2: 8.0},
    )


def test_features_have_expected_columns():
    s = ChaseState(
        match_id="m", season=2024, venue="V", target=160,
        runs_scored=80, wickets_lost=3, legal_balls_bowled=60,
        striker="A", non_striker="B", striker_balls_faced=20, non_striker_balls_faced=18,
        bowler="X",
    )
    feats = features_from_state(s, empty_stats(), pd.Series(dtype=float))
    assert set(feats.keys()) == set(FEATURE_COLUMNS)


def test_shrunk_sr_defaults_to_league_mean_for_unknown_player():
    assert shrunk_strike_rate(empty_stats(), "nobody", league_mean=130.0) == 130.0


def test_shrunk_economy_defaults_for_unknown_player():
    assert shrunk_economy(empty_stats(), "nobody", league_mean=8.0) == 8.0


def test_shrinkage_pulls_small_sample_toward_prior():
    stats = PlayerStats(
        batting_strike_rate=pd.Series({"A": 200.0}),
        batting_balls_faced=pd.Series({"A": 10}),
        batting_boundary_rate=pd.Series(dtype=float),
        bowler_economy=pd.Series(dtype=float),
        bowler_balls=pd.Series(dtype=float),
        bowler_phase_runs={},
        bowler_phase_balls={},
        league_phase_economy={0: 8.0, 1: 8.0, 2: 8.0},
    )
    # 10 balls at SR 200 with prior=60 at mean=130 -> (10*200 + 60*130)/70 = (2000+7800)/70
    sr = shrunk_strike_rate(stats, "A", league_mean=130.0)
    assert 130.0 < sr < 200.0


def test_venue_par_falls_back_to_league_mean():
    assert venue_par_score(pd.Series(dtype=float), "Unknown", league_mean=165.0) == 165.0


def test_required_run_rate_capped_when_no_balls_left():
    s = ChaseState(
        match_id="m", season=2024, venue="V", target=180,
        runs_scored=150, wickets_lost=8, legal_balls_bowled=120,
        striker="A", non_striker="B", striker_balls_faced=10, non_striker_balls_faced=10,
        bowler="X",
    )
    feats = features_from_state(s, empty_stats(), pd.Series(dtype=float))
    assert feats["required_run_rate"] == 36.0

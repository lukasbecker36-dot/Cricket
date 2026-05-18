"""Regression test: stats used for a training row must come strictly from prior seasons."""
import pandas as pd

from src.features.engineering import FEATURE_COLUMNS
from src.validation.walk_forward import build_dataset


def make_match(match_id: str, season: int, batter_runs_per_ball: int) -> pd.DataFrame:
    """Build a tiny 12-ball innings-2-only match for a single batter."""
    rows = []
    for i in range(12):
        rows.append({
            "match_id": match_id, "season": season, "venue": "V",
            "innings": 2, "over": i // 6, "ball": (i % 6) + 1,
            "batting_team": "T1", "bowling_team": "T2",
            "striker": "STAR", "non_striker": "B", "bowler": "X",
            "runs_batter": batter_runs_per_ball, "runs_extras": 0,
            "runs_total": batter_runs_per_ball,
            "extras_kind": None, "wicket": False, "dismissal_kind": None,
            "player_out": None, "target": 200, "is_legal_delivery": True,
        })
    # Innings 1 to set target / produce 1 row of innings-1 data per match
    for i in range(6):
        rows.append({
            "match_id": match_id, "season": season, "venue": "V",
            "innings": 1, "over": 0, "ball": i + 1,
            "batting_team": "T2", "bowling_team": "T1",
            "striker": "OTHER", "non_striker": "B", "bowler": "X",
            "runs_batter": 0, "runs_extras": 0, "runs_total": 0,
            "extras_kind": None, "wicket": False, "dismissal_kind": None,
            "player_out": None, "target": None, "is_legal_delivery": True,
        })
    return pd.DataFrame(rows)


def test_training_row_stats_use_only_prior_seasons():
    """STAR scores 0 in season 1 and 6/ball (impossible SR=600) in season 2.

    A season-2 training row's striker_sr feature should reflect season-1 stats only.
    If leakage were present, it would include season-2's massive numbers.
    """
    m1 = make_match("a", season=1, batter_runs_per_ball=0)
    m2 = make_match("b", season=2, batter_runs_per_ball=6)
    balls = pd.concat([m1, m2], ignore_index=True)
    matches = pd.DataFrame([
        {"match_id": "a", "season": 1, "winner": "T1"},
        {"match_id": "b", "season": 2, "winner": "T1"},
    ])

    df = build_dataset(balls, matches, up_to_season_for_stats=99, min_balls_into_chase=0)
    season2_rows = df[df["season"] == 2]
    assert not season2_rows.empty
    # All season-2 rows must use stats from season 1 only (where STAR scored 0).
    # Shrunk SR pulls towards league mean (130). With 0 SR from 12 balls and prior=60,
    # shrunk = (12*0 + 60*130)/(72) ~= 108.3. Definitely well below 600.
    assert season2_rows["striker_sr"].max() < 200.0, (
        f"striker_sr leaked future data: max {season2_rows['striker_sr'].max()}"
    )


def test_first_season_row_uses_league_mean():
    """No prior seasons => stats default to league mean for unknown player."""
    m1 = make_match("a", season=1, batter_runs_per_ball=4)
    matches = pd.DataFrame([{"match_id": "a", "season": 1, "winner": "T1"}])
    df = build_dataset(m1, matches, up_to_season_for_stats=99, min_balls_into_chase=0)
    # First row: STAR has no prior data, so striker_sr should be the league-mean default (130)
    assert abs(df.iloc[0]["striker_sr"] - 130.0) < 1e-6

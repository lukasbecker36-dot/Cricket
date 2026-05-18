import pandas as pd

from src.features.engineering import FEATURE_COLUMNS, replay_chase
from src.features.state import ChaseState


def make_balls(rows):
    """rows: list of (runs_total, runs_batter, wicket, is_legal). Builds an innings-2 frame."""
    out = []
    over, ball_in_over = 0, 0
    for runs_total, runs_batter, wicket, is_legal in rows:
        if is_legal:
            ball_in_over += 1
            if ball_in_over > 6:
                over += 1
                ball_in_over = 1
        out.append({
            "match_id": "m", "season": 2024, "venue": "V",
            "innings": 2, "over": over, "ball": ball_in_over if is_legal else 0,
            "batting_team": "T1", "bowling_team": "T2",
            "striker": "A", "non_striker": "B", "bowler": "X",
            "runs_batter": runs_batter, "runs_extras": runs_total - runs_batter,
            "runs_total": runs_total,
            "extras_kind": None if is_legal else "wides",
            "wicket": wicket, "dismissal_kind": "bowled" if wicket else None,
            "player_out": "A" if wicket else None,
            "target": 200, "is_legal_delivery": is_legal,
        })
    return pd.DataFrame(out)


def test_phase_indicator():
    s = ChaseState(
        match_id="m", season=2024, venue="V", target=200,
        runs_scored=0, wickets_lost=0, legal_balls_bowled=36,
        striker="A", non_striker="B", striker_balls_faced=0, non_striker_balls_faced=0,
        bowler="X",
    )
    assert s.phase == 1  # over 6 -> middle


def test_recent_features_added_to_columns():
    for col in ("runs_last_12_balls", "wickets_last_18_balls", "boundaries_last_over", "phase"):
        assert col in FEATURE_COLUMNS


def test_replay_chase_tracks_recent_runs():
    # 12 legal balls each scoring 1
    df = make_balls([(1, 1, False, True)] * 12 + [(0, 0, False, True)] * 6)
    states = list(replay_chase(df, label=1))
    # Before the 13th ball (index 12), runs_last_12_balls should be 12
    assert states[12].runs_last_12_balls == 12
    # State at index 17 = before 18th ball; 17 balls bowled, last 12 = balls 5..16
    # = balls 5..11 (7 ones) + balls 12..16 (5 zeros) = 7
    assert states[17].runs_last_12_balls == 7


def test_replay_chase_counts_boundaries_last_over():
    # 4 4s and 2 dots in one over
    df = make_balls([(4, 4, False, True)] * 4 + [(0, 0, False, True)] * 2)
    states = list(replay_chase(df, label=1))
    # State BEFORE the 6th ball: last over so far has 4 boundaries
    assert states[5].boundaries_last_over == 4


def test_replay_chase_counts_wickets_in_window():
    rows = [(0, 0, True, True), (0, 0, False, True)] * 5 + [(0, 0, False, True)] * 8
    df = make_balls(rows)
    states = list(replay_chase(df, label=0))
    # After all 10 legal balls with 5 wickets, before ball 11
    s10 = states[10]
    assert s10.wickets_last_18_balls == 5

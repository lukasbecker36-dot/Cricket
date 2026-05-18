"""Partnership age tracks legal balls played together and resets on wicket."""
import pandas as pd

from src.features.engineering import replay_chase


def make_balls(rows):
    """rows: list of (runs_total, runs_batter, wicket, is_legal)."""
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


def test_partnership_starts_at_zero():
    df = make_balls([(1, 1, False, True)])
    states = list(replay_chase(df, label=1))
    assert states[0].partnership_balls == 0


def test_partnership_counts_legal_balls():
    df = make_balls([(1, 1, False, True)] * 5)
    states = list(replay_chase(df, label=1))
    # Before ball 5 (index 4), 4 legal balls have been bowled with the same pair.
    assert states[4].partnership_balls == 4


def test_partnership_resets_after_wicket():
    # Three legal balls, then wicket on the 4th legal ball, then two more legal balls
    df = make_balls(
        [(0, 0, False, True)] * 3 + [(0, 0, True, True)] + [(0, 0, False, True)] * 2
    )
    states = list(replay_chase(df, label=0))
    # Before wicket-ball (index 3): partnership_balls = 3
    assert states[3].partnership_balls == 3
    # Before next ball (index 4): partnership reset to 0
    assert states[4].partnership_balls == 0
    # Before ball 5 (index 5): new partnership has played 1 ball
    assert states[5].partnership_balls == 1


def test_partnership_ignores_extras():
    # The wide is sorted to the start of the over by replay_chase (ball=0 sorts
    # before ball>=1). What matters: a wide never increments partnership_balls,
    # so by the time we see legal balls 2 and 3 the count reflects legal balls
    # only.
    df = make_balls([(0, 0, False, True), (1, 0, False, False), (0, 0, False, True), (0, 0, False, True)])
    states = list(replay_chase(df, label=1))
    # The state before the very first delivery has 0 partnership balls
    assert states[0].partnership_balls == 0
    # The state before the LAST delivery has 2 partnership balls (two prior legals)
    assert states[-1].partnership_balls == 2
    # And legal_balls_bowled matches partnership_balls when no wicket has occurred
    for s in states:
        assert s.partnership_balls == s.legal_balls_bowled

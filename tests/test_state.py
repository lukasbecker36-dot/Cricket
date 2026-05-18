from src.features.state import ChaseState


def make_state(**kwargs):
    base = dict(
        match_id="m1", season=2024, venue="Wankhede", target=180,
        runs_scored=0, wickets_lost=0, legal_balls_bowled=0,
        striker="A", non_striker="B", striker_balls_faced=0, non_striker_balls_faced=0,
        bowler="X",
    )
    base.update(kwargs)
    return ChaseState(**base)


def test_runs_required_floors_at_zero():
    s = make_state(target=100, runs_scored=120)
    assert s.runs_required == 0


def test_balls_remaining_floors_at_zero():
    s = make_state(legal_balls_bowled=150)
    assert s.balls_remaining == 0


def test_required_run_rate_when_balls_left():
    s = make_state(target=180, runs_scored=120, legal_balls_bowled=90)
    # 60 off 30 balls => 12 rpo
    assert s.required_run_rate == 12.0


def test_required_run_rate_inf_when_no_balls_left():
    s = make_state(target=180, runs_scored=170, legal_balls_bowled=120)
    assert s.required_run_rate == float("inf")


def test_current_run_rate_zero_at_start():
    s = make_state()
    assert s.current_run_rate == 0.0

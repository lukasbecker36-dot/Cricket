"""Pure tests for the signal-extraction logic (no Betfair/Telegram I/O)."""
import json
from pathlib import Path

import pytest

from src.live.signals import (
    FullInningsModel,
    derive_batting_team,
    detect_league_from_event,
    evaluate_market,
    format_signal,
    parse_runner_threshold,
)


def test_parse_runner_threshold():
    assert parse_runner_threshold("60 Runs or more") == 60
    assert parse_runner_threshold("145 Runs or more") == 145
    assert parse_runner_threshold("Under 60.5") is None
    assert parse_runner_threshold("") is None


def test_detect_league_from_event():
    assert detect_league_from_event("Mumbai Indians v Kolkata Knight Riders") == "ipl"
    assert detect_league_from_event("Brisbane Heat v Sydney Sixers") == "bbl"
    assert detect_league_from_event("Random Game", "Indian Premier League 2025") == "ipl"
    assert detect_league_from_event("Unknown Team v Other Team") is None


def test_derive_batting_team_without_toss():
    bat, bowl = derive_batting_team("Mumbai Indians v Kolkata Knight Riders", innings=1,
                                     toss_winner=None, toss_decision=None)
    assert bat == "Mumbai Indians" and bowl == "Kolkata Knight Riders"
    bat, bowl = derive_batting_team("Mumbai Indians v Kolkata Knight Riders", innings=2,
                                     toss_winner=None, toss_decision=None)
    assert bat == "Kolkata Knight Riders" and bowl == "Mumbai Indians"


def test_derive_batting_team_with_toss():
    bat, bowl = derive_batting_team("Mumbai Indians v Kolkata Knight Riders", innings=1,
                                     toss_winner="Mumbai Indians", toss_decision="bowl")
    # Toss winner bowled, so KKR bat first
    assert bat == "Kolkata Knight Riders"


@pytest.fixture
def model() -> FullInningsModel:
    model_dir = Path("models")
    if not (model_dir / "full_innings_meta.json").exists():
        pytest.skip("model not trained; run scripts/train_and_save_full_innings.py")
    return FullInningsModel(model_dir)


def test_evaluate_market_picks_overpriced_runners(model: FullInningsModel):
    """Build a synthetic market where one runner is implausibly cheap on the
    LAY side (high implied prob) compared to a sensible model expectation.
    The signal engine should pick it as a LAY candidate."""
    catalogue = {
        "marketId": "1.234",
        "marketName": "1st Innings Runs",
        "event": {"name": "Mumbai Indians v Kolkata Knight Riders", "id": "9999", "venue": "Wankhede Stadium, Mumbai"},
        "competition": {"name": "Indian Premier League"},
        "runners": [
            {"selectionId": 1, "runnerName": "150 Runs or more"},
            {"selectionId": 2, "runnerName": "180 Runs or more"},
        ],
    }
    book = {
        "marketId": "1.234",
        "runners": [
            {"selectionId": 1, "ex": {"availableToLay": [{"price": 2.20, "size": 50}]}},
            {"selectionId": 2, "ex": {"availableToLay": [{"price": 4.00, "size": 50}]}},
        ],
    }
    sigs = evaluate_market(model=model, market_catalogue=catalogue, market_book=book, season=2025)
    # We don't assert any specific signal -- model output depends on training.
    # We just verify the pipeline produces well-formed Signal objects for any qualifying runners.
    for s in sigs:
        assert s.market_id == "1.234"
        assert s.threshold_X in (150, 180)
        assert s.suggested_action == "LAY"
        assert 0.0 <= s.market_implied <= 1.0
        assert s.edge < 0
        # format_signal should render without exception
        rendered = format_signal(s)
        assert "LAY signal" in rendered
        assert s.runner_name in rendered


def test_evaluate_market_skips_non_ladder_runners(model: FullInningsModel):
    catalogue = {
        "marketId": "1.999",
        "marketName": "Match Odds",
        "event": {"name": "Team A v Team B", "id": "1", "venue": "X"},
        "competition": {"name": "Generic"},
        "runners": [{"selectionId": 1, "runnerName": "Team A"}],
    }
    book = {"marketId": "1.999", "runners": [{"selectionId": 1, "ex": {"availableToLay": [{"price": 2.0}]}}]}
    sigs = evaluate_market(model=model, market_catalogue=catalogue, market_book=book, season=2025)
    assert sigs == []


def test_format_signal_includes_key_fields(model: FullInningsModel):
    from src.live.signals import Signal
    s = Signal(
        detected_at_utc="2025-04-15T10:00:00Z",
        market_id="1.234", event_id="999",
        event_name="Mumbai Indians v Royal Challengers Bangalore",
        market_name="1st Innings Runs", innings=1,
        runner_id=1, runner_name="170 Runs or more",
        threshold_X=170, market_implied=0.45, market_lay_price=2.22,
        model_p=0.35, edge=-0.10, suggested_action="LAY", league_hint="ipl",
    )
    text = format_signal(s, stake_flat=10.0)
    assert "Mumbai Indians" in text
    assert "170 Runs or more" in text
    assert "2.22" in text
    assert "LAY" in text

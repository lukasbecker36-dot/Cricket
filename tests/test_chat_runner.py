"""Tests for the chat-driven runner's pure logic."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.live.chat_runner import (
    ChatRunner,
    evaluate_with_model,
    format_breakdown,
    infer_league_from_teams,
)
from src.live.chat_state import ChatStateStore, Session
from src.live.signals import FullInningsModel
from src.live.vision import MarketExtraction, RunnerExtraction


def test_infer_league_from_teams():
    assert infer_league_from_teams(["Mumbai Indians", "Kolkata Knight Riders"]) == "ipl"
    assert infer_league_from_teams(["Brisbane Heat", "Sydney Sixers"]) == "bbl"
    assert infer_league_from_teams(["Unknown FC", "Another FC"]) is None
    assert infer_league_from_teams([]) is None


def test_chat_state_roundtrip(tmp_path: Path):
    store = ChatStateStore(tmp_path / "state.json")
    assert store.session.teams == []
    store.update_session(teams=["A", "B"], league="ipl", venue="V", season=2025, last_innings=2)
    assert store.session.teams == ["A", "B"]
    assert store.session.league == "ipl"
    # Reopen from disk
    store2 = ChatStateStore(tmp_path / "state.json")
    assert store2.session.teams == ["A", "B"]
    assert store2.session.last_innings == 2


def test_chat_state_pending_lifecycle(tmp_path: Path):
    store = ChatStateStore(tmp_path / "state.json")
    assert store.pending is None
    store.set_pending({"foo": "bar"})
    assert store.pending == {"foo": "bar"}
    store.set_pending(None)
    assert store.pending is None


def test_chat_state_reset(tmp_path: Path):
    store = ChatStateStore(tmp_path / "state.json")
    store.update_session(teams=["A", "B"], league="ipl")
    store.set_pending({"x": 1})
    store.reset_session()
    assert store.session.teams == []
    assert store.pending is None


@pytest.fixture
def model() -> FullInningsModel:
    p = Path("models/full_innings_meta.json")
    if not p.exists():
        pytest.skip("model not trained")
    return FullInningsModel(Path("models"))


def test_evaluate_with_model_runs_end_to_end(model: FullInningsModel):
    extraction = MarketExtraction(
        teams=["Mumbai Indians", "Kolkata Knight Riders"],
        innings=1,
        market_name="1st Innings Runs",
        market_kind="ladder",
        venue="Wankhede Stadium, Mumbai",
        runners=[
            RunnerExtraction(threshold_X=150, back_price=2.10, lay_price=2.20),
            RunnerExtraction(threshold_X=170, back_price=3.30, lay_price=3.50),
            RunnerExtraction(threshold_X=190, back_price=6.40, lay_price=7.00),
        ],
        confidence="high",
        notes="",
    )
    session = Session(teams=["Mumbai Indians", "Kolkata Knight Riders"], league="ipl",
                      venue="Wankhede Stadium, Mumbai", season=2025, last_innings=1)
    signals, breakdown = evaluate_with_model(model, extraction, session)
    # We don't assert specific signals (depends on model). We verify shape.
    assert isinstance(signals, list)
    assert len(breakdown) == 3
    # Every breakdown row either has a skip_reason or full model output
    for b in breakdown:
        assert b["X"] in (150, 170, 190)
        if "skip_reason" not in b:
            assert "model_p" in b and "edge_pp" in b


def test_evaluate_skips_when_no_lay_price(model: FullInningsModel):
    extraction = MarketExtraction(
        teams=["Mumbai Indians", "Kolkata Knight Riders"], innings=1,
        market_name="1st Innings Runs", market_kind="ladder", venue="V",
        runners=[RunnerExtraction(threshold_X=160, back_price=None, lay_price=None)],
        confidence="low", notes="",
    )
    session = Session(teams=extraction.teams, league="ipl", season=2025, last_innings=1)
    signals, breakdown = evaluate_with_model(model, extraction, session)
    assert signals == []
    assert breakdown[0].get("skip_reason") == "no lay price"


def test_evaluate_line_market_produces_back_signal_or_skip(model: FullInningsModel):
    """Line market with explicit Under/Over runners: pipeline should produce a
    breakdown with both sides and (depending on model) optionally a BACK signal."""
    from src.live.chat_runner import evaluate_line_market_with_model
    extraction = MarketExtraction(
        teams=["Mumbai Indians", "Kolkata Knight Riders"],
        innings=1,
        market_name="1st Innings Runs Line",
        market_kind="line",
        venue="Wankhede Stadium, Mumbai",
        runners=[
            RunnerExtraction(threshold_X=164, back_price=None, lay_price=None, side="under"),
            RunnerExtraction(threshold_X=165, back_price=None, lay_price=None, side="over"),
        ],
        confidence="high", notes="",
    )
    session = Session(teams=extraction.teams, league="ipl",
                      venue="Wankhede Stadium, Mumbai", season=2025, last_innings=1)
    signals, breakdown = evaluate_line_market_with_model(model, extraction, session)
    # Two breakdown rows, one per side
    assert len(breakdown) == 2
    sides = {b["side"] for b in breakdown}
    assert sides == {"over", "under"}
    # Edge sign is the model's call; we just verify signals (if any) have the right shape
    for s in signals:
        assert s.suggested_action.startswith("BACK ")
        assert s.edge > 0


def test_format_breakdown_handles_line_rows():
    from src.live.chat_runner import format_breakdown as fb
    rows = [
        {"side": "over", "line_X": 165, "odds": 1.92, "implied": 0.521,
         "model_p": 0.60, "edge_pp": 7.9},
        {"side": "under", "line_X": 164, "odds": 1.92, "implied": 0.521,
         "model_p": 0.40, "edge_pp": -12.1},
    ]
    out = fb(rows)
    assert "over" in out
    assert "under" in out
    assert "BACK OVER" in out  # edge_pp 7.9 > 5 triggers tag


def test_format_breakdown_handles_mixed_rows():
    rows = [
        {"X": 100, "lay": 1.5, "implied": 0.667, "model_p": 0.6, "edge_pp": -6.7},
        {"X": 150, "lay": None, "skip_reason": "no lay price"},
        {"X": 200, "lay": 2.5, "implied": 0.4, "model_p": 0.45, "edge_pp": 5.0},
    ]
    out = format_breakdown(rows)
    assert "100" in out
    assert "no lay price" in out
    assert "200" in out
    assert "LAY" in out  # 100's -6.7pp triggers the tag

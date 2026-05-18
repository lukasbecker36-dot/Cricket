"""Parser test against a hand-rolled minimal Cricsheet-shaped match."""
from __future__ import annotations

from src.ingestion.cricsheet import parse_match
from src.ingestion.validate import validate_match


def minimal_raw() -> dict:
    delivery = {
        "batter": "A",
        "non_striker": "B",
        "bowler": "X",
        "runs": {"batter": 1, "extras": 0, "total": 1},
    }
    over_block = {"over": 0, "deliveries": [delivery] * 6}
    return {
        "info": {
            "season": 2024,
            "venue": "V",
            "teams": ["Team1", "Team2"],
            "dates": ["2024-04-01"],
            "toss": {"winner": "Team1", "decision": "bat"},
            "outcome": {"winner": "Team2"},
        },
        "innings": [
            {"team": "Team1", "overs": [over_block]},
            {"team": "Team2", "overs": [over_block]},
        ],
    }


def test_parse_minimal_match():
    meta, balls = parse_match("test_match", minimal_raw())
    assert meta.match_id == "test_match"
    assert meta.season == 2024
    assert meta.winner == "Team2"
    assert len(balls) == 12
    inn2 = [b for b in balls if b.innings == 2]
    assert inn2[0].target == 7  # 6 runs + 1
    validate_match(meta, balls)

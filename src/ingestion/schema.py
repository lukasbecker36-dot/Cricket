"""Canonical ball-by-ball schema. The only place external data quirks become normalised."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DismissalKind = Literal[
    "bowled",
    "caught",
    "caught and bowled",
    "lbw",
    "stumped",
    "run out",
    "hit wicket",
    "retired hurt",
    "retired out",
    "retired not out",
    "obstructing the field",
    "handled the ball",
    "hit the ball twice",
    "timed out",
]


class Ball(BaseModel):
    """One delivery. Every ingested match flattens to a sequence of these."""

    match_id: str
    season: int
    venue: str
    innings: int  # 1 or 2
    over: int  # 0..19
    ball: int  # 1..6 within the over (legal deliveries only; extras tracked separately)
    batting_team: str
    bowling_team: str
    striker: str
    non_striker: str
    bowler: str
    runs_batter: int = 0
    runs_extras: int = 0
    runs_total: int = 0
    extras_kind: str | None = None  # wide, noball, bye, legbye, penalty
    wicket: bool = False
    dismissal_kind: DismissalKind | None = None
    player_out: str | None = None
    target: int | None = None  # set on innings-2 balls
    is_legal_delivery: bool = True


class MatchMeta(BaseModel):
    match_id: str
    season: int
    date: str
    venue: str
    teams: list[str]
    toss_winner: str
    toss_decision: str
    winner: str | None = None
    result: str | None = None  # normal / tie / no result
    target_innings_2: int | None = None


BALL_COLUMNS: list[str] = list(Ball.model_fields.keys())
META_COLUMNS: list[str] = list(MatchMeta.model_fields.keys())

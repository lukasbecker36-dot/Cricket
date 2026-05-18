"""LiveFeedAdapter interface. The only abstraction the rest of the system depends on."""
from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from .schema import Ball


class LiveSnapshot(BaseModel):
    """A single poll response: match state at a moment in time."""

    match_id: str
    polled_at: datetime
    feed_timestamp: datetime | None
    balls_so_far: list[Ball]
    striker: str
    non_striker: str
    bowler: str
    target: int

    @property
    def staleness_seconds(self) -> float | None:
        if self.feed_timestamp is None:
            return None
        return (self.polled_at - self.feed_timestamp).total_seconds()


@runtime_checkable
class LiveFeedAdapter(Protocol):
    """Polled live feed. Implementations cache every response for post-match analysis."""

    def poll(self, match_id: str) -> LiveSnapshot: ...

    def list_live_matches(self) -> list[str]: ...

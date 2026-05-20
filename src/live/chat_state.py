"""Per-chat session state, persisted to JSON.

Tracks:
- session: the current match the user is screenshotting (teams, league, innings, venue, season)
- pending_extraction: a vision extraction waiting for user confirmation
- history: last N analysed screenshots (light record)

A single global state file is fine because we only serve one chat_id.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Session:
    teams: list[str] = field(default_factory=list)
    league: str | None = None
    venue: str | None = None
    season: int | None = None
    last_innings: int | None = None
    last_updated_utc: str = ""

    def to_dict(self) -> dict:
        return {
            "teams": self.teams,
            "league": self.league,
            "venue": self.venue,
            "season": self.season,
            "last_innings": self.last_innings,
            "last_updated_utc": self.last_updated_utc,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Session":
        return cls(
            teams=list(d.get("teams") or []),
            league=d.get("league"),
            venue=d.get("venue"),
            season=d.get("season"),
            last_innings=d.get("last_innings"),
            last_updated_utc=d.get("last_updated_utc", ""),
        )


class ChatStateStore:
    """JSON-backed state for a single chat. Atomic writes via temp file."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._state: dict = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return {"session": Session().to_dict(), "pending": None, "history": []}
        try:
            return json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("failed to load state, starting fresh: %s", e)
            return {"session": Session().to_dict(), "pending": None, "history": []}

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, indent=2))
        tmp.replace(self.path)

    @property
    def session(self) -> Session:
        return Session.from_dict(self._state.get("session") or {})

    def update_session(self, **kwargs) -> Session:
        s = self.session
        for k, v in kwargs.items():
            if v is not None:
                setattr(s, k, v)
        s.last_updated_utc = datetime.now(timezone.utc).isoformat()
        self._state["session"] = s.to_dict()
        self._save()
        return s

    def reset_session(self) -> None:
        self._state["session"] = Session().to_dict()
        self._state["pending"] = None
        self._save()

    @property
    def pending(self) -> dict | None:
        return self._state.get("pending")

    def set_pending(self, payload: dict | None) -> None:
        self._state["pending"] = payload
        self._save()

    def append_history(self, entry: dict, cap: int = 50) -> None:
        h = self._state.setdefault("history", [])
        h.append(entry)
        if len(h) > cap:
            self._state["history"] = h[-cap:]
        self._save()

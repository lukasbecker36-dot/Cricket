"""Cricket data client — RapidAPI 'free-cricbuzz-cricket-api'.

Confirmed endpoints (envelope is always {"status": "...", "response": ...}):
  GET /cricket-matches-upcoming
  GET /cricket-matches-recent
  GET /cricket-livescores
  GET /cricket-match-info?matchid=<id>
  GET /cricket-series                       (series list)
  GET /cricket-scorecard?matchid=<id>       (NEEDS CONFIRMATION — see note)

Auth headers: x-rapidapi-key, x-rapidapi-host. Key from RAPIDAPI_KEY env.

NOTE: this is a Cricbuzz-scraper API. The nested field paths below follow the
common Cricbuzz schema (team1/team2, venueInfo, tossResults, state/status) and
are marked VERIFY — they must be checked against a real live/upcoming match,
since all feeds were empty when the client was written. `phase_scores` needs a
scorecard/commentary endpoint that returns per-over progression; confirm its
exact name on RapidAPI ('Get Scorecard' / 'Get Commentary').
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

HOST = "free-cricbuzz-cricket-api.p.rapidapi.com"
BASE = f"https://{HOST}"


class CricketDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class MatchSummary:
    match_id: str
    name: str
    teams: tuple[str, str]
    venue: str
    date: str
    fmt: str
    state: str


class CricketDataClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("RAPIDAPI_KEY", "")
        if not self.api_key:
            raise CricketDataError("RAPIDAPI_KEY not set")
        self.calls = 0

    def _get(self, path: str, **params) -> dict:
        url = f"{BASE}/{path}"
        if params:
            url += "?" + urlencode(params)
        self.calls += 1
        try:
            req = Request(url, headers={
                "x-rapidapi-key": self.api_key,
                "x-rapidapi-host": HOST,
                "Content-Type": "application/json",
            })
            with urlopen(req, timeout=20) as r:
                data = json.loads(r.read())
        except Exception as e:
            raise CricketDataError(f"{path} request failed: {e}") from e
        if isinstance(data, dict) and data.get("status") not in ("success", None):
            raise CricketDataError(f"{path}: {data.get('message') or data.get('status')}")
        return data

    # ---- raw fetchers ----
    def upcoming_raw(self) -> list:
        return self._get("cricket-matches-upcoming").get("response", []) or []

    def recent_raw(self) -> list:
        return self._get("cricket-matches-recent").get("response", []) or []

    def livescores_raw(self) -> list:
        return self._get("cricket-livescores").get("response", []) or []

    def match_info_raw(self, match_id: str) -> dict:
        return self._get("cricket-match-info", matchid=match_id).get("response", {}) or {}

    def scorecard_raw(self, match_id: str) -> dict:
        # VERIFY endpoint name on RapidAPI; common variants below.
        return self._get("cricket-scorecard", matchid=match_id).get("response", {}) or {}

    # ---- normalised helpers (VERIFY field paths against live data) ----
    def match_info(self, match_id: str) -> dict:
        """Return a flat dict the bot understands. Field paths follow the
        Cricbuzz schema and must be verified against a live match."""
        r = self.match_info_raw(match_id)
        mi = r.get("matchInfo", r) if isinstance(r, dict) else {}
        t1 = (mi.get("team1") or {}).get("teamName") or (mi.get("team1") or {}).get("name")
        t2 = (mi.get("team2") or {}).get("teamName") or (mi.get("team2") or {}).get("name")
        venue = (mi.get("venueInfo") or {}).get("ground") or mi.get("venue") or ""
        toss = mi.get("tossResults") or {}
        return {
            "match_id": str(match_id),
            "name": mi.get("matchDesc") or f"{t1} v {t2}",
            "teams": [x for x in (t1, t2) if x],
            "venue": venue,
            "series": mi.get("seriesName") or "",
            "state": mi.get("state") or "",
            "status": mi.get("status") or "",
            "matchEnded": (mi.get("state", "").lower() in ("complete", "ended", "finished")),
            "tossWinner": toss.get("tossWinnerName") or toss.get("tossWinner"),
            "tossChoice": toss.get("decision") or toss.get("tossDecision"),
            "date": str(mi.get("startDate") or mi.get("matchStartTimestamp") or ""),
            "_raw": mi,
        }


def batting_first(info: dict) -> tuple[str | None, str | None]:
    toss_winner = info.get("tossWinner")
    toss_choice = (info.get("tossChoice") or "").lower()
    teams = info.get("teams") or []
    if not toss_winner or toss_choice not in ("bat", "bowl", "field", "batting", "bowling") or len(teams) != 2:
        return (None, None)
    other = teams[1] if toss_winner == teams[0] else teams[0]
    if toss_choice.startswith("bat"):
        return (toss_winner, other)
    return (other, toss_winner)


def phase_scores_from_scorecard(scard: dict) -> dict[tuple[int, str], float]:
    """Compute (innings, phase) -> runs at end of 6/10/15 overs + full innings.

    PLACEHOLDER: the exact scorecard/commentary schema is unconfirmed (feeds
    were empty when written). Expected to read per-innings over-by-over runs.
    Wire this up once a real scorecard response is captured.
    """
    raise NotImplementedError(
        "phase_scores: confirm the scorecard/commentary endpoint + schema "
        "against a real match, then implement over-by-over extraction here")

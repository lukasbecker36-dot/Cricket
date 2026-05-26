"""Cricket Data API client (cricketdata.org / cricapi.com v1).

Free tier = daily request quota, so callers must be frugal: poll fixtures a
few times a day, then only hit match endpoints at the key moments (toss,
innings end, match end). Every call is logged so quota use is visible.

Endpoints used:
  /v1/currentMatches   list of live/upcoming matches
  /v1/match_info       teams, venue, toss, status, score
  /v1/match_bbb        ball-by-ball (to compute phase scores for settling)

API key from CRICKET_API_KEY env var (never commit it).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

BASE = "https://api.cricapi.com/v1"


class CricketDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class MatchSummary:
    match_id: str
    name: str
    teams: tuple[str, str]
    venue: str
    date: str
    started: bool
    ended: bool
    status: str


class CricketDataClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("CRICKET_API_KEY", "")
        if not self.api_key:
            raise CricketDataError("CRICKET_API_KEY not set")
        self.calls = 0

    def _get(self, endpoint: str, **params) -> dict:
        params["apikey"] = self.api_key
        url = f"{BASE}/{endpoint}?{urlencode(params)}"
        self.calls += 1
        try:
            req = Request(url, headers={"User-Agent": "cricket-trading/1.0"})
            with urlopen(req, timeout=20) as r:
                data = json.loads(r.read())
        except Exception as e:
            raise CricketDataError(f"{endpoint} request failed: {e}") from e
        if data.get("status") != "success":
            raise CricketDataError(f"{endpoint} returned: {data.get('status')} / {data.get('reason') or data}")
        # surface quota info when present
        info = data.get("info") or {}
        if "hitsToday" in info and "hitsLimit" in info:
            logger.info("API quota: %s/%s today", info.get("hitsToday"), info.get("hitsLimit"))
        return data

    def current_matches(self) -> list[MatchSummary]:
        data = self._get("currentMatches", offset=0)
        out = []
        for m in data.get("data", []):
            teams = m.get("teams") or []
            if len(teams) != 2:
                continue
            out.append(MatchSummary(
                match_id=m.get("id", ""),
                name=m.get("name", ""),
                teams=(teams[0], teams[1]),
                venue=m.get("venue", ""),
                date=m.get("date", ""),
                started=bool(m.get("matchStarted", False)),
                ended=bool(m.get("matchEnded", False)),
                status=m.get("status", ""),
            ))
        return out

    def match_info(self, match_id: str) -> dict:
        return self._get("match_info", id=match_id).get("data", {})

    def ball_by_ball(self, match_id: str) -> list[dict]:
        return self._get("match_bbb", id=match_id).get("data", {}).get("bbb", [])


def batting_first(info: dict) -> tuple[str | None, str | None]:
    """Return (batting_team, bowling_team) for innings 1 from toss info.
    Returns (None, None) if toss/decision not yet available."""
    toss_winner = info.get("tossWinner")
    toss_choice = (info.get("tossChoice") or "").lower()  # 'bat' / 'bowl' / 'field'
    teams = info.get("teams") or []
    if not toss_winner or toss_choice not in ("bat", "bowl", "field") or len(teams) != 2:
        return (None, None)
    other = teams[1] if toss_winner == teams[0] else teams[0]
    if toss_choice == "bat":
        return (toss_winner, other)
    return (other, toss_winner)  # chose to bowl/field -> other team bats first


def phase_scores_from_bbb(bbb: list[dict]) -> dict[tuple[int, str], float]:
    """Compute cumulative runs at the end of overs 6/10/15 and the full innings,
    for each innings, from ball-by-ball data.

    Returns {(innings, phase_label): runs}. Phase keys match line_projector /
    trade_log: '6 over', '10 over', '15 over', '20 over (full)'.
    """
    # bbb entries have 'inning' (e.g. '1' or team string), 'over', 'ball', 'runs'
    # Normalise innings to 1/2 by order of appearance.
    inning_order: list[str] = []
    cum: dict[str, list[tuple[float, float]]] = {}  # inning_key -> [(over_float, runs_total)]
    running: dict[str, float] = {}
    for b in bbb:
        ik = str(b.get("inning", ""))
        if ik not in inning_order:
            inning_order.append(ik)
            running[ik] = 0.0
            cum[ik] = []
        runs = float(b.get("runs", 0) or 0)
        # some feeds separate extras; 'runs' is usually total off the ball
        running[ik] += runs
        over = float(b.get("over", 0) or 0)  # completed overs count
        cum[ik].append((over, running[ik]))

    def score_at_over(entries, target_over):
        # last cumulative score where completed overs < target_over (i.e. within the phase)
        best = None
        for over, total in entries:
            if over < target_over:
                best = total
            else:
                break
        return best

    out: dict[tuple[int, str], float] = {}
    for idx, ik in enumerate(inning_order[:2]):
        inn = idx + 1
        entries = cum[ik]
        for phase, ov in [("6 over", 6), ("10 over", 10), ("15 over", 15)]:
            s = score_at_over(entries, ov)
            if s is not None:
                out[(inn, phase)] = s
        if entries:
            out[(inn, "20 over (full)")] = entries[-1][1]
    return out

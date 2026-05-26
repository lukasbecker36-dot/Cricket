"""Match bot: auto-projects par lines around real fixtures and keeps a trade log.

Flow (par-lines-only; user compares to Betfair themselves):
  1. Poll fixtures. When a watched match's toss is known -> send inn1 par lines.
  2. User logs what they traded -> appended to the trade log.
  3. When inn1 ends (auto-detected or user-reported score) -> send inn2 par lines.
  4. User logs inn2 trades.
  5. When the match ends -> fetch ball-by-ball, compute phase scores, settle the
     log, and report W/L + running P&L.

API quota is precious (free tier = daily cap), so match endpoints are only hit
at state transitions, not on a tight loop.

Trade command syntax (forgiving):
  trade 1 6 under 55.5 @2.0 £5     -> innings 1, 6-over, under 55.5, odds 2.0, £5
  trade 2 10 over 95              -> odds/stake default to config
Other commands: log | status | watch <match_id> | help
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.ingestion.teams import canonical_team
from src.live.cricket_data import (CricketDataClient, batting_first,
                                    phase_scores_from_scorecard)
from src.live.line_projector import format_projections, project_innings
from src.live.signals import detect_league_from_event, load_models
from src.live.trade_log import Trade, log_trade, settle_match, summary

logger = logging.getLogger(__name__)

STATE_PATH = Path("data/match_bot_state.json")
PHASE_ALIASES = {"6": "6 over", "10": "10 over", "15": "15 over",
                 "20": "20 over (full)", "full": "20 over (full)"}
DEFAULT_ODDS = 2.0
DEFAULT_STAKE = 5.0

# API/event team & venue names -> the names our models were trained on.
# Seeded with known mismatches; extend as real API output is seen.
TEAM_NAME_MAP: dict[str, str] = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}
VENUE_NAME_MAP: dict[str, str] = {}


def normalize_team(name: str) -> str:
    return canonical_team(TEAM_NAME_MAP.get(name, name))


def normalize_venue(name: str) -> str:
    return VENUE_NAME_MAP.get(name, name)


@dataclass
class WatchedMatch:
    match_id: str
    event: str
    venue: str
    league: str | None
    season: int
    state: str = "watching"           # watching -> inn1_sent -> inn2_sent -> done
    batting_first: str | None = None
    bowling_first: str | None = None
    target: float | None = None       # inn1 total + 1, for inn2 projections


def _parse_trade(text: str, match: WatchedMatch) -> Trade | str:
    """Parse a 'trade ...' command into a Trade, or return an error string.

    e.g. 'trade 1 6 under 55.5 @2.0 £5' -> inn1, 6-over, under 55.5, odds 2.0, £5
    Odds (@) and stake (£) are optional and default from config.
    """
    low = text.lower()
    # pull odds/stake by their markers first, then strip them out
    odds = DEFAULT_ODDS
    stake = DEFAULT_STAKE
    mo = re.search(r"@\s*(\d+(?:\.\d+)?)", low)
    if mo:
        odds = float(mo.group(1))
    ms = re.search(r"£\s*(\d+(?:\.\d+)?)", low)
    if ms:
        stake = float(ms.group(1))
    stripped = re.sub(r"@\s*\d+(?:\.\d+)?", " ", low)
    stripped = re.sub(r"£\s*\d+(?:\.\d+)?", " ", stripped)
    toks = stripped.replace("trade", " ").split()

    side = next((x for x in toks if x in ("over", "under")), None)
    phase_tok = next((x for x in toks if x in PHASE_ALIASES), None)
    innings = next((int(x) for x in toks if x in ("1", "2")), None)
    if side is None or phase_tok is None or innings is None:
        return "couldn't parse — try: trade 1 6 under 55.5 @2.0 £5"
    # remaining numbers, excluding the innings and phase tokens -> the line
    consumed = {str(innings), phase_tok}
    line_nums = [float(x) for x in toks
                 if re.fullmatch(r"\d+(?:\.\d+)?", x) and x not in consumed]
    # prefer a .5 line; else the largest remaining number
    line = next((n for n in line_nums if n != int(n)), None)
    if line is None:
        line = max(line_nums) if line_nums else None
    if line is None:
        return "couldn't parse the line value — try: trade 1 6 under 55.5 @2.0 £5"
    return Trade(
        logged_at_utc="", match_id=match.match_id, event=match.event,
        innings=innings, phase=PHASE_ALIASES[phase_tok], side=side,
        line=line, odds=odds, stake=stake,
    )


class MatchBot:
    def __init__(self, telegram, api: CricketDataClient, model_dir: Path = Path("models")):
        self.tg = telegram
        self.api = api
        self.registry = load_models(model_dir)
        self.watched: dict[str, WatchedMatch] = {}
        self._load_state()

    # ---- persistence ----
    def _load_state(self):
        if STATE_PATH.exists():
            blob = json.loads(STATE_PATH.read_text())
            self.watched = {k: WatchedMatch(**v) for k, v in blob.items()}

    def _save_state(self):
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps({k: v.__dict__ for k, v in self.watched.items()}, indent=2))

    # ---- projections ----
    def _send_innings_lines(self, m: WatchedMatch, innings: int):
        if innings == 1:
            bat, bowl = m.batting_first, m.bowling_first
            target = None
        else:
            bat, bowl = m.bowling_first, m.batting_first  # chasing team bats 2nd
            target = m.target
        projs = project_innings(
            registry=self.registry, batting_team=normalize_team(bat),
            bowling_team=normalize_team(bowl), venue=normalize_venue(m.venue),
            season=m.season, innings=innings, league=m.league, target=target,
        )
        header = f"{m.event} — {'1st' if innings == 1 else '2nd'} Innings ({bat} batting)"
        self.tg.send(format_projections(projs, header))

    # ---- polling (call at intervals; frugal) ----
    def poll(self):
        for mid, m in list(self.watched.items()):
            if m.state == "done":
                continue
            try:
                info = self.api.match_info(mid)
            except Exception as e:
                logger.warning("poll %s failed: %s", mid, e)
                continue
            if m.state == "watching":
                bat, bowl = batting_first(info)
                if bat:
                    m.batting_first, m.bowling_first = bat, bowl
                    m.state = "inn1_sent"
                    self._send_innings_lines(m, 1)
            ended = bool(info.get("matchEnded"))
            if ended and m.state != "done":
                self._settle(m)
                m.state = "done"
        self._save_state()

    def _settle(self, m: WatchedMatch):
        try:
            scard = self.api.scoreboard_raw(m.match_id)
            scores = phase_scores_from_scorecard(scard)
        except NotImplementedError:
            self.tg.send(f"ℹ️ {m.event} ended — auto-settle not wired yet "
                         f"(scorecard endpoint pending). Settle manually with the score.")
            return
        except Exception as e:
            self.tg.send(f"⚠️ couldn't fetch scorecard to settle {m.event}: {e}")
            return
        settled = settle_match(m.match_id, scores)
        if settled:
            lines = [f"*{m.event} — settled*"]
            for t in settled:
                res = "WON" if t.won else ("VOID" if t.won is None else "lost")
                lines.append(f"  inn{t.innings} {t.phase} {t.side} {t.line:g}: {res} (£{t.pnl:+.2f})")
            lines.append("")
            lines.append(summary())
            self.tg.send("\n".join(lines))

    # ---- message handling ----
    def handle_message(self, text: str):
        t = text.strip()
        low = t.lower()
        if low in ("help", "/help"):
            self.tg.send("Commands: trade <inn> <phase> <over/under> <line> [@odds] [£stake] | "
                         "inn1 <score> | status | watch <match_id>")
        elif low in ("status", "/status", "log", "/log"):
            self.tg.send(summary())
        elif low.startswith("trade"):
            m = self._active_match()
            if m is None:
                self.tg.send("no active match to attach the trade to")
                return
            res = _parse_trade(t, m)
            if isinstance(res, str):
                self.tg.send(res)
            else:
                log_trade(res)
                self.tg.send(f"logged: inn{res.innings} {res.phase} {res.side} {res.line:g} "
                             f"@{res.odds:g} £{res.stake:g}")
        elif low.startswith("inn1"):
            m = self._active_match()
            nums = re.findall(r"\d+", t)
            if m and nums:
                m.target = float(nums[-1]) + 1.0
                m.state = "inn2_sent"
                self._send_innings_lines(m, 2)
                self._save_state()
            else:
                self.tg.send("usage: inn1 <score>  (e.g. 'inn1 180')")
        elif low.startswith("watch"):
            mid = t.split()[-1]
            self._add_watch(mid)

    def _active_match(self) -> WatchedMatch | None:
        for m in self.watched.values():
            if m.state in ("inn1_sent", "inn2_sent"):
                return m
        return None

    def _add_watch(self, match_id: str):
        try:
            info = self.api.match_info(match_id)
        except Exception as e:
            self.tg.send(f"couldn't load match {match_id}: {e}")
            return
        teams = info.get("teams") or []
        event = info.get("name") or " v ".join(teams)
        season = int((info.get("dateTimeGMT") or info.get("date") or "2026")[:4])
        league = detect_league_from_event(event, info.get("series") or "")
        self.watched[match_id] = WatchedMatch(
            match_id=match_id, event=event, venue=info.get("venue", ""),
            league=league, season=season)
        self._save_state()
        self.tg.send(f"👀 watching {event} ({league or 'league?'}) — will send inn1 lines at toss")

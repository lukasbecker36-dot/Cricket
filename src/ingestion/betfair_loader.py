"""Parse Betfair BASIC stream files (one .bz2 per market) into clean time-series.

BASIC plan only gives last-traded price (LTP), not the back/lay order book.
We expose what's there and let the backtest layer make assumptions about
spread / slippage rather than fabricating an order book here.
"""
from __future__ import annotations

import bz2
import json
import logging
import tarfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BetfairMarket:
    """One Match Odds market: metadata + LTP time-series."""

    market_id: str
    event_id: str
    event_name: str
    market_time: str  # ISO 8601 UTC
    runners: dict[int, str]  # selection_id -> team name
    ticks: pd.DataFrame  # columns: pt_ms, selection_id, ltp


def parse_stream(raw: bytes) -> BetfairMarket | None:
    """Parse one decompressed stream (concatenated mcm JSON lines)."""
    market_id = ""
    event_id = ""
    event_name = ""
    market_time = ""
    runners: dict[int, str] = {}
    rows: list[tuple[int, int, float]] = []

    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        pt = obj.get("pt")
        for change in obj.get("mc", []):
            market_id = market_id or change.get("id", "")
            md = change.get("marketDefinition")
            if md is not None:
                event_id = event_id or str(md.get("eventId", ""))
                event_name = event_name or md.get("eventName", "")
                market_time = market_time or md.get("marketTime", "")
                for r in md.get("runners", []):
                    rid = r.get("id")
                    name = r.get("name")
                    if rid is not None and name and rid not in runners:
                        runners[int(rid)] = name
            for rc in change.get("rc", []) or []:
                ltp = rc.get("ltp")
                rid = rc.get("id")
                if ltp is not None and rid is not None and pt is not None:
                    rows.append((int(pt), int(rid), float(ltp)))

    if not market_id or not rows:
        return None
    ticks = pd.DataFrame(rows, columns=["pt_ms", "selection_id", "ltp"])
    return BetfairMarket(
        market_id=market_id,
        event_id=event_id,
        event_name=event_name,
        market_time=market_time,
        runners=runners,
        ticks=ticks,
    )


def load_from_tar(tar_path: Path, archive_path: str) -> BetfairMarket | None:
    with tarfile.open(tar_path, "r") as tar:
        member = tar.getmember(archive_path)
        f = tar.extractfile(member)
        if f is None:
            return None
        raw = bz2.decompress(f.read())
    return parse_stream(raw)


def implied_market_probability(
    ticks: pd.DataFrame, chasing_selection_id: int
) -> pd.DataFrame:
    """Reduce a two-runner LTP stream to (pt_ms, market_prob_chasing).

    Implied prob_i = 1 / ltp_i; normalised by sum to remove the overround so the
    two probabilities sum to 1. Forward-fills missing legs across ticks.
    """
    wide = (
        ticks.pivot_table(index="pt_ms", columns="selection_id", values="ltp", aggfunc="last")
        .sort_index()
        .ffill()
        .dropna()
    )
    if chasing_selection_id not in wide.columns or wide.shape[1] != 2:
        return pd.DataFrame(columns=["pt_ms", "market_prob_chasing"])
    other = next(c for c in wide.columns if c != chasing_selection_id)
    p_chasing = 1.0 / wide[chasing_selection_id]
    p_other = 1.0 / wide[other]
    norm = p_chasing + p_other
    out = (p_chasing / norm).reset_index()
    out.columns = ["pt_ms", "market_prob_chasing"]
    return out


def match_betfair_to_cricsheet(
    betfair_index: list[dict],
    cricsheet_matches: pd.DataFrame,
    team_aliases: dict[str, str],
) -> pd.DataFrame:
    """Join Betfair markets to Cricsheet match_ids by (date, both teams).

    team_aliases maps Betfair franchise names to the canonical names Cricsheet
    uses (which can differ across seasons -- e.g. 'Bengaluru' vs 'Bangalore').
    Returns one row per resolved Betfair market with the cricsheet_match_id.
    """
    cs = cricsheet_matches.copy()
    cs["date_only"] = pd.to_datetime(cs["date"]).dt.date.astype(str)
    cs["teams_set"] = cs["teams"].apply(
        lambda ts: frozenset(ts) if hasattr(ts, "__iter__") and not isinstance(ts, str) else frozenset()
    )

    rows: list[dict] = []
    misses: list[str] = []
    for entry in betfair_index:
        market_time = entry.get("market_time", "")
        try:
            mt_date = datetime.fromisoformat(market_time.replace("Z", "+00:00")).date()
        except ValueError:
            continue
        bf_teams = frozenset(team_aliases.get(t, t) for _, t in entry.get("runners", []))
        if len(bf_teams) < 2:
            continue
        match = cs[(cs["date_only"] == str(mt_date)) & (cs["teams_set"] == bf_teams)]
        if match.empty:
            # Allow a one-day window for late-evening matches that span midnight UTC
            for delta in (-1, 1):
                from datetime import timedelta
                d2 = str(mt_date + timedelta(days=delta))
                match = cs[(cs["date_only"] == d2) & (cs["teams_set"] == bf_teams)]
                if not match.empty:
                    break
        if match.empty:
            misses.append(f"{entry.get('event_name')} on {mt_date}")
            continue
        rows.append({
            "archive_path": entry["archive_path"],
            "market_id": entry["market_id"],
            "cricsheet_match_id": match.iloc[0]["match_id"],
            "event_name": entry["event_name"],
            "market_time": market_time,
            "runners_json": json.dumps(entry["runners"]),
        })
    if misses:
        logger.info("could not resolve %d Betfair markets; first few: %s", len(misses), misses[:5])
    return pd.DataFrame(rows)

"""Extract per-runner time-anchor data for all 'Innings Runs Total' market types.

Generalizes pp_time_anchors.py to multiple targets:
  6 Overs   -> target_balls=36   (powerplay)
  10 Overs  -> target_balls=60
  15 Overs  -> target_balls=90
  Full Innings (no over qualifier) -> target_balls=120

Walks the all-markets archive ONCE and writes a unified parquet keyed by
market_type. Downstream analysis can then filter by type.
"""
from __future__ import annotations

import bz2
import json
import logging
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingestion.storage import read_balls
from src.logging_setup import configure_logging

logger = logging.getLogger(__name__)

ANCHORS_MIN = [60, 30, 10, 1]

# (market_type_label, balls, name_regex)
MARKET_PATTERNS = [
    ("6_over",        36,  re.compile(r"(1st|first|2nd|second).*?6 ?Overs?", re.IGNORECASE)),
    ("10_over",       60,  re.compile(r"(1st|first|2nd|second).*?10 ?Overs?", re.IGNORECASE)),
    ("15_over",       90,  re.compile(r"(1st|first|2nd|second).*?15 ?Overs?", re.IGNORECASE)),
    ("full_innings", 120,  re.compile(r"^(1st|first|2nd|second) +Innings +Runs$", re.IGNORECASE)),
]


def classify_market(name: str) -> tuple[str, int] | None:
    for label, balls, pat in MARKET_PATTERNS:
        if pat.search(name):
            return label, balls
    return None


def parse_runner_threshold(name: str) -> int | None:
    m = re.match(r"^(\d+)\s+Runs?\s+or\s+more", name, re.IGNORECASE)
    return int(m.group(1)) if m else None


def header_definition(bz2_bytes: bytes):
    d = bz2.BZ2Decompressor()
    head = d.decompress(bz2_bytes[:131072])
    nl = head.find(b"\n")
    if nl == -1:
        return None
    obj = json.loads(head[:nl].decode("utf-8"))
    for change in obj.get("mc", []):
        md = change.get("marketDefinition")
        if md is not None:
            runners = [(r.get("id"), r.get("name", "")) for r in md.get("runners", [])]
            return md, runners
    return None


def collect_history(raw: bytes):
    history: dict[int, list[tuple[int, float]]] = {}
    inplay_start: int | None = None
    last_pt: int | None = None
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("op") != "mcm":
            continue
        pt = obj.get("pt")
        if pt is not None:
            last_pt = int(pt)
        for change in obj.get("mc", []):
            md = change.get("marketDefinition")
            if md is not None and inplay_start is None and md.get("inPlay") is True:
                inplay_start = last_pt
            for rc in change.get("rc", []) or []:
                ltp = rc.get("ltp")
                rid = rc.get("id")
                if ltp is None or rid is None or last_pt is None:
                    continue
                history.setdefault(int(rid), []).append((last_pt, float(ltp)))
    for rid in history:
        history[rid].sort(key=lambda x: x[0])
    return history, inplay_start


def ltp_at_or_before(hist, target_ms):
    best = None
    for pt, ltp in hist:
        if pt <= target_ms:
            best = ltp
        else:
            break
    return best


def compute_actual_total(match_id: str, innings: int, target_balls: int, balls: pd.DataFrame) -> int | None:
    inn = balls[(balls["match_id"] == match_id) & (balls["innings"] == innings)]
    if inn.empty:
        return None
    inn = inn.sort_values(["over", "ball", "is_legal_delivery"], ascending=[True, True, False]).reset_index(drop=True)
    legal_idx = np.where(inn["is_legal_delivery"].values)[0]
    if len(legal_idx) < target_balls:
        # For 'full_innings', allow shorter innings (the actual total at innings end)
        if target_balls == 120 and len(legal_idx) >= 30:  # at least 5 overs played
            return int(inn["runs_total"].sum())
        return None
    cut = legal_idx[target_balls - 1] + 1
    return int(inn.iloc[:cut]["runs_total"].sum())


def main() -> int:
    configure_logging()
    archive = Path("data/raw/betfair/betfair_all_markets.dat")
    leagues = ["ipl", "bbl", "psl", "cpl", "ntb"]

    all_balls, all_matches = [], []
    for league in leagues:
        d = Path("data/processed") if league == "ipl" else Path(f"data/processed/{league}")
        b = read_balls(d)
        m = pd.read_parquet(d / "matches.parquet")
        if not b.empty: all_balls.append(b)
        if not m.empty: all_matches.append(m)
    balls = pd.concat(all_balls, ignore_index=True)
    matches = pd.concat(all_matches, ignore_index=True)
    matches["teams_set"] = matches["teams"].apply(
        lambda ts: frozenset(ts) if hasattr(ts, "__iter__") and not isinstance(ts, str) else frozenset()
    )
    lookup = matches[["match_id", "date", "teams_set"]].drop_duplicates("match_id")

    all_teams = set()
    for ts in matches["teams"]:
        if hasattr(ts, "__iter__") and not isinstance(ts, str):
            all_teams.update(ts)
    teams_map = {t: t for t in all_teams}
    teams_map["Royal Challengers Bengaluru"] = "Royal Challengers Bangalore"

    league_by_match = {}
    for league in leagues:
        d = Path("data/processed") if league == "ipl" else Path(f"data/processed/{league}")
        m = pd.read_parquet(d / "matches.parquet")
        for mid in m["match_id"].astype(str).unique():
            league_by_match[mid] = league

    rows = []
    n_scanned = 0
    n_classified = 0
    n_matched = 0
    type_counts = {}
    with tarfile.open(archive, "r") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".bz2"):
                continue
            n_scanned += 1
            f = tar.extractfile(member)
            if f is None: continue
            data = f.read()
            head = header_definition(data)
            if head is None: continue
            md, runners = head
            mname = md.get("name", "")
            classified = classify_market(mname)
            if classified is None:
                continue
            market_type, target_balls = classified
            n_classified += 1
            type_counts[market_type] = type_counts.get(market_type, 0) + 1

            innings = 1 if mname.lower().startswith("1st") or mname.lower().startswith("first") else 2
            event_name = md.get("eventName", "")
            mt_date = (md.get("marketTime") or "")[:10]
            teams_in_event = [teams_map[t] for t in teams_map if t in event_name]
            if len(set(teams_in_event)) < 2:
                continue
            cs_ids = lookup[
                (lookup["date"].astype(str) == mt_date) &
                (lookup["teams_set"].apply(lambda s: set(teams_in_event[:2]).issubset(s)))
            ]["match_id"].unique()
            if len(cs_ids) == 0:
                continue
            match_id = cs_ids[0]
            try:
                raw = bz2.decompress(data)
            except OSError:
                continue
            history, inplay_start = collect_history(raw)
            if not history:
                continue
            actual = compute_actual_total(match_id, innings, target_balls, balls)
            if actual is None:
                continue
            for sid, name in runners:
                if sid is None:
                    continue
                X = parse_runner_threshold(name)
                if X is None:
                    continue
                rid = int(sid)
                hist = history.get(rid, [])
                if not hist:
                    continue
                row = {
                    "market_type": market_type,
                    "match_id": match_id,
                    "league": league_by_match.get(str(match_id)),
                    "innings": innings,
                    "threshold_X": X,
                    "actual_total": actual,
                    "actual_over_X": int(actual >= X),
                    "inplay_start_ms": inplay_start,
                    "first_ltp": hist[0][1],
                }
                for mins in ANCHORS_MIN:
                    target = (inplay_start - mins * 60_000) if inplay_start is not None else None
                    row[f"ltp_t_minus_{mins}"] = ltp_at_or_before(hist, target) if target is not None else None
                rows.append(row)
            n_matched += 1
    logger.info("scanned %d markets, %d classified, %d matched to Cricsheet",
                n_scanned, n_classified, n_matched)
    logger.info("classified by market_type: %s", type_counts)

    df = pd.DataFrame(rows)
    out = Path("data/processed/all_innings_markets.parquet")
    df.to_parquet(out, index=False)
    logger.info("wrote %d rows -> %s", len(df), out)
    logger.info("matched rows by market_type: %s", df["market_type"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

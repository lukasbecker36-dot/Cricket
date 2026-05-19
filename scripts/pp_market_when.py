"""Three-snapshot PP threshold analysis: opening vs last-pre-inplay vs last-observed.

Per runner, track three prices independently (no all-runners-priced
constraint). Compare calibration across the three snapshots to see whether
the mispricing we observed lives in the early/opening window or persists
through pre-match discovery and into in-play.
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


PP_NAME_PATTERN = re.compile(r"(1st|first|2nd|second).*?6 ?Overs?", re.IGNORECASE)


def parse_runner_threshold(name: str) -> int | None:
    m = re.match(r"^(\d+)\s+Runs?\s+or\s+more", name, re.IGNORECASE)
    return int(m.group(1)) if m else None


def header_definition(bz2_bytes: bytes):
    decompressor = bz2.BZ2Decompressor()
    head = decompressor.decompress(bz2_bytes[:131072])
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


def three_snapshots(raw: bytes):
    """Per runner: opening (first observed), last-pre-inplay, last-observed."""
    opening: dict[int, float] = {}
    last_pre: dict[int, float] = {}
    last: dict[int, float] = {}
    in_play = False
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("op") != "mcm":
            continue
        for change in obj.get("mc", []):
            md = change.get("marketDefinition")
            if md is not None and md.get("inPlay") is True:
                in_play = True
            for rc in change.get("rc", []) or []:
                ltp = rc.get("ltp")
                rid = rc.get("id")
                if ltp is None or rid is None:
                    continue
                rid = int(rid)
                ltp = float(ltp)
                if rid not in opening:
                    opening[rid] = ltp
                if not in_play:
                    last_pre[rid] = ltp
                last[rid] = ltp
    return opening, last_pre, last


def compute_actual_pp_total(match_id: str, innings: int, balls: pd.DataFrame) -> int | None:
    inn = balls[(balls["match_id"] == match_id) & (balls["innings"] == innings)]
    if inn.empty:
        return None
    inn = inn.sort_values(["over", "ball", "is_legal_delivery"], ascending=[True, True, False]).reset_index(drop=True)
    legal_idx = np.where(inn["is_legal_delivery"].values)[0]
    if len(legal_idx) < 36:
        return None
    return int(inn.iloc[: legal_idx[35] + 1]["runs_total"].sum())


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

    rows = []
    n_scanned = 0
    n_matched = 0
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
            if not PP_NAME_PATTERN.search(mname):
                continue
            innings = 1 if "1st" in mname.lower() or "first" in mname.lower() else 2
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
            opening, last_pre, last = three_snapshots(raw)
            if not opening:
                continue
            actual = compute_actual_pp_total(match_id, innings, balls)
            if actual is None:
                continue
            for sid, name in runners:
                if sid is None:
                    continue
                X = parse_runner_threshold(name)
                if X is None:
                    continue
                rid = int(sid)
                rows.append({
                    "match_id": match_id,
                    "innings": innings,
                    "threshold_X": X,
                    "p_open":     opening.get(rid),
                    "p_last_pre": last_pre.get(rid),
                    "p_last":     last.get(rid),
                    "actual_over_X": int(actual >= X),
                    "actual_pp_total": actual,
                })
            n_matched += 1
    logger.info("scanned %d markets, matched %d", n_scanned, n_matched)

    df = pd.DataFrame(rows)
    if df.empty:
        logger.error("no rows produced")
        return 1
    out = Path("data/processed/pp_three_snapshots.parquet")
    df.to_parquet(out, index=False)
    logger.info("wrote %d rows -> %s", len(df), out)

    for col, label in [
        ("p_open",     "OPENING (first observed)"),
        ("p_last_pre", "LAST PRE-INPLAY"),
        ("p_last",     "LAST OBSERVED (often near settlement)"),
    ]:
        sub = df.dropna(subset=[col]).copy()
        sub = sub[sub[col] > 1.0]
        sub["implied"] = 1.0 / sub[col]
        sub["bin"] = pd.cut(sub["implied"], bins=np.arange(0, 1.05, 0.1), include_lowest=True)
        cal = sub.groupby("bin").agg(
            n=("implied", "size"),
            mean_implied=("implied", "mean"),
            actual_rate=("actual_over_X", "mean"),
        ).dropna()
        cal["edge_pp"] = (cal["actual_rate"] - cal["mean_implied"]) * 100
        print(f"\n=== {label} (n={len(sub):,}) ===")
        print(cal.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

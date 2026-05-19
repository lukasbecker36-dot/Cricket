"""Scan for powerplay (6 Over) markets in the all-markets archive, extract
pre-match implied probabilities per runner, match to Cricsheet, and check
calibration vs the actual powerplay total.

This gives a model-free first read: is the market for PP totals systematically
biased at any specific run level?
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

from src.ingestion.cricsheet import iter_match_jsons  # not used; kept for symmetry
from src.ingestion.storage import read_balls
from src.logging_setup import configure_logging

logger = logging.getLogger(__name__)


PP_NAME_PATTERN = re.compile(r"(1st|first|2nd|second).*?6 ?Overs?", re.IGNORECASE)


def parse_runner_threshold(name: str) -> int | None:
    m = re.match(r"^(\d+)\s+Runs?\s+or\s+more", name, re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))


def first_definition_and_runners(bz2_bytes: bytes) -> tuple[dict, list[tuple[int, str]]] | None:
    decompressor = bz2.BZ2Decompressor()
    head = decompressor.decompress(bz2_bytes[:131072])
    first_newline = head.find(b"\n")
    if first_newline == -1:
        return None
    first_line = head[:first_newline].decode("utf-8")
    obj = json.loads(first_line)
    for change in obj.get("mc", []):
        md = change.get("marketDefinition")
        if md is None:
            continue
        runners = [(r.get("id"), r.get("name", "")) for r in md.get("runners", [])]
        return md, runners
    return None


def extract_first_full_snapshot(raw: bytes) -> tuple[int, dict[int, float]] | None:
    """Return (pt_ms, {selection_id: ltp}) at the first tick where every runner has an LTP."""
    seen: dict[int, float] = {}
    pt = None
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("op") != "mcm":
            continue
        pt = obj.get("pt", pt)
        for change in obj.get("mc", []):
            for rc in change.get("rc", []) or []:
                ltp = rc.get("ltp")
                if ltp is not None and rc.get("id") is not None:
                    seen[int(rc["id"])] = float(ltp)
    if not seen or pt is None:
        return None
    return int(pt), seen


def analyse_archive(tar_path: Path, balls_by_match: pd.DataFrame, teams_map: dict[str, str]) -> pd.DataFrame:
    """Walk the archive, find PP markets, match to Cricsheet, record outcomes."""
    rows = []
    n_scanned = 0
    n_matched = 0
    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".bz2"):
                continue
            n_scanned += 1
            f = tar.extractfile(member)
            if f is None:
                continue
            data = f.read()
            head = first_definition_and_runners(data)
            if head is None:
                continue
            md, runners = head
            mname = md.get("name", "")
            if not PP_NAME_PATTERN.search(mname):
                continue
            innings = 1 if "1st" in mname.lower() or "first" in mname.lower() else 2
            event_name = md.get("eventName", "")
            market_time = md.get("marketTime", "")
            mt_date = market_time[:10]
            # match to Cricsheet via event_name + date
            teams_in_event = []
            for t in teams_map:
                if t in event_name:
                    teams_in_event.append(teams_map[t])
            if len(set(teams_in_event)) < 2:
                continue
            cs_match_ids = balls_by_match[
                (balls_by_match["date"].astype(str) == mt_date) &
                (balls_by_match["teams_set"].apply(lambda s: set(teams_in_event[:2]).issubset(s)))
            ]["match_id"].unique()
            if len(cs_match_ids) == 0:
                # +/-1 day
                from datetime import date, timedelta
                try:
                    y, m, d = mt_date.split("-")
                    base = date(int(y), int(m), int(d))
                    for delta in (-1, 1):
                        cand = (base + timedelta(days=delta)).isoformat()
                        cs_match_ids = balls_by_match[
                            (balls_by_match["date"].astype(str) == cand) &
                            (balls_by_match["teams_set"].apply(lambda s: set(teams_in_event[:2]).issubset(s)))
                        ]["match_id"].unique()
                        if len(cs_match_ids) > 0:
                            break
                except ValueError:
                    pass
            if len(cs_match_ids) == 0:
                continue
            match_id = cs_match_ids[0]

            # Decompress fully and extract first full price snapshot
            try:
                raw = bz2.decompress(data)
            except OSError:
                continue
            snap = extract_first_full_snapshot(raw)
            if snap is None:
                continue
            pt_ms, prices = snap

            # Build list of (threshold, ltp) for runners parseable as "X or more"
            thresholds = []
            for sid, name in runners:
                X = parse_runner_threshold(name)
                if X is None or sid is None:
                    continue
                ltp = prices.get(int(sid))
                if ltp is None or ltp <= 1.0:
                    continue
                thresholds.append((X, ltp))
            if not thresholds:
                continue

            # Actual PP total for this match's relevant innings
            inn_balls = balls_by_match[balls_by_match["match_id"] == match_id]
            if inn_balls.empty:
                continue
            actual_pp_total = compute_actual_pp_total(match_id, innings)
            if actual_pp_total is None:
                continue
            for X, ltp in thresholds:
                rows.append({
                    "match_id": match_id,
                    "innings": innings,
                    "threshold_X": X,
                    "ltp": ltp,
                    "implied_p_over": 1.0 / ltp,
                    "actual_pp_total": actual_pp_total,
                    "actual_over_X": int(actual_pp_total >= X),
                })
            n_matched += 1
    logger.info("scanned %d markets, matched %d to Cricsheet", n_scanned, n_matched)
    return pd.DataFrame(rows)


# Cache for compute_actual_pp_total
_balls_cache = {"df": None}


def set_balls_cache(df: pd.DataFrame) -> None:
    _balls_cache["df"] = df


def compute_actual_pp_total(match_id: str, innings: int) -> int | None:
    balls = _balls_cache["df"]
    inn = balls[(balls["match_id"] == match_id) & (balls["innings"] == innings)]
    if inn.empty:
        return None
    inn = inn.sort_values(["over", "ball", "is_legal_delivery"], ascending=[True, True, False]).reset_index(drop=True)
    legal_idx = np.where(inn["is_legal_delivery"].values)[0]
    if len(legal_idx) < 36:
        return None
    cut_36 = legal_idx[35] + 1
    return int(inn.iloc[:cut_36]["runs_total"].sum())


def main() -> int:
    configure_logging()
    archive = Path("data/raw/betfair/betfair_all_markets.dat")

    # Load all leagues' Cricsheet matches+balls combined
    leagues = ["ipl", "bbl", "psl", "cpl", "ntb"]
    all_balls = []
    all_matches = []
    for league in leagues:
        d = Path("data/processed") if league == "ipl" else Path(f"data/processed/{league}")
        b = read_balls(d)
        m = pd.read_parquet(d / "matches.parquet")
        if not b.empty:
            all_balls.append(b)
        if not m.empty:
            all_matches.append(m)
    balls = pd.concat(all_balls, ignore_index=True)
    matches = pd.concat(all_matches, ignore_index=True)
    matches["teams_set"] = matches["teams"].apply(
        lambda ts: frozenset(ts) if hasattr(ts, "__iter__") and not isinstance(ts, str) else frozenset()
    )
    # Build a lookup frame: match_id, date, teams_set
    lookup = matches[["match_id", "date", "teams_set"]].drop_duplicates("match_id")
    # Inject ball-frame date column for matching (use match meta's)
    set_balls_cache(balls)
    lookup["match_id_str"] = lookup["match_id"].astype(str)

    # Identity teams_map: trivially, each team name maps to itself
    all_teams = set()
    for ts in matches["teams"]:
        if hasattr(ts, "__iter__") and not isinstance(ts, str):
            all_teams.update(ts)
    teams_map = {t: t for t in all_teams}
    # Add Betfair -> Cricsheet rename
    teams_map["Royal Challengers Bengaluru"] = "Royal Challengers Bangalore"

    # Run analysis
    df = analyse_archive(archive, lookup, teams_map)
    if df.empty:
        logger.error("no PP markets matched")
        return 1

    out = Path("data/processed/pp_market_calibration.parquet")
    df.to_parquet(out, index=False)
    logger.info("wrote %d (match, threshold) rows -> %s", len(df), out)

    # Calibration bins
    df = df.copy()
    df["implied_bin"] = pd.cut(df["implied_p_over"], bins=np.arange(0, 1.05, 0.1), include_lowest=True)
    cal = df.groupby("implied_bin").agg(
        n=("implied_p_over", "size"),
        mean_implied=("implied_p_over", "mean"),
        actual_rate=("actual_over_X", "mean"),
    ).dropna()
    cal["edge_pp"] = (cal["actual_rate"] - cal["mean_implied"]) * 100
    print("\n=== PP market calibration (pooled across leagues) ===")
    print(cal.to_string())

    # Per-innings split
    for inn in (1, 2):
        sub = df[df["innings"] == inn]
        if sub.empty:
            continue
        sub_cal = sub.copy()
        sub_cal["implied_bin"] = pd.cut(sub_cal["implied_p_over"], bins=np.arange(0, 1.05, 0.1), include_lowest=True)
        c = sub_cal.groupby("implied_bin").agg(
            n=("implied_p_over", "size"),
            mean_implied=("implied_p_over", "mean"),
            actual_rate=("actual_over_X", "mean"),
        ).dropna()
        c["edge_pp"] = (c["actual_rate"] - c["mean_implied"]) * 100
        print(f"\n=== Innings {inn} only ===")
        print(c.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

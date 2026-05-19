"""Per-runner LTP at specific time anchors before in-play start.

Previous analyses used 'last observed' = the very last LTP per runner in
the whole stream, which is contaminated by in-play prices. This script
parses the inPlay=true transition timestamp from the stream's
marketDefinition messages, then extracts each runner's LTP at:
  - 60 minutes before in-play start
  - 30 minutes before in-play start
  - 10 minutes before in-play start
  - 1 minute before in-play start
  - First observed price (opening reference)

For each snapshot we recompute calibration vs actual outcomes. This tells
us at what point in the market's life-cycle the mispricing is largest --
and whether it survives all the way to the moment a real trader would
place their last pre-match bet.
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
ANCHORS_MIN = [60, 30, 10, 1]  # minutes before in-play start


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


def collect_history(raw: bytes):
    """Per-runner LTP history sorted by pt_ms, plus first inPlay=true timestamp."""
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


def ltp_at_or_before(hist: list[tuple[int, float]], target_ms: int) -> float | None:
    """Latest LTP <= target_ms. Returns None if no prior tick."""
    best = None
    for pt, ltp in hist:
        if pt <= target_ms:
            best = ltp
        else:
            break
    return best


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
    league_by_match: dict[str, str] = {}
    for league in leagues:
        d = Path("data/processed") if league == "ipl" else Path(f"data/processed/{league}")
        m = pd.read_parquet(d / "matches.parquet")
        for mid in m["match_id"].astype(str).unique():
            league_by_match[mid] = league

    all_teams = set()
    for ts in matches["teams"]:
        if hasattr(ts, "__iter__") and not isinstance(ts, str):
            all_teams.update(ts)
    teams_map = {t: t for t in all_teams}
    teams_map["Royal Challengers Bengaluru"] = "Royal Challengers Bangalore"

    rows = []
    n_scanned = 0
    n_matched = 0
    n_inplay_known = 0
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
            history, inplay_start = collect_history(raw)
            if not history:
                continue
            if inplay_start is not None:
                n_inplay_known += 1
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
                hist = history.get(rid, [])
                if not hist:
                    continue
                row = {
                    "match_id": match_id,
                    "innings": innings,
                    "threshold_X": X,
                    "actual_over_X": int(actual >= X),
                    "actual_pp_total": actual,
                    "inplay_start_ms": inplay_start,
                    "league": league_by_match.get(str(match_id)),
                    "first_ltp": hist[0][1],
                }
                for mins in ANCHORS_MIN:
                    target = (inplay_start - mins * 60_000) if inplay_start is not None else None
                    row[f"ltp_t_minus_{mins}"] = ltp_at_or_before(hist, target) if target is not None else None
                rows.append(row)
            n_matched += 1
    logger.info("scanned %d, matched %d markets, %d had inplay_start", n_scanned, n_matched, n_inplay_known)

    df = pd.DataFrame(rows)
    if df.empty:
        return 1
    out = Path("data/processed/pp_time_anchors.parquet")
    df.to_parquet(out, index=False)
    logger.info("wrote %d rows -> %s", len(df), out)

    # Per-anchor calibration
    for col, label in [
        ("first_ltp",        "OPENING (first observed)"),
        ("ltp_t_minus_60",   "T-60 min"),
        ("ltp_t_minus_30",   "T-30 min"),
        ("ltp_t_minus_10",   "T-10 min"),
        ("ltp_t_minus_1",    "T-1 min (last pre-inplay)"),
    ]:
        sub = df.dropna(subset=[col]).copy()
        sub = sub[sub[col] > 1.0]
        if sub.empty:
            continue
        sub["implied"] = 1.0 / sub[col]
        sub["bin"] = pd.cut(sub["implied"], bins=np.arange(0, 1.05, 0.1), include_lowest=True)
        cal = sub.groupby("bin").agg(
            n=("implied", "size"),
            mean_implied=("implied", "mean"),
            actual_rate=("actual_over_X", "mean"),
        ).dropna()
        cal["edge_pp"] = (cal["actual_rate"] - cal["mean_implied"]) * 100
        print(f"\n=== {label}  (n_rows={len(sub):,}) ===")
        print(cal.round(4).to_string())

    # Focused: mid-bin (0.4-0.5) edge_pp at each anchor
    print("\n=== Mid-bin (0.40-0.50 implied) by anchor, by league ===")
    rows_summary = []
    for col, label in [("first_ltp", "OPEN"), ("ltp_t_minus_60", "T-60"),
                       ("ltp_t_minus_30", "T-30"), ("ltp_t_minus_10", "T-10"),
                       ("ltp_t_minus_1", "T-1")]:
        for league in leagues + ["ALL"]:
            sub = df if league == "ALL" else df[df["league"] == league]
            sub = sub.dropna(subset=[col])
            sub = sub[sub[col] > 1.0]
            sub = sub[(1/sub[col] >= 0.4) & (1/sub[col] < 0.5)]
            if len(sub) < 5:
                continue
            implied_mean = (1.0 / sub[col]).mean()
            actual = sub["actual_over_X"].mean()
            rows_summary.append({
                "anchor": label,
                "league": league.upper(),
                "n": int(len(sub)),
                "implied": round(float(implied_mean), 3),
                "actual": round(float(actual), 3),
                "edge_pp": round(float((actual - implied_mean) * 100), 2),
            })
    print(pd.DataFrame(rows_summary).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

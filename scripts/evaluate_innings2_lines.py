"""Test innings-2 phase Line markets using the existing innings-1-trained
phase models. Pure transfer test: does the same model work in innings 2,
or do we need innings-2-specific training?

Innings 2 phase totals are sampled the same way as innings 1 (cumulative
runs at end of overs 6/10/15). For markets where the chase ended before
the phase completed, we exclude (incomplete phase).
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
from src.live.signals import load_models
from src.logging_setup import configure_logging

logger = logging.getLogger(__name__)

LINE_NAME_PATTERNS = [
    ("phase_6",   36,  re.compile(r"^(1st|first|2nd|second) +Innings +6 +Overs? +Line$",  re.IGNORECASE)),
    ("phase_10",  60,  re.compile(r"^(1st|first|2nd|second) +Innings +10 +Overs? +Line$", re.IGNORECASE)),
    ("phase_15",  90,  re.compile(r"^(1st|first|2nd|second) +Innings +15 +Overs? +Line$", re.IGNORECASE)),
    ("full_innings", 120, re.compile(r"^(1st|first|2nd|second) +Innings +Runs +Line$", re.IGNORECASE)),
]


def header_definition(bz2_bytes: bytes):
    head = bz2.BZ2Decompressor().decompress(bz2_bytes[:131072])
    nl = head.find(b"\n")
    if nl == -1: return None
    obj = json.loads(head[:nl].decode("utf-8"))
    for change in obj.get("mc", []):
        md = change.get("marketDefinition")
        if md is not None: return md
    return None


def collect_line_history(raw: bytes):
    history: list[tuple[int, float]] = []
    inplay_start: int | None = None
    last_pt: int | None = None
    for line in raw.splitlines():
        if not line.strip(): continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("op") != "mcm": continue
        pt = obj.get("pt")
        if pt is not None:
            last_pt = int(pt)
        for change in obj.get("mc", []):
            md = change.get("marketDefinition")
            if md is not None and inplay_start is None and md.get("inPlay") is True:
                inplay_start = last_pt
            for rc in change.get("rc", []) or []:
                ltp = rc.get("ltp")
                if ltp is not None and last_pt is not None:
                    history.append((last_pt, float(ltp)))
    history.sort(key=lambda x: x[0])
    return history, inplay_start


def line_at_or_before(hist, target_ms):
    best = None
    for pt, val in hist:
        if pt <= target_ms:
            best = val
        else:
            break
    return best


def phase_total(balls, mid, innings, target_balls):
    inn = balls[(balls["match_id"] == mid) & (balls["innings"] == innings)]
    if inn.empty: return None
    inn = inn.sort_values(["over", "ball", "is_legal_delivery"], ascending=[True, True, False]).reset_index(drop=True)
    legal_idx = np.where(inn["is_legal_delivery"].values)[0]
    if target_balls == 120:
        return int(inn["runs_total"].sum()) if len(legal_idx) >= 30 else None
    if len(legal_idx) < target_balls: return None
    cut = legal_idx[target_balls - 1] + 1
    return int(inn.iloc[:cut]["runs_total"].sum())


def main() -> int:
    configure_logging()
    LEAGUES = ["ipl", "bbl", "psl", "cpl", "ntb"]

    all_balls, matches_dfs = [], []
    league_by_match = {}
    for league in LEAGUES:
        d = Path("data/processed") if league == "ipl" else Path(f"data/processed/{league}")
        b = read_balls(d)
        if not b.empty: all_balls.append(b)
        m = pd.read_parquet(d / "matches.parquet")
        matches_dfs.append(m)
        for mid in m["match_id"].astype(str).unique():
            league_by_match[mid] = league
    balls = pd.concat(all_balls, ignore_index=True)
    matches = pd.concat(matches_dfs, ignore_index=True)
    matches["teams_set"] = matches["teams"].apply(
        lambda ts: frozenset(ts) if hasattr(ts, "__iter__") and not isinstance(ts, str) else frozenset()
    )
    lookup = matches[["match_id", "date", "teams_set"]].drop_duplicates("match_id")
    season_map = balls[["match_id", "season"]].drop_duplicates("match_id").set_index("match_id")["season"].astype(int).to_dict()
    venue_map = balls[["match_id", "venue"]].drop_duplicates("match_id").set_index("match_id")["venue"].to_dict()
    bat_team_map = {(mid, int(inn)): g["batting_team"].iloc[0] for (mid, inn), g in balls.groupby(["match_id", "innings"])}
    all_teams = set()
    for ts in matches["teams"]:
        if hasattr(ts, "__iter__") and not isinstance(ts, str):
            all_teams.update(ts)
    teams_map = {t: t for t in all_teams}
    teams_map["Royal Challengers Bengaluru"] = "Royal Challengers Bangalore"

    registry = load_models(Path("models"))
    archives = [Path("data/raw/betfair/betfair_all_markets.dat"),
                Path("data/raw/betfair/betfair_2026.dat")]

    rows = []
    for arc in archives:
        if not arc.exists(): continue
        with tarfile.open(arc, "r") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".bz2"): continue
                f = tar.extractfile(member)
                if f is None: continue
                data = f.read()
                md = header_definition(data)
                if md is None: continue
                mname = md.get("name", "")
                phase_info = next(
                    ((lbl, tb) for lbl, tb, p in LINE_NAME_PATTERNS if p.match(mname.strip())), None
                )
                if phase_info is None: continue
                phase_label, target_balls = phase_info
                innings = 1 if mname.lower().startswith("1st") or mname.lower().startswith("first") else 2
                event_name = md.get("eventName", "")
                mt_date = (md.get("marketTime") or "")[:10]
                teams_in_event = [teams_map[t] for t in teams_map if t in event_name]
                if len(set(teams_in_event)) < 2: continue
                cs_ids = lookup[
                    (lookup["date"].astype(str) == mt_date) &
                    (lookup["teams_set"].apply(lambda s: set(teams_in_event[:2]).issubset(s)))
                ]["match_id"].unique()
                if len(cs_ids) == 0: continue
                match_id = cs_ids[0]
                try:
                    raw = bz2.decompress(data)
                except OSError:
                    continue
                hist, inplay_start = collect_line_history(raw)
                if not hist or inplay_start is None: continue
                actual = phase_total(balls, match_id, innings, target_balls)
                if actual is None: continue
                line_t1 = line_at_or_before(hist, inplay_start - 60_000)
                if line_t1 is None: continue
                rows.append({
                    "phase": phase_label, "target_balls": target_balls,
                    "match_id": match_id, "league": league_by_match.get(str(match_id)),
                    "innings": innings, "line_t_minus_1": line_t1, "actual_total": actual,
                    "archive": arc.name,
                })

    df = pd.DataFrame(rows)
    if df.empty: return 1
    logger.info("rows: %d", len(df))

    df["season"] = df["match_id"].astype(str).map(season_map)
    df["venue"] = df["match_id"].astype(str).map(venue_map)
    df["batting_team"] = df.apply(lambda r: bat_team_map.get((r["match_id"], int(r["innings"])), ""), axis=1)
    df["bowling_team"] = df.apply(
        lambda r: next(
            (t for (m, inn), t in bat_team_map.items() if m == r["match_id"] and inn != int(r["innings"])), ""), axis=1)

    LINE_ODDS = 1.92
    IMPLIED = 1.0 / LINE_ODDS
    EDGE = 0.05

    def score(row):
        key = row["phase"].replace("phase_", "") if row["phase"] != "full_innings" else "full"
        m = registry.get(key)
        if m is None: return np.nan
        return m.predict_p(
            threshold_X=int(round(row["line_t_minus_1"])), implied_open=IMPLIED,
            batting_team=row["batting_team"], bowling_team=row["bowling_team"],
            venue=row["venue"], season=int(row["season"]), innings=int(row["innings"]),
            league=row["league"],
        )

    df["model_p"] = df.apply(score, axis=1)
    df = df.dropna(subset=["model_p"])
    df["actual_over_line"] = (df["actual_total"] > df["line_t_minus_1"]).astype(int)
    df["actual_under_line"] = (df["actual_total"] < df["line_t_minus_1"]).astype(int)

    def backtest(sub):
        if sub.empty: return None
        signal = np.where(sub["model_p"] >= 0.5 + EDGE, "back_over",
                          np.where(sub["model_p"] <= 0.5 - EDGE, "back_under", "skip"))
        sub = sub.assign(signal=signal)
        active = sub[sub["signal"] != "skip"]
        if active.empty: return None
        won = np.where(active["signal"] == "back_over",
                       active["actual_over_line"] == 1,
                       active["actual_under_line"] == 1)
        pnl = np.where(won, 100 * (LINE_ODDS - 1) * 0.95, -100.0)
        return {"n": int(len(active)), "pnl": float(pnl.sum()),
                "roi": float(pnl.sum() / (len(active) * 100)),
                "win_rate": float(np.mean(won))}

    print(f"\n===== INNINGS 1 vs INNINGS 2 phase Line backtests =====")
    print(f"(model trained on innings-1 data only; this tests transfer to innings 2)")
    for phase in ["phase_6", "phase_10", "phase_15", "full_innings"]:
        print(f"\n=== {phase} ===")
        for inn in (1, 2):
            sub = df[(df["phase"] == phase) & (df["innings"] == inn)]
            r = backtest(sub)
            if r and r["n"] > 0:
                print(f"  innings {inn}: n={r['n']:4d}  pnl=£{r['pnl']:+.0f}  ROI={r['roi']:+.2%}  win={r['win_rate']:.1%}")
            else:
                print(f"  innings {inn}: no trades")

    print(f"\n===== STRICT OOS (2025+) by innings =====")
    df_oos = df[df["season"] >= 2025]
    for phase in ["phase_6", "phase_10", "phase_15", "full_innings"]:
        for inn in (1, 2):
            sub = df_oos[(df_oos["phase"] == phase) & (df_oos["innings"] == inn)]
            r = backtest(sub)
            if r and r["n"] >= 10:
                print(f"  {phase} inn{inn}: n={r['n']:3d}  ROI={r['roi']:+.2%}  win={r['win_rate']:.1%}")

    # Per league for innings 2 across phases
    print(f"\n===== INNINGS 2 per league (all phases combined) =====")
    inn2 = df[df["innings"] == 2]
    for L, g in inn2.groupby("league"):
        r = backtest(g)
        if r and r["n"] > 0:
            print(f"  {L.upper():5s} n={r['n']:3d}  ROI={r['roi']:+.2%}  win={r['win_rate']:.1%}")

    out = Path("data/processed/eval_innings2_lines.parquet")
    df.to_parquet(out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

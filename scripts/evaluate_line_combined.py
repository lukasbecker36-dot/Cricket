"""Bigger sample on Innings Runs Line markets: both archives + 1st and 2nd
innings. Same backtest framework as evaluate_2026_line.py.
"""
from __future__ import annotations

import bz2
import json
import logging
import re
import tarfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.ingestion.storage import read_balls
from src.logging_setup import configure_logging

logger = logging.getLogger(__name__)

NAME_PATTERN = re.compile(r"^(1st|first|2nd|second) +Innings +Runs +Line$", re.IGNORECASE)


def header_definition(bz2_bytes: bytes):
    head = bz2.BZ2Decompressor().decompress(bz2_bytes[:131072])
    nl = head.find(b"\n")
    if nl == -1:
        return None
    obj = json.loads(head[:nl].decode("utf-8"))
    for change in obj.get("mc", []):
        md = change.get("marketDefinition")
        if md is not None:
            return md
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


def innings_total(match_id: str, innings: int, balls: pd.DataFrame) -> int | None:
    inn = balls[(balls["match_id"] == match_id) & (balls["innings"] == innings)]
    if inn.empty:
        return None
    return int(inn["runs_total"].sum())


def extract_from_archive(
    archive_path: Path,
    lookup: pd.DataFrame,
    league_by_match: dict,
    teams_map: dict,
    balls: pd.DataFrame,
) -> list[dict]:
    rows = []
    n_scanned, n_line, n_matched = 0, 0, 0
    with tarfile.open(archive_path, "r") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".bz2"): continue
            n_scanned += 1
            f = tar.extractfile(member)
            if f is None: continue
            data = f.read()
            md = header_definition(data)
            if md is None: continue
            mname = md.get("name", "")
            if not NAME_PATTERN.match(mname.strip()): continue
            n_line += 1
            innings = 1 if mname.lower().startswith("1st") or mname.lower().startswith("first") else 2
            event_name = md.get("eventName", "")
            mt_date = (md.get("marketTime") or "")[:10]
            teams_in_event = [teams_map[t] for t in teams_map if t in event_name]
            if len(set(teams_in_event)) < 2: continue
            cs_ids = lookup[
                (lookup["date"].astype(str) == mt_date) &
                (lookup["teams_set"].apply(lambda s: set(teams_in_event[:2]).issubset(s)))
            ]["match_id"].unique()
            if len(cs_ids) == 0:
                from datetime import date, timedelta
                try:
                    y, mo, d = mt_date.split("-")
                    base = date(int(y), int(mo), int(d))
                    for delta in (-1, 1):
                        cand = (base + timedelta(days=delta)).isoformat()
                        cs_ids = lookup[
                            (lookup["date"].astype(str) == cand) &
                            (lookup["teams_set"].apply(lambda s: set(teams_in_event[:2]).issubset(s)))
                        ]["match_id"].unique()
                        if len(cs_ids) > 0: break
                except ValueError:
                    pass
            if len(cs_ids) == 0: continue
            match_id = cs_ids[0]
            n_matched += 1
            try:
                raw = bz2.decompress(data)
            except OSError:
                continue
            hist, inplay_start = collect_line_history(raw)
            if not hist or inplay_start is None: continue
            actual = innings_total(match_id, innings, balls)
            if actual is None: continue
            line_t1 = line_at_or_before(hist, inplay_start - 60_000)
            if line_t1 is None: continue
            rows.append({
                "match_id": match_id,
                "league": league_by_match.get(str(match_id)),
                "innings": innings,
                "actual_total": actual,
                "line_t_minus_1": line_t1,
                "archive": str(archive_path.name),
            })
    logger.info("%s: scanned=%d line=%d matched=%d -> %d rows with T-1",
                archive_path.name, n_scanned, n_line, n_matched, len(rows))
    return rows


def main() -> int:
    configure_logging()

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
    league_by_match = {}
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
    season_map = balls[["match_id", "season"]].drop_duplicates("match_id").set_index("match_id")["season"].astype(int).to_dict()
    venue_map = balls[["match_id", "venue"]].drop_duplicates("match_id").set_index("match_id")["venue"].to_dict()
    bat_team_map = {(mid, int(inn)): g["batting_team"].iloc[0]
                    for (mid, inn), g in balls.groupby(["match_id", "innings"])}

    archives = [
        Path("data/raw/betfair/betfair_all_markets.dat"),
        Path("data/raw/betfair/betfair_2026.dat"),
    ]
    all_rows = []
    for arc in archives:
        if not arc.exists():
            logger.warning("missing archive: %s", arc)
            continue
        all_rows.extend(extract_from_archive(arc, lookup, league_by_match, teams_map, balls))

    df = pd.DataFrame(all_rows)
    logger.info("total rows: %d", len(df))
    if df.empty:
        return 1
    df["season"] = df["match_id"].astype(str).map(season_map)
    df["venue"] = df["match_id"].astype(str).map(venue_map)
    df["batting_team"] = df.apply(lambda r: bat_team_map.get((r["match_id"], int(r["innings"])), ""), axis=1)
    df["bowling_team"] = df.apply(
        lambda r: next(
            (t for (m, inn), t in bat_team_map.items() if m == r["match_id"] and inn != int(r["innings"])), ""), axis=1)

    # Load saved full_innings model + lookups
    model_dir = Path("models")
    with open(model_dir / "full_innings_meta.json") as f:
        meta = json.load(f)
    features = meta["features"]
    default_par = meta["default_par"]
    leagues_meta = meta["leagues"]
    booster = lgb.Booster(model_file=str(model_dir / "full_innings_gbm.lgb"))
    with open(model_dir / "full_innings_bat_pp.json") as f: bat_pp = json.load(f)
    with open(model_dir / "full_innings_bowl_pp.json") as f: bowl_pp = json.load(f)
    with open(model_dir / "full_innings_venue_par.json") as f: venue_par = json.load(f)

    def lookup_pp(table, team, season):
        return float(table.get(f"{team}|{int(season)}", default_par))

    df["bat_prior"] = df.apply(lambda r: lookup_pp(bat_pp, r["batting_team"], r["season"]), axis=1)
    df["bowl_prior"] = df.apply(lambda r: lookup_pp(bowl_pp, r["bowling_team"], r["season"]), axis=1)
    df["venue_par"] = df.apply(lambda r: lookup_pp(venue_par, r["venue"], r["season"]), axis=1)

    LINE_ODDS = 1.92
    IMPLIED_AT_LINE_ODDS = 1.0 / LINE_ODDS
    df["threshold_X"] = df["line_t_minus_1"].round().astype(int)
    df["implied_open"] = IMPLIED_AT_LINE_ODDS
    df["x_minus_par"] = df["threshold_X"] - df["venue_par"]
    df["x_minus_bat"] = df["threshold_X"] - df["bat_prior"]
    df["x_minus_bowl"] = df["threshold_X"] - df["bowl_prior"]
    for L in leagues_meta:
        df[f"is_{L}"] = (df["league"] == L).astype(int)
    df = df.dropna(subset=features + ["season"])
    df["model_p"] = booster.predict(df[features].to_numpy(dtype=np.float32))
    df["actual_over_line"] = (df["actual_total"] > df["line_t_minus_1"]).astype(int)
    df["actual_under_line"] = (df["actual_total"] < df["line_t_minus_1"]).astype(int)

    out = Path("data/processed/eval_line_combined.parquet")
    df.to_parquet(out, index=False)
    logger.info("wrote %s", out)

    def backtest(sub, signal_col="signal"):
        if sub.empty: return None
        won = np.where(
            sub[signal_col] == "back_over", sub["actual_over_line"] == 1,
            np.where(sub[signal_col] == "back_under", sub["actual_under_line"] == 1, False))
        pnl = np.where(won, 100 * (LINE_ODDS - 1) * 0.95, -100.0)
        return {"n": int(len(sub)), "pnl": float(pnl.sum()),
                "roi": float(pnl.sum() / (len(sub) * 100)),
                "win_rate": float(np.mean(won))}

    print(f"\n========== INNINGS RUNS LINE: combined 2021-2026 sample ==========")
    print(f"Total rows with T-1 line + outcome: {len(df)}")
    print(f"\nDistribution by archive:")
    print(df["archive"].value_counts().to_string())
    print(f"\nDistribution by league/innings:")
    print(df.groupby(["league", "innings"]).size().to_string())

    print(f"\n=== Edge threshold sweep (LINE_ODDS={LINE_ODDS}, commission 5%%) ===")
    for edge in [0.02, 0.03, 0.05, 0.07, 0.10, 0.15]:
        sub = df[((df["model_p"] >= 0.50 + edge) | (df["model_p"] <= 0.50 - edge))].copy()
        sub["signal"] = np.where(sub["model_p"] >= 0.50 + edge, "back_over", "back_under")
        r = backtest(sub)
        if r and r["n"] > 0:
            print(f"  edge>={edge:.2f}: n={r['n']:4d}  pnl={r['pnl']:+8.0f}  ROI={r['roi']:+7.2%}  win={r['win_rate']:.1%}")

    # Pick a working threshold and report detail
    edge_th = 0.05
    sub = df[((df["model_p"] >= 0.50 + edge_th) | (df["model_p"] <= 0.50 - edge_th))].copy()
    sub["signal"] = np.where(sub["model_p"] >= 0.50 + edge_th, "back_over", "back_under")
    print(f"\n=== Detail at edge>={edge_th:.2f} ===")
    r = backtest(sub)
    print(f"Overall: n={r['n']}  pnl=£{r['pnl']:+.0f}  ROI={r['roi']:+.2%}  win={r['win_rate']:.1%}")
    print(f"\nBy direction:")
    for d in ["back_over", "back_under"]:
        s = sub[sub["signal"] == d]
        rr = backtest(s)
        if rr and rr["n"] > 0:
            print(f"  {d:10s}: n={rr['n']:3d}  ROI={rr['roi']:+7.2%}  win={rr['win_rate']:.1%}")
    print(f"\nBy league:")
    for L, g in sub.groupby("league"):
        rr = backtest(g)
        if rr and rr["n"] > 0:
            print(f"  {L.upper():5s}: n={rr['n']:3d}  ROI={rr['roi']:+7.2%}  win={rr['win_rate']:.1%}")
    print(f"\nBy innings:")
    for inn, g in sub.groupby("innings"):
        rr = backtest(g)
        if rr and rr["n"] > 0:
            print(f"  innings {inn}: n={rr['n']:3d}  ROI={rr['roi']:+7.2%}  win={rr['win_rate']:.1%}")

    # Honest test: train-test split by year so we don't double-dip on training data
    # The full_innings model was trained on data up through early 2025. Hold out
    # only the 2025+ rows as 'fresh' test.
    print(f"\n=== Out-of-sample only (matches in 2025 or later) ===")
    df_oos = df[df["season"] >= 2025].copy()
    sub_oos = df_oos[((df_oos["model_p"] >= 0.50 + edge_th) | (df_oos["model_p"] <= 0.50 - edge_th))].copy()
    sub_oos["signal"] = np.where(sub_oos["model_p"] >= 0.50 + edge_th, "back_over", "back_under")
    r = backtest(sub_oos)
    if r and r["n"] > 0:
        print(f"  n={r['n']}  pnl=£{r['pnl']:+.0f}  ROI={r['roi']:+.2%}  win={r['win_rate']:.1%}")
        for L, g in sub_oos.groupby("league"):
            rr = backtest(g)
            if rr and rr["n"] > 0:
                print(f"    {L.upper():5s}: n={rr['n']:3d}  ROI={rr['roi']:+7.2%}  win={rr['win_rate']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

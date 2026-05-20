"""Evaluate the saved full_innings selectivity model on the Jan-May 2026 Betfair
archive — pure out-of-sample test since the model was trained on data ending
in the previous archive (2021-2025).
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

PP_NAME = re.compile(r"^(1st|first|2nd|second) +Innings +Runs$", re.IGNORECASE)


def parse_runner_threshold(name: str) -> int | None:
    m = re.match(r"^(\d+)\s+Runs?\s+or\s+more", name, re.IGNORECASE)
    return int(m.group(1)) if m else None


def header_definition(bz2_bytes: bytes):
    head = bz2.BZ2Decompressor().decompress(bz2_bytes[:131072])
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


def compute_actual_innings1(match_id: str, balls: pd.DataFrame) -> int | None:
    inn = balls[(balls["match_id"] == match_id) & (balls["innings"] == 1)]
    if inn.empty:
        return None
    return int(inn["runs_total"].sum())


def main() -> int:
    configure_logging()
    archive = Path("data/raw/betfair/betfair_2026.dat")

    leagues = ["ipl", "bbl", "psl", "cpl", "ntb"]
    all_balls, all_matches = [], []
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
    season_map = balls[["match_id", "season"]].drop_duplicates("match_id").set_index("match_id")["season"].astype(int).to_dict()
    venue_map = balls[["match_id", "venue"]].drop_duplicates("match_id").set_index("match_id")["venue"].to_dict()
    bat_team_map = {(mid, int(inn)): g["batting_team"].iloc[0]
                    for (mid, inn), g in balls.groupby(["match_id", "innings"])}

    # --- Scan archive for multi-runner 1st Innings Runs markets -----
    rows = []
    n_scanned = 0
    n_classified = 0
    n_matched = 0
    n_outcome = 0
    with tarfile.open(archive, "r") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".bz2"):
                continue
            n_scanned += 1
            f = tar.extractfile(member)
            if f is None:
                continue
            data = f.read()
            head = header_definition(data)
            if head is None:
                continue
            md, runners = head
            mname = md.get("name", "")
            if not PP_NAME.match(mname.strip()):
                continue
            if len(runners) < 5:
                continue  # ladder requires multiple runners
            n_classified += 1
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
                # +/- 1 day window for cross-midnight matches
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
                        if len(cs_ids) > 0:
                            break
                except ValueError:
                    pass
            if len(cs_ids) == 0:
                continue
            match_id = cs_ids[0]
            n_matched += 1
            try:
                raw = bz2.decompress(data)
            except OSError:
                continue
            history, inplay_start = collect_history(raw)
            if not history:
                continue
            actual = compute_actual_innings1(match_id, balls)
            if actual is None:
                continue
            n_outcome += 1
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
                first_ltp = hist[0][1]
                ltp_t1 = ltp_at_or_before(hist, inplay_start - 60_000) if inplay_start else None
                rows.append({
                    "match_id": match_id,
                    "league": league_by_match.get(str(match_id)),
                    "innings": innings,
                    "threshold_X": X,
                    "actual_total": actual,
                    "actual_over_X": int(actual >= X),
                    "first_ltp": first_ltp,
                    "ltp_t_minus_1": ltp_t1,
                })
    logger.info("scanned %d, multi-runner 1st innings: %d, matched to Cricsheet: %d, with outcome: %d",
                n_scanned, n_classified, n_matched, n_outcome)

    df = pd.DataFrame(rows)
    if df.empty:
        logger.error("no rows extracted")
        return 1
    df["implied_open"] = 1.0 / df["first_ltp"]
    df["implied_t1"] = np.where(
        df["ltp_t_minus_1"].notna() & (df["ltp_t_minus_1"] > 1.0),
        1.0 / df["ltp_t_minus_1"], np.nan,
    )
    df["season"] = df["match_id"].astype(str).map(season_map)
    df["venue"] = df["match_id"].astype(str).map(venue_map)
    df["batting_team"] = df.apply(lambda r: bat_team_map.get((r["match_id"], int(r["innings"])), ""), axis=1)
    df["bowling_team"] = df.apply(
        lambda r: next(
            (t for (m, inn), t in bat_team_map.items()
             if m == r["match_id"] and inn != int(r["innings"])), ""), axis=1)

    # --- Load saved model + lookups -----
    model_dir = Path("models")
    with open(model_dir / "full_innings_meta.json") as f:
        meta = json.load(f)
    features = meta["features"]
    threshold = meta["edge_threshold"]
    implied_min = meta["implied_min"]
    implied_max = meta["implied_max"]
    default_par = meta["default_par"]
    leagues_meta = meta["leagues"]
    booster = lgb.Booster(model_file=str(model_dir / "full_innings_gbm.lgb"))
    with open(model_dir / "full_innings_bat_pp.json") as f:
        bat_pp = json.load(f)
    with open(model_dir / "full_innings_bowl_pp.json") as f:
        bowl_pp = json.load(f)
    with open(model_dir / "full_innings_venue_par.json") as f:
        venue_par = json.load(f)

    def lookup(table, team, season):
        return float(table.get(f"{team}|{int(season)}", default_par))

    df["bat_prior"] = df.apply(lambda r: lookup(bat_pp, r["batting_team"], r["season"]), axis=1)
    df["bowl_prior"] = df.apply(lambda r: lookup(bowl_pp, r["bowling_team"], r["season"]), axis=1)
    df["venue_par"] = df.apply(lambda r: lookup(venue_par, r["venue"], r["season"]), axis=1)
    df["x_minus_par"] = df["threshold_X"] - df["venue_par"]
    df["x_minus_bat"] = df["threshold_X"] - df["bat_prior"]
    df["x_minus_bowl"] = df["threshold_X"] - df["bowl_prior"]
    for L in leagues_meta:
        df[f"is_{L}"] = (df["league"] == L).astype(int)

    df = df.dropna(subset=["season", "actual_over_X"] + features)
    df["model_p"] = booster.predict(df[features].to_numpy(dtype=np.float32))
    df["edge_open"] = df["model_p"] - df["implied_open"]
    df["edge_t1"] = df["model_p"] - df["implied_t1"]
    logger.info("scored %d (match, runner) rows", len(df))

    out = Path("data/processed/eval_2026.parquet")
    df.to_parquet(out, index=False)
    logger.info("wrote %s", out)

    # --- Backtest selection at the validated threshold -----
    def backtest(sub: pd.DataFrame, price_col: str, stake: float = 100.0, commission: float = 0.05):
        if sub.empty:
            return None
        price = 1.0 / sub[price_col]
        won = sub["actual_over_X"] == 0
        pnl = np.where(won, stake * (1 - commission), -stake * (price - 1.0))
        return {
            "n": len(sub),
            "pnl": float(pnl.sum()),
            "roi": float(pnl.sum() / (len(sub) * stake)),
            "win_rate": float(won.mean()),
        }

    print(f"\n===== 2026 OUT-OF-SAMPLE (Jan-May 2026 Betfair archive) =====")
    print(f"Markets matched to Cricsheet: {n_matched} / {n_classified} multi-runner candidates "
          f"({n_scanned} total markets scanned)")
    print(f"Total (match, runner) rows scored: {len(df)}")
    print(f"Threshold (locked from training): edge < {threshold}")
    print(f"\nLeagues represented in matched markets:")
    print(df.groupby("league")["match_id"].nunique().to_string())

    print(f"\n--- OPENING price execution ---")
    sel = df[(df["edge_open"] < threshold) & (df["implied_open"] >= implied_min) & (df["implied_open"] <= implied_max)]
    res = backtest(sel, "implied_open")
    if res:
        print(f"  n={res['n']}  pnl=£{res['pnl']:+.0f}  ROI={res['roi']:+.2%}  win_rate={res['win_rate']:.1%}")
    print(f"  Per league:")
    for L, group in sel.groupby("league"):
        r = backtest(group, "implied_open")
        if r:
            print(f"    {L.upper():5s}  n={r['n']:3d}  pnl=£{r['pnl']:+7.0f}  ROI={r['roi']:+.2%}  win_rate={r['win_rate']:.1%}")

    print(f"\n--- T-1 min price execution (real pre-match price) ---")
    sel_t1 = df.dropna(subset=["implied_t1"])
    sel_t1 = sel_t1[(sel_t1["edge_t1"] < threshold) & (sel_t1["implied_t1"] >= implied_min) & (sel_t1["implied_t1"] <= implied_max)]
    res_t1 = backtest(sel_t1, "implied_t1")
    if res_t1:
        print(f"  n={res_t1['n']}  pnl=£{res_t1['pnl']:+.0f}  ROI={res_t1['roi']:+.2%}  win_rate={res_t1['win_rate']:.1%}")
    print(f"  Per league:")
    for L, group in sel_t1.groupby("league"):
        r = backtest(group, "implied_t1")
        if r:
            print(f"    {L.upper():5s}  n={r['n']:3d}  pnl=£{r['pnl']:+7.0f}  ROI={r['roi']:+.2%}  win_rate={r['win_rate']:.1%}")

    # Sanity: model log-loss vs market log-loss
    from sklearn.metrics import log_loss
    p_market = np.clip(df["implied_open"], 1e-3, 1 - 1e-3)
    p_model = np.clip(df["model_p"], 1e-3, 1 - 1e-3)
    y = df["actual_over_X"]
    print(f"\nLog-loss (full universe, OPENING prices):")
    print(f"  market: {log_loss(y, p_market):.4f}")
    print(f"  model:  {log_loss(y, p_model):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

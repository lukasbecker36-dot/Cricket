"""Test 'Innings Runs Line' markets — single-threshold over/under format with real
back/lay liquidity. Strategy: our saved full_innings model predicts P(total >=
line) at the line value the market is showing. If model edge >= 5pp vs the
50/50 implied by ~1.90 odds, take a side.

For each market:
  1. Find the line value at our anchor (T-1 min before in-play)
  2. Compute model P(actual >= line)
  3. If P > 0.55: back OVER (model edge +5pp+)
  4. If P < 0.45: back UNDER
  5. Else: no trade
  6. Settle at assumed decimal odds 1.92 (typical line-market price)
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
    """For a Line market: list of (pt_ms, line_value) and first inPlay timestamp."""
    history: list[tuple[int, float]] = []
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

    NAME_PATTERN = re.compile(r"^(1st|first|2nd|second) +Innings +Runs +Line$", re.IGNORECASE)

    rows = []
    n_scanned = 0
    n_line_markets = 0
    n_matched = 0
    with tarfile.open(archive, "r") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".bz2"):
                continue
            n_scanned += 1
            f = tar.extractfile(member)
            if f is None: continue
            data = f.read()
            md = header_definition(data)
            if md is None: continue
            mname = md.get("name", "")
            if not NAME_PATTERN.match(mname.strip()):
                continue
            n_line_markets += 1
            innings = 1 if mname.lower().startswith("1st") or mname.lower().startswith("first") else 2
            if innings != 1:  # full innings 2 is complicated by chase ending early
                continue
            event_name = md.get("eventName", "")
            mt_date = (md.get("marketTime") or "")[:10]
            teams_in_event = [teams_map[t] for t in teams_map if t in event_name]
            if len(set(teams_in_event)) < 2: continue
            cs_ids = lookup[
                (lookup["date"].astype(str) == mt_date) &
                (lookup["teams_set"].apply(lambda s: set(teams_in_event[:2]).issubset(s)))
            ]["match_id"].unique()
            if len(cs_ids) == 0:
                continue
            match_id = cs_ids[0]
            n_matched += 1
            try:
                raw = bz2.decompress(data)
            except OSError:
                continue
            hist, inplay_start = collect_line_history(raw)
            if not hist or inplay_start is None:
                continue
            actual = compute_actual_innings1(match_id, balls)
            if actual is None:
                continue
            line_open = hist[0][1]
            line_t10 = line_at_or_before(hist, inplay_start - 10 * 60_000)
            line_t1  = line_at_or_before(hist, inplay_start - 60_000)
            rows.append({
                "match_id": match_id,
                "league": league_by_match.get(str(match_id)),
                "season": season_map.get(match_id),
                "venue": venue_map.get(match_id),
                "innings": innings,
                "actual_total": actual,
                "line_open":     line_open,
                "line_t_minus_10": line_t10,
                "line_t_minus_1":  line_t1,
                "n_updates": len(hist),
            })
    logger.info("scanned %d, line markets: %d, matched: %d", n_scanned, n_line_markets, n_matched)

    df = pd.DataFrame(rows).dropna(subset=["line_t_minus_1"])
    logger.info("rows with T-1 line: %d", len(df))
    df["batting_team"] = df.apply(lambda r: bat_team_map.get((r["match_id"], int(r["innings"])), ""), axis=1)
    df["bowling_team"] = df.apply(
        lambda r: next(
            (t for (m, inn), t in bat_team_map.items() if m == r["match_id"] and inn != int(r["innings"])), ""), axis=1)

    # Load saved model + lookups
    model_dir = Path("models")
    with open(model_dir / "full_innings_meta.json") as f:
        meta = json.load(f)
    features = meta["features"]
    default_par = meta["default_par"]
    leagues_meta = meta["leagues"]
    booster = lgb.Booster(model_file=str(model_dir / "full_innings_gbm.lgb"))
    with open(model_dir / "full_innings_bat_pp.json") as f:
        bat_pp = json.load(f)
    with open(model_dir / "full_innings_bowl_pp.json") as f:
        bowl_pp = json.load(f)
    with open(model_dir / "full_innings_venue_par.json") as f:
        venue_par = json.load(f)

    def lookup_pp(table, team, season):
        return float(table.get(f"{team}|{int(season)}", default_par))

    df["bat_prior"] = df.apply(lambda r: lookup_pp(bat_pp, r["batting_team"], r["season"]), axis=1)
    df["bowl_prior"] = df.apply(lambda r: lookup_pp(bowl_pp, r["bowling_team"], r["season"]), axis=1)
    df["venue_par"] = df.apply(lambda r: lookup_pp(venue_par, r["venue"], r["season"]), axis=1)

    # For each match, compute model P(actual >= line_t_minus_1) using the saved model.
    # The model takes 'threshold_X' as a feature; we set it = the line value.
    # implied_open is the model's other feature (the market price's implied); we use 0.5
    # since line markets are effectively 50/50 at standard odds.
    LINE_ODDS = 1.92  # typical exchange line market decimal odds
    IMPLIED_AT_LINE_ODDS = 1.0 / LINE_ODDS  # ~0.521
    df["threshold_X"] = df["line_t_minus_1"].round().astype(int)
    df["implied_open"] = IMPLIED_AT_LINE_ODDS
    df["x_minus_par"] = df["threshold_X"] - df["venue_par"]
    df["x_minus_bat"] = df["threshold_X"] - df["bat_prior"]
    df["x_minus_bowl"] = df["threshold_X"] - df["bowl_prior"]
    for L in leagues_meta:
        df[f"is_{L}"] = (df["league"] == L).astype(int)
    df = df.dropna(subset=features + ["season"])
    df["model_p"] = booster.predict(df[features].to_numpy(dtype=np.float32))
    # actual outcome relative to line
    df["actual_over_line"] = (df["actual_total"] > df["line_t_minus_1"]).astype(int)
    df["actual_under_line"] = (df["actual_total"] < df["line_t_minus_1"]).astype(int)

    # Strategy
    EDGE = 0.05
    df["signal"] = np.where(
        df["model_p"] >= 0.50 + EDGE, "back_over",
        np.where(df["model_p"] <= 0.50 - EDGE, "back_under", "no_trade")
    )
    print(f"\n=== 1st Innings Runs Line: 2026 OOS evaluation ===")
    print(f"Matched markets: {len(df)}")
    print(f"\nSignal distribution:")
    print(df["signal"].value_counts().to_string())

    # Backtest:
    # back at odds 1.92, stake 100; win 92 * (1-commission) if right, lose 100 if wrong
    def backtest_line(sub):
        if sub.empty: return None
        won = np.where(
            sub["signal"] == "back_over",  sub["actual_over_line"] == 1,
            np.where(sub["signal"] == "back_under", sub["actual_under_line"] == 1, False)
        )
        pnl = np.where(won, 100 * (LINE_ODDS - 1) * 0.95, -100.0)
        return {
            "n": int(len(sub)), "pnl": float(pnl.sum()),
            "roi": float(pnl.sum() / (len(sub) * 100)) if len(sub) else 0.0,
            "win_rate": float(np.mean(won)),
        }

    trades = df[df["signal"] != "no_trade"]
    print(f"\n=== Backtest (stake £100/trade, odds {LINE_ODDS}, 5%% commission) ===")
    res = backtest_line(trades)
    if res:
        print(f"Overall: n={res['n']}, pnl=£{res['pnl']:+.0f}, ROI={res['roi']:+.2%}, win={res['win_rate']:.1%}")

    print(f"\nPer signal direction:")
    for sig in ["back_over", "back_under"]:
        s = trades[trades["signal"] == sig]
        r = backtest_line(s)
        if r:
            print(f"  {sig:10s}: n={r['n']:3d}, pnl={r['pnl']:+.0f}, ROI={r['roi']:+.2%}, win={r['win_rate']:.1%}")

    print(f"\nPer league:")
    for L, g in trades.groupby("league"):
        r = backtest_line(g)
        if r and r["n"] > 0:
            print(f"  {L.upper():5s}: n={r['n']:3d}, pnl={r['pnl']:+.0f}, ROI={r['roi']:+.2%}, win={r['win_rate']:.1%}")

    # Sweep edge threshold
    print(f"\n=== Edge threshold sweep ===")
    for edge in [0.02, 0.03, 0.05, 0.07, 0.10]:
        sub = df[((df["model_p"] >= 0.50 + edge) | (df["model_p"] <= 0.50 - edge))].copy()
        sub["signal"] = np.where(sub["model_p"] >= 0.50 + edge, "back_over", "back_under")
        r = backtest_line(sub)
        if r and r["n"] > 0:
            print(f"  edge>={edge:.2f}: n={r['n']:3d}, pnl={r['pnl']:+.0f}, ROI={r['roi']:+.2%}, win={r['win_rate']:.1%}")

    out = Path("data/processed/eval_2026_line.parquet")
    df.to_parquet(out, index=False)
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

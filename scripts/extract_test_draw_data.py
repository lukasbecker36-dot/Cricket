"""Phase 1 — Extract Betfair test Match Odds draw prices, join with Cricsheet outcomes.

Output: data/processed/test_draws.parquet with one row per (test match, market_id):
  match_id_cricsheet, event_date, team1, team2, venue (city), winner,
  outcome ('win'/'draw'/'tie'/'no_result'),
  draw_price_t60, team1_price_t60, team2_price_t60,
  draw_open_price, team1_open_price, team2_open_price,
  inplay_start_ms, market_id, archive

For each market we pick the last LTP <= (inplay_start - 60s) per runner.
"""
from __future__ import annotations

import bz2
import json
import logging
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.logging_setup import configure_logging

logger = logging.getLogger(__name__)

COUNTRIES = {
    "England", "Australia", "India", "New Zealand", "South Africa", "Pakistan",
    "Sri Lanka", "West Indies", "Bangladesh", "Zimbabwe", "Afghanistan", "Ireland",
}


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


def collect_history(raw: bytes):
    """Return (history_by_runner, inplay_start_ms)."""
    hist: dict[int, list[tuple[int, float]]] = defaultdict(list)
    inplay_start = None
    last_pt = None
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
                if ltp is not None and rid is not None and last_pt is not None:
                    hist[int(rid)].append((last_pt, float(ltp)))
    for rid in hist:
        hist[rid].sort(key=lambda x: x[0])
    return hist, inplay_start


def price_at_or_before(hist_for_runner, target_ms):
    best = None
    for pt, val in hist_for_runner:
        if pt <= target_ms:
            best = val
        else:
            break
    return best


def first_price(hist_for_runner):
    return hist_for_runner[0][1] if hist_for_runner else None


def parse_cricsheet_outcomes(zpath: Path) -> pd.DataFrame:
    rows = []
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if not name.endswith(".json"):
                continue
            with z.open(name) as f:
                try:
                    d = json.load(f)
                except json.JSONDecodeError:
                    continue
            info = d.get("info") or {}
            if info.get("match_type") != "Test":
                continue
            dates = info.get("dates") or []
            if not dates:
                continue
            outcome = info.get("outcome") or {}
            if "winner" in outcome:
                result = "win"
                winner = outcome["winner"]
            elif outcome.get("result") == "draw":
                result, winner = "draw", None
            elif outcome.get("result") == "tie":
                result, winner = "tie", None
            elif outcome.get("result") == "no result":
                result, winner = "no_result", None
            else:
                result, winner = "unknown", None
            teams = info.get("teams", [])
            if len(teams) != 2:
                continue
            rows.append({
                "match_id_cricsheet": name.replace(".json", ""),
                "start_date": dates[0],
                "teams_set": frozenset(teams),
                "venue": info.get("venue", ""),
                "city": info.get("city", ""),
                "winner": winner,
                "outcome": result,
            })
    return pd.DataFrame(rows)


def main() -> int:
    configure_logging()

    # 1. Load Cricsheet outcomes
    zpath = Path("data/raw/cricsheet/tests_json.zip")
    cs = parse_cricsheet_outcomes(zpath)
    logger.info("Cricsheet test matches: %d", len(cs))

    # 2. Scan Betfair archives for test Match Odds + draw runner
    archives = [
        Path("data/raw/betfair/betfair_all_markets.dat"),
        Path("data/raw/betfair/betfair_2026.dat"),
    ]

    rows = []
    for arc in archives:
        if not arc.exists():
            continue
        logger.info("Scanning %s", arc.name)
        with tarfile.open(arc, "r") as tar:
            n_scanned = 0
            n_matched = 0
            for member in tar:
                if not member.isfile() or not member.name.endswith(".bz2"):
                    continue
                n_scanned += 1
                f = tar.extractfile(member)
                if f is None:
                    continue
                data = f.read()
                md = header_definition(data)
                if md is None:
                    continue
                if md.get("name") != "Match Odds":
                    continue
                runners_meta = md.get("runners", [])
                runner_names = {r.get("id"): r.get("name") for r in runners_meta}
                if "The Draw" not in runner_names.values():
                    continue
                ev = md.get("eventName", "")
                parts = ev.split(" v ")
                if len(parts) != 2:
                    continue
                t1, t2 = parts[0].strip(), parts[1].strip()
                if t1 not in COUNTRIES or t2 not in COUNTRIES:
                    continue
                mt_date = (md.get("marketTime") or "")[:10]
                # Decompress full stream
                try:
                    raw = bz2.decompress(data)
                except OSError:
                    continue
                hist, inplay_start = collect_history(raw)
                if inplay_start is None:
                    # market never went inplay; skip
                    continue
                # Find runner ids for t1, t2, draw
                name_to_id = {v: k for k, v in runner_names.items()}
                id_t1 = name_to_id.get(t1)
                id_t2 = name_to_id.get(t2)
                id_dr = name_to_id.get("The Draw")
                if not all([id_t1, id_t2, id_dr]):
                    continue
                target_ms = inplay_start - 60_000
                row = {
                    "event": ev,
                    "team1": t1,
                    "team2": t2,
                    "event_date": mt_date,
                    "market_id": md.get("id") or member.name,
                    "archive": arc.name,
                    "inplay_start_ms": inplay_start,
                    "team1_price_t60": price_at_or_before(hist.get(id_t1, []), target_ms),
                    "team2_price_t60": price_at_or_before(hist.get(id_t2, []), target_ms),
                    "draw_price_t60":  price_at_or_before(hist.get(id_dr, []), target_ms),
                    "team1_open_price": first_price(hist.get(id_t1, [])),
                    "team2_open_price": first_price(hist.get(id_t2, [])),
                    "draw_open_price": first_price(hist.get(id_dr, [])),
                }
                # Need at least the draw price
                if row["draw_price_t60"] is None:
                    continue
                rows.append(row)
                n_matched += 1
            logger.info("  %s: scanned=%d, test-match-odds-matched=%d", arc.name, n_scanned, n_matched)

    bf = pd.DataFrame(rows)
    logger.info("Betfair test market rows: %d", len(bf))

    # 3. Join with Cricsheet
    bf["teams_set"] = bf.apply(lambda r: frozenset([r["team1"], r["team2"]]), axis=1)
    joined = bf.merge(cs, on=["teams_set", "event_date"], how="left", suffixes=("", "_cs"),
                      left_on=["teams_set", "event_date"], right_on=["teams_set", "start_date"]) \
        if False else bf.merge(cs, left_on=["teams_set", "event_date"],
                               right_on=["teams_set", "start_date"], how="left")
    n_unmatched = joined["match_id_cricsheet"].isna().sum()
    logger.info("Unmatched Betfair rows (no Cricsheet outcome): %d", n_unmatched)
    joined = joined.dropna(subset=["match_id_cricsheet"]).copy()
    logger.info("Joined rows: %d", len(joined))

    # 4. De-duplicate: one row per (cricsheet match, market_id) - drop duplicate market files
    joined = joined.drop_duplicates(subset=["match_id_cricsheet", "market_id"])
    logger.info("After de-dup: %d", len(joined))

    # 5. Summary stats
    print("\n=== Joined dataset summary ===")
    print(f"Total rows: {len(joined)}")
    print(f"Unique tests (cricsheet match_id): {joined['match_id_cricsheet'].nunique()}")
    print(f"\nOutcome distribution:")
    print(joined.drop_duplicates("match_id_cricsheet")["outcome"].value_counts(normalize=True).mul(100).round(1).to_string())

    print(f"\nMarket-implied draw probability distribution (T-60s):")
    joined["draw_implied_t60"] = 1.0 / joined["draw_price_t60"]
    desc = joined.drop_duplicates("match_id_cricsheet")["draw_implied_t60"].describe()
    print(desc.round(3).to_string())

    # Calibration sanity: market implied vs actual
    print(f"\nMarket draw prob deciles vs actual draw rate:")
    unique = joined.drop_duplicates("match_id_cricsheet").copy()
    unique["actual_draw"] = (unique["outcome"] == "draw").astype(int)
    unique["bucket"] = pd.qcut(unique["draw_implied_t60"], q=5, duplicates="drop")
    print(unique.groupby("bucket", observed=True).agg(
        n=("actual_draw", "size"),
        market_prob=("draw_implied_t60", "mean"),
        actual_rate=("actual_draw", "mean"),
    ).round(3).to_string())

    out = Path("data/processed/test_draws.parquet")
    joined_to_save = joined.drop(columns=["teams_set"])
    joined_to_save.to_parquet(out, index=False)
    logger.info("Saved: %s", out)

    # 6. Naive lay-the-draw P&L simulation
    print("\n=== Naive lay-the-draw P&L (lay every market, £10 stake, 5% commission) ===")
    unique_for_pnl = joined.drop_duplicates("match_id_cricsheet").copy()
    unique_for_pnl["lay_price"] = unique_for_pnl["draw_price_t60"]
    unique_for_pnl["is_draw"] = (unique_for_pnl["outcome"] == "draw").astype(int)
    unique_for_pnl["is_tie"] = (unique_for_pnl["outcome"] == "tie").astype(int)
    unique_for_pnl["is_noresult"] = (unique_for_pnl["outcome"] == "no_result").astype(int)
    STAKE = 10.0
    COMMISSION = 0.05
    # Lay payoff: win STAKE * (1 - commission) if NOT draw, else lose STAKE * (price - 1)
    # For ties: in cricket Match Odds with The Draw, a tie typically goes to "The Draw" runner.
    # Treat tie + draw together as "draw side wins"
    draw_side_wins = (unique_for_pnl["is_draw"] | unique_for_pnl["is_tie"]).astype(bool)
    unique_for_pnl["pnl"] = STAKE * (1 - COMMISSION)
    unique_for_pnl.loc[draw_side_wins, "pnl"] = -STAKE * (unique_for_pnl["lay_price"] - 1)
    unique_for_pnl.loc[unique_for_pnl["is_noresult"] == 1, "pnl"] = 0.0  # void

    print(f"  n trades: {len(unique_for_pnl)}")
    print(f"  n draws (lay losses): {draw_side_wins.sum()}")
    print(f"  n results (lay wins): {(~draw_side_wins & (unique_for_pnl['is_noresult'] == 0)).sum()}")
    print(f"  n void (no result): {unique_for_pnl['is_noresult'].sum()}")
    print(f"  total P&L: £{unique_for_pnl['pnl'].sum():+.2f}")
    print(f"  ROI on stake: {unique_for_pnl['pnl'].sum() / (len(unique_for_pnl) * STAKE):+.2%}")
    avg_liability = (unique_for_pnl["lay_price"] - 1).mean() * STAKE
    print(f"  avg liability per trade: £{avg_liability:.2f}")
    print(f"  ROI on capital tied up: {unique_for_pnl['pnl'].sum() / (len(unique_for_pnl) * (avg_liability + STAKE)):+.2%}")

    print("\n=== Year-by-year breakdown ===")
    unique_for_pnl["year"] = unique_for_pnl["event_date"].str[:4]
    yr = unique_for_pnl.groupby("year").agg(
        n=("pnl", "size"),
        draws=("is_draw", "sum"),
        pnl=("pnl", "sum"),
    ).round(2)
    print(yr.to_string())

    print("\n=== By market-implied draw bucket ===")
    unique_for_pnl["draw_implied"] = 1.0 / unique_for_pnl["lay_price"]
    unique_for_pnl["bucket"] = pd.qcut(unique_for_pnl["draw_implied"], q=5, duplicates="drop")
    bt = unique_for_pnl.groupby("bucket", observed=True).agg(
        n=("pnl", "size"),
        market_prob=("draw_implied", "mean"),
        actual_rate=("is_draw", "mean"),
        pnl=("pnl", "sum"),
    ).round(3)
    bt["roi_on_stake"] = (bt["pnl"] / (bt["n"] * STAKE)).round(3)
    print(bt.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

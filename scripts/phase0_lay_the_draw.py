"""Phase 0 de-risk for lay-the-draw test cricket project.

Three checks:
  1. Cricsheet test outcome distribution (overall + per team-pair + per venue)
  2. oddspapi.io draw runner verification on an upcoming test
  3. Note availability of historical Betfair test archives

Outputs are printed to stdout. No git commits, no model files.
"""
from __future__ import annotations

import json
import os
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from urllib.request import urlopen, Request


def parse_outcome(info: dict) -> str:
    out = info.get("outcome", {})
    if "winner" in out:
        return "win"
    if out.get("result") == "draw":
        return "draw"
    if out.get("result") == "tie":
        return "tie"
    if out.get("result") == "no result":
        return "no_result"
    return "unknown"


def main() -> int:
    print("=" * 60)
    print("PHASE 0 — lay-the-draw de-risk")
    print("=" * 60)

    # --- 1. Cricsheet test outcome distribution ---
    zpath = Path("data/raw/cricsheet/tests_json.zip")
    if not zpath.exists():
        print(f"ERROR: missing {zpath}")
        return 1

    rows = []
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if n.endswith(".json")]
        for name in names:
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
            outcome = parse_outcome(info)
            winner = (info.get("outcome") or {}).get("winner")
            rows.append({
                "match_id": name.replace(".json", ""),
                "start_date": dates[0],
                "year": int(str(dates[0])[:4]),
                "teams": tuple(sorted(info.get("teams", []))),
                "venue": info.get("venue", ""),
                "city": info.get("city", ""),
                "outcome": outcome,
                "winner": winner,
                "match_days": len(dates),
            })

    df = pd.DataFrame(rows)
    print(f"\nTotal tests in Cricsheet: {len(df)}")
    print(f"Date range: {df['start_date'].min()} to {df['start_date'].max()}")

    print("\n=== A. Overall outcome distribution ===")
    print(df["outcome"].value_counts(normalize=True).mul(100).round(1).to_string())
    print(f"\n  Draw rate (overall): {(df['outcome']=='draw').mean():.1%}")

    print("\n=== B. Draw rate by year (last 10 years) ===")
    recent = df[df["year"] >= df["year"].max() - 10]
    yearly = recent.groupby("year").agg(
        n=("match_id", "size"),
        draw_rate=("outcome", lambda x: (x == "draw").mean()),
    )
    print(yearly.round(3).to_string())

    print("\n=== C. Draw rate by team-pair (pairs with >= 20 matches) ===")
    pair_stats = df.groupby("teams").agg(
        n=("match_id", "size"),
        draws=("outcome", lambda x: (x == "draw").sum()),
        draw_rate=("outcome", lambda x: (x == "draw").mean()),
    ).query("n >= 20").sort_values("draw_rate", ascending=False)
    print(pair_stats.round(3).to_string())

    print("\n=== D. Draw rate by venue (>= 15 matches) ===")
    venue_stats = df.groupby("venue").agg(
        n=("match_id", "size"),
        draw_rate=("outcome", lambda x: (x == "draw").mean()),
    ).query("n >= 15").sort_values("draw_rate", ascending=False).head(20)
    print(venue_stats.round(3).to_string())

    # ENG-NZ specific
    print("\n=== E. England vs New Zealand head-to-head ===")
    eng_nz = df[df["teams"].apply(lambda t: set(t) == {"England", "New Zealand"})]
    print(f"  Total: {len(eng_nz)} tests")
    print(f"  Draws: {(eng_nz['outcome']=='draw').sum()} ({(eng_nz['outcome']=='draw').mean():.1%})")
    print(f"  Last 10 ENG-NZ tests outcomes: {eng_nz.sort_values('start_date').tail(10)['outcome'].tolist()}")

    # --- 2. oddspapi draw runner verification ---
    print("\n=== F. oddspapi.io test cricket coverage ===")
    api_key = os.environ.get("ODDSPAPI_KEY", "12d98085-f712-4517-916f-8e61e6ff76db")
    # Look for International Tests competition; oddspapi calls this in their list
    try:
        with urlopen(f"https://api.oddspapi.io/v4/tournaments?sportId=27&apiKey={api_key}", timeout=15) as r:
            tournaments = json.loads(r.read())
        test_tournaments = [t for t in tournaments
                            if "test" in (t.get("tournamentName") or "").lower()]
        print(f"  Cricket tournaments containing 'test' in name: {len(test_tournaments)}")
        for t in test_tournaments[:10]:
            print(f"    {t.get('tournamentId')} | {t.get('tournamentName')}")
    except Exception as e:
        print(f"  ERROR fetching tournaments: {e}")
        test_tournaments = []

    # Try to find a current/upcoming test on Betfair Exchange
    if test_tournaments:
        for t in test_tournaments[:3]:
            tid = t.get("tournamentId")
            try:
                url = f"https://api.oddspapi.io/v4/odds-by-tournaments?bookmaker=betfair-ex&tournamentIds={tid}&apiKey={api_key}"
                with urlopen(url, timeout=15) as r:
                    fixtures = json.loads(r.read())
                if isinstance(fixtures, dict) and "error" in fixtures:
                    continue
                if not fixtures:
                    print(f"    {t.get('tournamentName')}: 0 fixtures with Betfair odds")
                    continue
                print(f"\n  {t.get('tournamentName')}: {len(fixtures)} fixtures")
                # Inspect first fixture's markets
                fx = fixtures[0]
                bf = (fx.get("bookmakerOdds") or {}).get("betfair-ex") or {}
                mkts = bf.get("markets") or {}
                print(f"    Sample fixture {fx.get('fixtureId')} markets: {list(mkts.keys())}")
                # Check market 273 = 1X2
                if "273" in mkts:
                    outcomes = mkts["273"].get("outcomes") or {}
                    print(f"    1X2 outcomes (draw runner check):")
                    for oid, o in outcomes.items():
                        players = (o.get("players") or {}).get("0") or {}
                        price = players.get("price")
                        traded = (players.get("exchangeMeta") or {}).get("tradedVolume")
                        print(f"      outcome={oid}  price={price}  tradedVolume={traded}")
                    print("    -> if draw runner shows price + volume, oddspapi is viable")
                else:
                    print("    !!! 1X2 (market 273) NOT in markets for this fixture")
            except Exception as e:
                print(f"    error for {tid}: {e}")
                continue

    # --- 3. Betfair test archive note ---
    print("\n=== G. Historical Betfair test archives ===")
    betfair_archives = list(Path("data/raw/betfair").glob("*.dat"))
    print(f"  Local Betfair .dat archives: {len(betfair_archives)}")
    for p in betfair_archives:
        print(f"    {p.name} ({p.stat().st_size // (1024*1024)} MB)")
    print("  NOTE: existing archives are T20 only. For test backtest we either:")
    print("    a) Ask user to source Betfair test archives (likely paid Betfair PRO data)")
    print("    b) Scrape Pinnacle historical odds as a proxy (less accurate, OK for sanity check)")
    print("    c) Forward-validate only on ENG-NZ + future series (no historical backtest)")

    print("\n" + "=" * 60)
    print("PHASE 0 SUMMARY (read the numbers above):")
    print("=" * 60)
    overall_draw = (df['outcome']=='draw').mean()
    print(f"  Draw rate overall: {overall_draw:.1%}")
    print(f"  Variance per pair: max {pair_stats['draw_rate'].max():.1%} | min {pair_stats['draw_rate'].min():.1%}")
    print(f"  Recent (last 5 years) draw rate: {df[df['year'] >= df['year'].max()-5]['outcome'].eq('draw').mean():.1%}")
    print()
    print("Decision gates:")
    print(f"  Overall draw rate in [10%, 35%]?  {'YES' if 0.10 <= overall_draw <= 0.35 else 'NO -- pivot'}")
    print(f"  Per-pair variance > 15pp?  {'YES' if (pair_stats['draw_rate'].max() - pair_stats['draw_rate'].min()) > 0.15 else 'NO -- features may not help'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

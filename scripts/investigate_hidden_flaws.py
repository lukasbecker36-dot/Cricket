"""Investigate two suspected hidden flaws:
  #1 Rain-reduced innings polluting the full_innings training set.
  #2 Team rebrands / new franchises falling through to default_par.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingestion.storage import read_balls

LEAGUES = ["ipl", "bbl", "psl", "cpl", "ntb"]


def main():
    all_balls, league_by_match = [], {}
    matches_meta = []
    for lg in LEAGUES:
        d = Path("data/processed") if lg == "ipl" else Path(f"data/processed/{lg}")
        b = read_balls(d)
        if not b.empty:
            all_balls.append(b)
        m = pd.read_parquet(d / "matches.parquet")
        m["league"] = lg
        matches_meta.append(m)
        for _, r in m.iterrows():
            league_by_match[str(r["match_id"])] = lg
    balls = pd.concat(all_balls, ignore_index=True)

    # ---------- #1 rain-reduced innings ----------
    print("=" * 64)
    print("#1  RAIN-REDUCED INNINGS IN full_innings TRAINING")
    print("=" * 64)
    rows = []
    for (mid, inn), g in balls.groupby(["match_id", "innings"]):
        if int(inn) != 1:
            continue
        li = int(g["is_legal_delivery"].sum())
        wk = int(g["wicket"].fillna(False).astype(bool).sum())
        total = int(g["runs_total"].sum())
        rows.append({"match_id": mid, "season": int(g["season"].iloc[0]),
                     "league": league_by_match.get(str(mid), "?"),
                     "legal_balls": li, "wickets": wk, "total": total})
    inn1 = pd.DataFrame(rows)
    # current rule: counted if legal_balls >= 30
    counted = inn1[inn1["legal_balls"] >= 30]
    # "complete" = ~full 20 overs (>=118 legal) OR all out (10 wickets)
    complete = counted[(counted["legal_balls"] >= 118) | (counted["wickets"] >= 10)]
    reduced = counted[(counted["legal_balls"] < 118) & (counted["wickets"] < 10)]
    print(f"Total inn1 counted in full_innings training (>=30 balls): {len(counted)}")
    print(f"  complete (>=118 balls or all-out): {len(complete)}")
    print(f"  REDUCED (<118 balls, not all-out): {len(reduced)}  ({len(reduced)/len(counted):.1%})")
    print(f"\nReduced-over share by league:")
    tab = counted.assign(reduced=(counted["legal_balls"] < 118) & (counted["wickets"] < 10)) \
                 .groupby("league")["reduced"].agg(["mean", "sum", "size"]).round(3)
    print(tab.to_string())
    print(f"\nMean total: complete={complete['total'].mean():.1f}  reduced={reduced['total'].mean():.1f}")
    print(f"  -> reduced innings are {complete['total'].mean()-reduced['total'].mean():.1f} runs lower on average,")
    print(f"     mislabelled as full-innings totals (drags priors/labels DOWN).")

    # ---------- #2 team prior misses ----------
    print("\n" + "=" * 64)
    print("#2  TEAM REBRANDS / NEW FRANCHISES -> default_par MISSES")
    print("=" * 64)
    # Load the deployed full_innings priors
    bat = json.loads(Path("models/full_innings_bat_prior.json").read_text())
    bat_keys = set(bat.keys())
    # For each (batting_team, season) actually played, is there a prior key?
    bt = balls[balls["innings"] == 1][["match_id", "season", "batting_team"]].drop_duplicates()
    bt["season"] = bt["season"].astype(int)
    miss = Counter()
    hit = Counter()
    for _, r in bt.iterrows():
        key = f"{r['batting_team']}|{r['season']}"
        if key in bat_keys:
            hit[r["batting_team"]] += 1
        else:
            miss[r["batting_team"]] += 1
    total_innings = sum(hit.values()) + sum(miss.values())
    print(f"(batting_team, season) prior lookups: {total_innings}")
    print(f"  hits: {sum(hit.values())}  misses (->default_par): {sum(miss.values())} ({sum(miss.values())/total_innings:.1%})")
    print(f"\nTeams with the MOST prior misses (rebrands/new franchises/first season):")
    for team, n in miss.most_common(20):
        h = hit.get(team, 0)
        print(f"  {team:<34} misses={n:3d}  hits={h:3d}")

    # Specifically check 2024-2026 misses (recent, what we'd trade now)
    print(f"\nRecent (season>=2024) misses — teams we'd be mispricing today:")
    recent = bt[bt["season"] >= 2024]
    rmiss = Counter()
    for _, r in recent.iterrows():
        if f"{r['batting_team']}|{r['season']}" not in bat_keys:
            rmiss[r["batting_team"]] += 1
    for team, n in rmiss.most_common(20):
        print(f"  {team:<34} {n} season(s) missing prior")


if __name__ == "__main__":
    main()

"""Build leak-free per-innings player-strength features.

Two innings-level features that capture the actual XI's quality (not just the
franchise's historical average):
  bat_strength  = mean shrunk batting strike-rate of the players who batted
  bowl_strength = mean shrunk economy of the players who bowled

Leak control (per CLAUDE.md): a player's rating for season S uses ONLY seasons
< S. Aggregating over WHO batted/bowled is fine — that's known at toss; only
their PRIOR stats are used, never this match's.

Shrinkage (empirical-Bayes style): thin-sample players are pulled toward the
league's runs-per-ball, so a debutant doesn't get a wild rating.

Output: data/processed/innings_player_strength.parquet
  columns: match_id, innings, bat_strength, bowl_strength
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingestion.storage import read_balls
from src.logging_setup import configure_logging

logger = logging.getLogger(__name__)

LEAGUES = ["ipl", "bbl", "psl", "cpl", "ntb"]
BAT_PSEUDO = 120.0   # pseudo-balls pulling batting SR toward league mean
BOWL_PSEUDO = 120.0  # pseudo-balls pulling economy toward league mean


def build_player_season_ratings(balls: pd.DataFrame):
    """Return (bat_rating, bowl_rating) dicts keyed 'player|season', using
    strictly-prior-season data with shrinkage toward the league runs-per-ball."""
    legal = balls[balls["is_legal_delivery"]].copy()

    # Per (player, season) batting: runs off bat + balls faced as striker
    bat = legal.groupby(["striker", "season"]).agg(
        runs=("runs_batter", "sum"), balls=("striker", "size")).reset_index()
    bat.rename(columns={"striker": "player"}, inplace=True)
    # Per (player, season) bowling: runs conceded (batter+wide+noball) + legal balls bowled
    legal["bowl_conceded"] = legal["runs_batter"] + legal["runs_extras"]
    bowl = legal.groupby(["bowler", "season"]).agg(
        conceded=("bowl_conceded", "sum"), balls=("bowler", "size")).reset_index()
    bowl.rename(columns={"bowler": "player"}, inplace=True)

    seasons = sorted(balls["season"].unique())
    bat_rating, bowl_rating = {}, {}
    for s in seasons:
        prior = legal[legal["season"] < s]
        if prior.empty:
            continue
        league_rpb = prior["runs_batter"].sum() / max(1, len(prior))  # runs per ball baseline
        # batting: cumulative prior runs/balls per player
        pb = bat[bat["season"] < s].groupby("player").agg(runs=("runs", "sum"), balls=("balls", "sum"))
        for player, r in pb.iterrows():
            shrunk = (r["runs"] + BAT_PSEUDO * league_rpb) / (r["balls"] + BAT_PSEUDO)
            bat_rating[f"{player}|{s}"] = shrunk * 100.0  # strike rate
        # bowling: cumulative prior conceded/balls per player
        pw = bowl[bowl["season"] < s].groupby("player").agg(conceded=("conceded", "sum"), balls=("balls", "sum"))
        for player, r in pw.iterrows():
            shrunk = (r["conceded"] + BOWL_PSEUDO * league_rpb) / (r["balls"] + BOWL_PSEUDO)
            bowl_rating[f"{player}|{s}"] = shrunk * 6.0  # economy (runs/over)
    return bat_rating, bowl_rating, league_rpb


def main() -> int:
    configure_logging()
    all_balls = []
    for lg in LEAGUES:
        d = Path("data/processed") if lg == "ipl" else Path(f"data/processed/{lg}")
        b = read_balls(d)
        if not b.empty:
            all_balls.append(b)
    balls = pd.concat(all_balls, ignore_index=True)
    logger.info("loaded %d balls", len(balls))

    bat_rating, bowl_rating, _ = build_player_season_ratings(balls)
    logger.info("player-season ratings: %d batting, %d bowling", len(bat_rating), len(bowl_rating))

    # Precompute per-season default ratings ONCE (mean of that season's rated
    # players) — fallback for players with no prior rating.
    def season_defaults(table):
        acc: dict[int, list] = {}
        for k, v in table.items():
            s = int(k.rsplit("|", 1)[1])
            acc.setdefault(s, []).append(v)
        overall = float(np.mean(list(table.values()))) if table else 0.0
        return {s: float(np.mean(vs)) for s, vs in acc.items()}, overall

    bat_def_by_season, bat_overall = season_defaults(bat_rating)
    bowl_def_by_season, bowl_overall = season_defaults(bowl_rating)

    rows = []
    for (mid, innings), g in balls.groupby(["match_id", "innings"]):
        season = int(g["season"].iloc[0])
        batters = set(g["striker"].dropna()) | set(g["non_striker"].dropna())
        bowlers = set(g[g["is_legal_delivery"]]["bowler"].dropna())
        bat_def = bat_def_by_season.get(season, bat_overall)
        bowl_def = bowl_def_by_season.get(season, bowl_overall)
        bat_vals = [bat_rating.get(f"{p}|{season}", bat_def) for p in batters]
        bowl_vals = [bowl_rating.get(f"{p}|{season}", bowl_def) for p in bowlers]
        if not bat_vals or not bowl_vals:
            continue
        rows.append({
            "match_id": mid, "innings": int(innings),
            "bat_strength": float(np.mean(bat_vals)),
            "bowl_strength": float(np.mean(bowl_vals)),
            "season": season,
        })
    df = pd.DataFrame(rows)
    out = Path("data/processed/innings_player_strength.parquet")
    df.to_parquet(out, index=False)
    logger.info("saved %s (%d innings)", out, len(df))

    # sanity: do high bat_strength innings score more? correlate with actual phase-6 total
    print("\n=== sanity check ===")
    print(df[["bat_strength", "bowl_strength"]].describe().round(2).to_string())
    print(f"\nbat_strength range: {df['bat_strength'].min():.0f}-{df['bat_strength'].max():.0f} (strike rate)")
    print(f"bowl_strength range: {df['bowl_strength'].min():.1f}-{df['bowl_strength'].max():.1f} (economy)")
    # correlate bat_strength with innings full total
    tot = balls.groupby(["match_id", "innings"])["runs_total"].sum().reset_index().rename(columns={"runs_total": "total"})
    m = df.merge(tot, on=["match_id", "innings"])
    print(f"\ncorr(bat_strength, innings total): {m['bat_strength'].corr(m['total']):+.3f}")
    print(f"corr(bowl_strength, innings total): {m['bowl_strength'].corr(m['total']):+.3f}  (expect +: weaker attacks concede more)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

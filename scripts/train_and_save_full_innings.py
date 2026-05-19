"""Train the full_innings selectivity model on ALL available data and save
to disk so the live runner can load it.

Saves:
  models/full_innings_gbm.lgb       LightGBM booster
  models/full_innings_meta.json     features, threshold, league defaults, leagues list
  models/full_innings_bat_pp.json   {team_season_key: prior batting innings avg}
  models/full_innings_bowl_pp.json  {team_season_key: prior bowling concede avg}
  models/full_innings_venue_par.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.ingestion.storage import read_balls
from src.logging_setup import configure_logging

logger = logging.getLogger(__name__)

# Locked threshold from triple-fold validation (see hardening script)
EDGE_THRESHOLD = -0.03
MODEL_DIR = Path("models")


def build_team_stats_full_innings(balls: pd.DataFrame):
    rows = []
    for (mid, innings), g in balls.groupby(["match_id", "innings"]):
        g = g.sort_values(["over", "ball", "is_legal_delivery"], ascending=[True, True, False]).reset_index(drop=True)
        legal_idx = np.where(g["is_legal_delivery"].values)[0]
        if len(legal_idx) < 30:
            continue
        total = int(g["runs_total"].sum())
        rows.append({
            "season": int(g["season"].iloc[0]),
            "batting_team": g["batting_team"].iloc[0],
            "bowling_team": g["bowling_team"].iloc[0],
            "venue": g["venue"].iloc[0],
            "total": total,
        })
    pp_df = pd.DataFrame(rows)
    bat, bowl, ven = {}, {}, {}
    for s in sorted(pp_df["season"].unique()):
        prior = pp_df[pp_df["season"] < s]
        if prior.empty:
            continue
        for t, m in prior.groupby("batting_team")["total"].mean().items():
            bat[f"{t}|{int(s)}"] = float(m)
        for t, m in prior.groupby("bowling_team")["total"].mean().items():
            bowl[f"{t}|{int(s)}"] = float(m)
        for v, m in prior.groupby("venue")["total"].mean().items():
            ven[f"{v}|{int(s)}"] = float(m)
    return bat, bowl, ven


def main() -> int:
    configure_logging()
    df = pd.read_parquet("data/processed/all_innings_markets.parquet")
    df = df[df["market_type"] == "full_innings"].copy()

    leagues = ["ipl", "bbl", "psl", "cpl", "ntb"]
    all_balls = []
    for league in leagues:
        d = Path("data/processed") if league == "ipl" else Path(f"data/processed/{league}")
        b = read_balls(d)
        if not b.empty:
            all_balls.append(b)
    balls = pd.concat(all_balls, ignore_index=True)
    season_map = balls[["match_id", "season"]].drop_duplicates("match_id").set_index("match_id")["season"].astype(int).to_dict()
    venue_map = balls[["match_id", "venue"]].drop_duplicates("match_id").set_index("match_id")["venue"].to_dict()
    bat_team_map = {(mid, int(inn)): g["batting_team"].iloc[0]
                    for (mid, inn), g in balls.groupby(["match_id", "innings"])}

    bat_pp, bowl_pp, ven_par = build_team_stats_full_innings(balls)
    default_par = float(np.mean(list(ven_par.values()))) if ven_par else 150.0

    df["season"] = df["match_id"].astype(str).map(season_map)
    df["venue"] = df["match_id"].astype(str).map(venue_map)
    df["batting_team"] = df.apply(lambda r: bat_team_map.get((r["match_id"], int(r["innings"])), ""), axis=1)
    df["bowling_team"] = df.apply(
        lambda r: next(
            (t for (m, inn), t in bat_team_map.items() if m == r["match_id"] and inn != int(r["innings"])), ""), axis=1)
    df["bat_prior"]  = df.apply(lambda r: bat_pp.get(f"{r['batting_team']}|{int(r['season'])}", default_par), axis=1)
    df["bowl_prior"] = df.apply(lambda r: bowl_pp.get(f"{r['bowling_team']}|{int(r['season'])}", default_par), axis=1)
    df["venue_par"]  = df.apply(lambda r: ven_par.get(f"{r['venue']}|{int(r['season'])}", default_par), axis=1)
    df = df.dropna(subset=["first_ltp"])
    df = df[df["first_ltp"] > 1.0].copy()
    df["implied_open"] = 1.0 / df["first_ltp"]
    df["x_minus_par"]  = df["threshold_X"] - df["venue_par"]
    df["x_minus_bat"]  = df["threshold_X"] - df["bat_prior"]
    df["x_minus_bowl"] = df["threshold_X"] - df["bowl_prior"]
    for L in leagues:
        df[f"is_{L}"] = (df["league"] == L).astype(int)

    features = [
        "threshold_X", "implied_open",
        "bat_prior", "bowl_prior", "venue_par",
        "x_minus_par", "x_minus_bat", "x_minus_bowl",
        "innings",
    ] + [f"is_{L}" for L in leagues]
    df = df.dropna(subset=["season", "actual_over_X"] + features)
    logger.info("training on full dataset: %d rows", len(df))

    x = df[features].to_numpy(dtype=np.float32)
    y = df["actual_over_X"].to_numpy(dtype=np.float32)
    # Internal 90/10 split for early stopping only (not for tuning)
    cut = int(len(x) * 0.9)
    booster = lgb.train(
        params={
            "objective": "binary", "metric": "binary_logloss",
            "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 100,
            "verbose": -1, "seed": 17, "deterministic": True,
        },
        train_set=lgb.Dataset(x[:cut], label=y[:cut], feature_name=features),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(x[cut:], label=y[cut:], feature_name=features)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(MODEL_DIR / "full_innings_gbm.lgb"))

    meta = {
        "features": features,
        "edge_threshold": EDGE_THRESHOLD,
        "implied_min": 0.10,
        "implied_max": 0.90,
        "leagues": leagues,
        "default_par": default_par,
        "trained_rows": int(len(x)),
        "best_iter": int(booster.best_iteration),
    }
    (MODEL_DIR / "full_innings_meta.json").write_text(json.dumps(meta, indent=2))
    (MODEL_DIR / "full_innings_bat_pp.json").write_text(json.dumps(bat_pp))
    (MODEL_DIR / "full_innings_bowl_pp.json").write_text(json.dumps(bowl_pp))
    (MODEL_DIR / "full_innings_venue_par.json").write_text(json.dumps(ven_par))
    logger.info("saved model + metadata to %s", MODEL_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

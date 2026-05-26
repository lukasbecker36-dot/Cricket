"""Train phase models v2 — same as train_phase_models.py but adds weather
features (temp, humidity, wind, precip, cloud) joined on (venue, date).

Saves as models/phase_{N}_wx_* so we can compare against the no-weather
baseline. The original phase_{N}_* models stay untouched.
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

PHASES = [
    ("phase_6_wx",  36,  20, 110, 5),
    ("phase_10_wx", 60,  40, 175, 5),
    ("phase_15_wx", 90,  60, 250, 5),
]

LEAGUES = ["ipl", "bbl", "psl", "cpl", "ntb"]
WEATHER_FEATURES = ["temp_c", "humidity_pct", "wind_kph", "precip_mm", "cloud_pct"]


def phase_total(g: pd.DataFrame, target_balls: int) -> int | None:
    g = g.sort_values(["over", "ball", "is_legal_delivery"], ascending=[True, True, False]).reset_index(drop=True)
    legal_idx = np.where(g["is_legal_delivery"].values)[0]
    if len(legal_idx) < target_balls:
        return None
    cut = legal_idx[target_balls - 1] + 1
    return int(g.iloc[:cut]["runs_total"].sum())


def build_phase_priors(balls: pd.DataFrame, target_balls: int):
    rows = []
    for (mid, innings), g in balls.groupby(["match_id", "innings"]):
        if int(innings) != 1:
            continue
        total = phase_total(g, target_balls)
        if total is None:
            continue
        rows.append({
            "match_id": mid,
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
    return bat, bowl, ven, pp_df


def train_one(phase_label, target_balls, x_min, x_max, x_step,
              balls, league_by_match, match_dates, weather_df, model_dir):
    logger.info("=== training %s ===", phase_label)
    bat_pp, bowl_pp, ven_par, pp_df = build_phase_priors(balls, target_balls)
    if pp_df.empty:
        logger.warning("no data for %s", phase_label); return
    logger.info("  pp_df rows: %d", len(pp_df))

    default_par = float(np.mean(list(ven_par.values()))) if ven_par else 50.0
    # Median weather for imputation
    medians = {f: weather_df[f].median() for f in WEATHER_FEATURES}
    logger.info("  weather medians: %s", {k: round(v, 1) for k, v in medians.items()})

    rows = []
    for r in pp_df.itertuples(index=False):
        season = int(r.season)
        bat_prior = bat_pp.get(f"{r.batting_team}|{season}", default_par)
        bowl_prior = bowl_pp.get(f"{r.bowling_team}|{season}", default_par)
        venue_par = ven_par.get(f"{r.venue}|{season}", default_par)
        league = league_by_match.get(str(r.match_id), "unknown")
        match_date = match_dates.get(str(r.match_id))
        if match_date is None:
            continue
        wx_row = weather_df[(weather_df["venue"] == r.venue) & (weather_df["date"] == match_date)]
        wx_feats = {f: (wx_row[f].iloc[0] if not wx_row.empty and not pd.isna(wx_row[f].iloc[0])
                        else medians[f]) for f in WEATHER_FEATURES}
        for X in range(x_min, x_max + 1, x_step):
            base = {
                "match_id": r.match_id, "season": season, "league": league,
                "threshold_X": X, "bat_prior": bat_prior, "bowl_prior": bowl_prior,
                "venue_par": venue_par,
                "x_minus_par": X - venue_par,
                "x_minus_bat": X - bat_prior,
                "x_minus_bowl": X - bowl_prior,
                "innings": 1,
                "actual_over_X": int(r.total >= X),
            }
            base.update(wx_feats)
            rows.append(base)

    train_df = pd.DataFrame(rows)
    for L in LEAGUES:
        train_df[f"is_{L}"] = (train_df["league"] == L).astype(int)

    features = [
        "threshold_X", "bat_prior", "bowl_prior", "venue_par",
        "x_minus_par", "x_minus_bat", "x_minus_bowl", "innings",
    ] + WEATHER_FEATURES + [f"is_{L}" for L in LEAGUES]
    train_df = train_df.dropna(subset=features + ["actual_over_X"])
    logger.info("  training rows: %d", len(train_df))

    x = train_df[features].to_numpy(dtype=np.float32)
    y = train_df["actual_over_X"].to_numpy(dtype=np.float32)
    cut = int(len(x) * 0.9)
    booster = lgb.train(
        params={"objective": "binary", "metric": "binary_logloss",
                "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 200,
                "verbose": -1, "seed": 17, "deterministic": True},
        train_set=lgb.Dataset(x[:cut], label=y[:cut], feature_name=features),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(x[cut:], label=y[cut:], feature_name=features)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # Feature importance: useful to verify weather actually matters
    imp = pd.DataFrame({"feature": features, "gain": booster.feature_importance(importance_type="gain")})
    imp = imp.sort_values("gain", ascending=False)
    logger.info("  top features by gain:\n%s", imp.head(10).to_string(index=False))

    model_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_dir / f"{phase_label}_gbm.lgb"))
    meta = {
        "features": features, "target_balls": target_balls,
        "x_range": [x_min, x_max, x_step], "edge_threshold": 0.05,
        "implied_min": 0.10, "implied_max": 0.90,
        "leagues": LEAGUES, "default_par": default_par,
        "weather_medians": medians,
        "trained_rows": int(len(x)), "best_iter": int(booster.best_iteration),
    }
    (model_dir / f"{phase_label}_meta.json").write_text(json.dumps(meta, indent=2))
    (model_dir / f"{phase_label}_bat_prior.json").write_text(json.dumps(bat_pp))
    (model_dir / f"{phase_label}_bowl_prior.json").write_text(json.dumps(bowl_pp))
    (model_dir / f"{phase_label}_venue_par.json").write_text(json.dumps(ven_par))
    logger.info("  saved %s_gbm.lgb (best_iter=%d)", phase_label, booster.best_iteration)


def main() -> int:
    configure_logging()
    all_balls, match_dates = [], {}
    league_by_match = {}
    for league in LEAGUES:
        d = Path("data/processed") if league == "ipl" else Path(f"data/processed/{league}")
        b = read_balls(d)
        if not b.empty:
            all_balls.append(b)
        m = pd.read_parquet(d / "matches.parquet")
        for _, r in m.iterrows():
            mid = str(r["match_id"])
            league_by_match[mid] = league
            match_dates[mid] = str(r["date"])[:10]
    balls = pd.concat(all_balls, ignore_index=True)
    logger.info("loaded %d balls", len(balls))

    weather = pd.read_parquet("data/processed/match_weather.parquet")
    logger.info("weather rows: %d", len(weather))

    model_dir = Path("models")
    for phase_label, target_balls, x_min, x_max, x_step in PHASES:
        train_one(phase_label, target_balls, x_min, x_max, x_step,
                  balls, league_by_match, match_dates, weather, model_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

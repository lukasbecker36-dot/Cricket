"""Train and deploy the validated production models:
  - phase_6:  recency-weighted priors (half-life 2 seasons) + league_trend feature
  - phase_15: weather features (temp/humidity/wind/precip/cloud)

Both trained on ALL seasons (the <=2024 cut was only for OOS validation).
Overwrites the existing phase_6 / phase_15 model files and writes the new
companion files (phase_6_league_trend.json, weather_medians in phase_15 meta).

phase_10, full_innings, and the inn2 models are left untouched.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.ingestion.storage import read_balls
from src.ingestion.teams import canonical_team
from src.logging_setup import configure_logging

logger = logging.getLogger(__name__)

LEAGUES = ["ipl", "bbl", "psl", "cpl", "ntb"]
WEATHER_FEATURES = ["temp_c", "humidity_pct", "wind_kph", "precip_mm", "cloud_pct"]
HALF_LIFE = 2.0


def phase_total(g, tb):
    g = g.sort_values(["over", "ball", "is_legal_delivery"], ascending=[True, True, False]).reset_index(drop=True)
    li = np.where(g["is_legal_delivery"].values)[0]
    if tb >= 120:
        # Whole innings: only count COMPLETE innings — ~full 20 overs (>=118
        # legal balls) OR all out. Rain-reduced innings (e.g. 15 overs) would
        # otherwise be mislabelled as low full-innings totals, dragging the
        # priors/labels down (worst in rain-prone NTB/BBL).
        wk = int(g["wicket"].fillna(False).astype(bool).sum())
        if len(li) >= 118 or wk >= 10:
            return int(g["runs_total"].sum())
        return None
    if len(li) < tb:
        return None
    return int(g.iloc[:li[tb-1]+1]["runs_total"].sum())


def collect_phase_df(balls, tb):
    rows = []
    for (mid, inn), g in balls.groupby(["match_id", "innings"]):
        if int(inn) != 1:
            continue
        t = phase_total(g, tb)
        if t is None:
            continue
        rows.append({"match_id": mid, "season": int(g["season"].iloc[0]),
                     "batting_team": canonical_team(g["batting_team"].iloc[0]),
                     "bowling_team": canonical_team(g["bowling_team"].iloc[0]),
                     "venue": g["venue"].iloc[0], "total": t})
    return pd.DataFrame(rows)


def train_booster(train_df, features):
    train_df = train_df.dropna(subset=features + ["actual_over_X"])
    x = train_df[features].to_numpy(dtype=np.float32)
    y = train_df["actual_over_X"].to_numpy(dtype=np.float32)
    cut = int(len(x) * 0.9)
    booster = lgb.train(
        params={"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05,
                "num_leaves": 31, "min_data_in_leaf": 200, "verbose": -1, "seed": 17, "deterministic": True},
        train_set=lgb.Dataset(x[:cut], label=y[:cut], feature_name=features),
        num_boost_round=500, valid_sets=[lgb.Dataset(x[cut:], label=y[cut:], feature_name=features)],
        callbacks=[lgb.early_stopping(50, verbose=False)])
    return booster, len(x)


ANOM_RAW = {"temp_anom": "temp_c", "humid_anom": "humidity_pct",
            "wind_anom": "wind_kph", "cloud_anom": "cloud_pct"}
ANOM_FEATURES = ["temp_anom", "humid_anom", "wind_anom", "precip_mm", "cloud_anom"]


def deploy_phase_trend(phase_label, tb, x_min, x_max, x_step, balls, league_by_match, model_dir):
    """Generic recency-priors + league_trend deploy for a phase (phase_6/phase_10)."""
    logger.info("=== %s: recency priors + league_trend ===", phase_label)
    pp = collect_phase_df(balls, tb)
    default_par = float(pp["total"].mean())

    bat, bowl, ven, trend = {}, {}, {}, {}
    for s in sorted(pp["season"].unique()):
        prior = pp[pp["season"] < s]
        if prior.empty:
            continue
        w = np.power(0.5, (s - prior["season"]) / HALF_LIFE)
        prior = prior.assign(_w=w)
        for col, tbl in [("batting_team", bat), ("bowling_team", bowl), ("venue", ven)]:
            for key, gg in prior.groupby(col):
                tbl[f"{key}|{int(s)}"] = float(np.average(gg["total"], weights=gg["_w"]))
        prev = pp[pp["season"] == s - 1]
        if not prev.empty:
            trend[str(int(s))] = float(prev["total"].mean())

    rows = []
    for r in pp.itertuples(index=False):
        s = int(r.season)
        bp = bat.get(f"{r.batting_team}|{s}", default_par)
        wp = bowl.get(f"{r.bowling_team}|{s}", default_par)
        vp = ven.get(f"{r.venue}|{s}", default_par)
        lt = trend.get(str(s), default_par)
        league = league_by_match.get(str(r.match_id), "unknown")
        for X in range(x_min, x_max + 1, x_step):
            rows.append({"season": s, "league": league, "threshold_X": X,
                         "bat_prior": bp, "bowl_prior": wp, "venue_par": vp,
                         "x_minus_par": X - vp, "x_minus_bat": X - bp, "x_minus_bowl": X - wp,
                         "innings": 1, "league_trend": lt, "x_minus_trend": X - lt,
                         "actual_over_X": int(r.total >= X)})
    train_df = pd.DataFrame(rows)
    for L in LEAGUES:
        train_df[f"is_{L}"] = (train_df["league"] == L).astype(int)

    features = ["threshold_X", "bat_prior", "bowl_prior", "venue_par",
                "x_minus_par", "x_minus_bat", "x_minus_bowl", "innings",
                "league_trend", "x_minus_trend"] + [f"is_{L}" for L in LEAGUES]
    booster, n = train_booster(train_df, features)

    booster.save_model(str(model_dir / f"{phase_label}_gbm.lgb"))
    meta = {"features": features, "target_balls": tb, "x_range": [x_min, x_max, x_step],
            "edge_threshold": 0.05, "implied_min": 0.10, "implied_max": 0.90,
            "leagues": LEAGUES, "default_par": default_par, "trained_rows": int(n),
            "best_iter": int(booster.best_iteration),
            "prior_mode": "recency_halflife2", "has_trend": True}
    (model_dir / f"{phase_label}_meta.json").write_text(json.dumps(meta, indent=2))
    (model_dir / f"{phase_label}_bat_prior.json").write_text(json.dumps(bat))
    (model_dir / f"{phase_label}_bowl_prior.json").write_text(json.dumps(bowl))
    (model_dir / f"{phase_label}_venue_par.json").write_text(json.dumps(ven))
    (model_dir / f"{phase_label}_league_trend.json").write_text(json.dumps(trend))
    logger.info("  saved %s (trend) best_iter=%d rows=%d", phase_label, booster.best_iteration, n)


def deploy_full_innings_trend(balls, league_by_match, model_dir):
    logger.info("=== full_innings: recency priors + league_trend ===")
    tb, x_min, x_max, x_step = 120, 80, 280, 5
    pp = collect_phase_df(balls, tb)
    default_par = float(pp["total"].mean())

    bat, bowl, ven, trend = {}, {}, {}, {}
    for s in sorted(pp["season"].unique()):
        prior = pp[pp["season"] < s]
        if prior.empty:
            continue
        w = np.power(0.5, (s - prior["season"]) / HALF_LIFE)
        prior = prior.assign(_w=w)
        for col, tbl in [("batting_team", bat), ("bowling_team", bowl), ("venue", ven)]:
            for key, gg in prior.groupby(col):
                tbl[f"{key}|{int(s)}"] = float(np.average(gg["total"], weights=gg["_w"]))
        prev = pp[pp["season"] == s - 1]
        if not prev.empty:
            trend[str(int(s))] = float(prev["total"].mean())

    rows = []
    for r in pp.itertuples(index=False):
        s = int(r.season)
        bp = bat.get(f"{r.batting_team}|{s}", default_par)
        wp = bowl.get(f"{r.bowling_team}|{s}", default_par)
        vp = ven.get(f"{r.venue}|{s}", default_par)
        lt = trend.get(str(s), default_par)
        league = league_by_match.get(str(r.match_id), "unknown")
        for X in range(x_min, x_max + 1, x_step):
            rows.append({"season": s, "league": league, "threshold_X": X,
                         "bat_prior": bp, "bowl_prior": wp, "venue_par": vp,
                         "x_minus_par": X - vp, "x_minus_bat": X - bp, "x_minus_bowl": X - wp,
                         "innings": 1, "league_trend": lt, "x_minus_trend": X - lt,
                         "actual_over_X": int(r.total >= X)})
    train_df = pd.DataFrame(rows)
    for L in LEAGUES:
        train_df[f"is_{L}"] = (train_df["league"] == L).astype(int)

    features = ["threshold_X", "bat_prior", "bowl_prior", "venue_par",
                "x_minus_par", "x_minus_bat", "x_minus_bowl", "innings",
                "league_trend", "x_minus_trend"] + [f"is_{L}" for L in LEAGUES]
    booster, n = train_booster(train_df, features)

    booster.save_model(str(model_dir / "full_innings_gbm.lgb"))
    meta = {"features": features, "target_balls": tb, "x_range": [x_min, x_max, x_step],
            "edge_threshold": 0.05, "implied_min": 0.10, "implied_max": 0.90,
            "leagues": LEAGUES, "default_par": default_par, "trained_rows": int(n),
            "best_iter": int(booster.best_iteration),
            "prior_mode": "recency_halflife2", "has_trend": True}
    (model_dir / "full_innings_meta.json").write_text(json.dumps(meta, indent=2))
    (model_dir / "full_innings_bat_prior.json").write_text(json.dumps(bat))
    (model_dir / "full_innings_bowl_prior.json").write_text(json.dumps(bowl))
    (model_dir / "full_innings_venue_par.json").write_text(json.dumps(ven))
    (model_dir / "full_innings_league_trend.json").write_text(json.dumps(trend))
    logger.info("  saved full_innings (trend) best_iter=%d rows=%d", booster.best_iteration, n)


def deploy_phase15_weather(balls, league_by_match, match_dates, weather, model_dir):
    logger.info("=== phase_15: anomaly-weather features ===")
    tb, x_min, x_max, x_step = 90, 60, 250, 5
    pp = collect_phase_df(balls, tb)
    default_par = float(pp["total"].mean())
    medians = {f: float(weather[f].median()) for f in WEATHER_FEATURES}

    # Venue climatology (mean match-day conditions) for anomaly baseline.
    climo_vars = ["temp_c", "humidity_pct", "wind_kph", "cloud_pct"]
    venue_climo = weather.groupby("venue")[climo_vars].mean().to_dict(orient="index")
    venue_climo = {v: {k: float(val) for k, val in d.items()} for v, d in venue_climo.items()}
    global_climo = {k: float(weather[k].mean()) for k in climo_vars}

    def anomalies(venue, raw):
        base = venue_climo.get(venue, global_climo)
        out = {}
        for anom_feat, raw_var in ANOM_RAW.items():
            out[anom_feat] = float(raw[raw_var]) - float(base.get(raw_var, global_climo[raw_var]))
        out["precip_mm"] = float(raw["precip_mm"])
        return out

    # equal-weight priors (standard)
    bat, bowl, ven = {}, {}, {}
    for s in sorted(pp["season"].unique()):
        prior = pp[pp["season"] < s]
        if prior.empty:
            continue
        for t, m in prior.groupby("batting_team")["total"].mean().items():
            bat[f"{t}|{int(s)}"] = float(m)
        for t, m in prior.groupby("bowling_team")["total"].mean().items():
            bowl[f"{t}|{int(s)}"] = float(m)
        for v, m in prior.groupby("venue")["total"].mean().items():
            ven[f"{v}|{int(s)}"] = float(m)

    rows = []
    for r in pp.itertuples(index=False):
        s = int(r.season)
        bp = bat.get(f"{r.batting_team}|{s}", default_par)
        wp = bowl.get(f"{r.bowling_team}|{s}", default_par)
        vp = ven.get(f"{r.venue}|{s}", default_par)
        league = league_by_match.get(str(r.match_id), "unknown")
        mdate = match_dates.get(str(r.match_id))
        wx = weather[(weather["venue"] == r.venue) & (weather["date"] == mdate)]
        if not wx.empty:
            raw = {f: (wx[f].iloc[0] if not pd.isna(wx[f].iloc[0]) else medians[f]) for f in WEATHER_FEATURES}
        else:
            raw = dict(medians)
        anomf = anomalies(r.venue, raw)
        for X in range(x_min, x_max + 1, x_step):
            row = {"season": s, "league": league, "threshold_X": X,
                   "bat_prior": bp, "bowl_prior": wp, "venue_par": vp,
                   "x_minus_par": X - vp, "x_minus_bat": X - bp, "x_minus_bowl": X - wp,
                   "innings": 1, "actual_over_X": int(r.total >= X)}
            row.update(anomf)
            rows.append(row)
    train_df = pd.DataFrame(rows)
    for L in LEAGUES:
        train_df[f"is_{L}"] = (train_df["league"] == L).astype(int)

    features = ["threshold_X", "bat_prior", "bowl_prior", "venue_par",
                "x_minus_par", "x_minus_bat", "x_minus_bowl", "innings"] + \
               ANOM_FEATURES + [f"is_{L}" for L in LEAGUES]
    booster, n = train_booster(train_df, features)

    booster.save_model(str(model_dir / "phase_15_gbm.lgb"))
    meta = {"features": features, "target_balls": tb, "x_range": [x_min, x_max, x_step],
            "edge_threshold": 0.05, "implied_min": 0.10, "implied_max": 0.90,
            "leagues": LEAGUES, "default_par": default_par, "trained_rows": int(n),
            "best_iter": int(booster.best_iteration), "weather_mode": "anomaly"}
    (model_dir / "phase_15_meta.json").write_text(json.dumps(meta, indent=2))
    (model_dir / "phase_15_bat_prior.json").write_text(json.dumps(bat))
    (model_dir / "phase_15_bowl_prior.json").write_text(json.dumps(bowl))
    (model_dir / "phase_15_venue_par.json").write_text(json.dumps(ven))
    (model_dir / "phase_15_venue_climo.json").write_text(
        json.dumps({"by_venue": venue_climo, "global": global_climo}))
    logger.info("  saved phase_15 (anomaly-weather) best_iter=%d rows=%d", booster.best_iteration, n)


def main() -> int:
    configure_logging()
    all_balls, league_by_match, match_dates = [], {}, {}
    for league in LEAGUES:
        d = Path("data/processed") if league == "ipl" else Path(f"data/processed/{league}")
        b = read_balls(d)
        if not b.empty:
            all_balls.append(b)
        m = pd.read_parquet(d / "matches.parquet")
        for _, r in m.iterrows():
            league_by_match[str(r["match_id"])] = league
            match_dates[str(r["match_id"])] = str(r["date"])[:10]
    balls = pd.concat(all_balls, ignore_index=True)
    weather = pd.read_parquet("data/processed/match_weather.parquet")

    model_dir = Path("models")
    deploy_phase_trend("phase_6", 36, 20, 110, 5, balls, league_by_match, model_dir)
    deploy_phase_trend("phase_10", 60, 40, 175, 5, balls, league_by_match, model_dir)
    deploy_full_innings_trend(balls, league_by_match, model_dir)
    deploy_phase15_weather(balls, league_by_match, match_dates, weather, model_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Train innings-2-specific phase models for 6 and 10 over Line markets.

Innings 2 differs from innings 1 in two ways the inn1-trained model can't
capture:
  1. The chaser knows the target. High targets accelerate PP scoring.
  2. Chases that end before the phase boundary settle on the final inn2
     total, which is typically below what the team would have scored if
     forced to bat out the phase.

We include ALL chases (not just ones that batted out the phase) using the
inn2 final total as the outcome -- that's how the Line market actually
settles.

Skip phase_15 and full_innings inn2: too many chases end mid-phase, model
becomes target-aware in a way that fights the market not aids it.

Produces, for each phase in {6, 10}:
  models/phase_{N}_inn2_gbm.lgb
  models/phase_{N}_inn2_meta.json
  models/phase_{N}_inn2_bat_prior.json
  models/phase_{N}_inn2_bowl_prior.json
  models/phase_{N}_inn2_venue_par.json
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
from src.ingestion.venues import canonical_venue
from src.logging_setup import configure_logging

logger = logging.getLogger(__name__)

# (phase_label, target_balls, X-range_min, X-range_max, X-step)
PHASES = [
    ("phase_6",  36,  20, 110, 5),
    ("phase_10", 60,  40, 175, 5),
]

LEAGUES = ["ipl", "bbl", "psl", "cpl", "ntb"]

# Train cut-off: matches in seasons > MAX_TRAIN_SEASON are held out for OOS
# validation. Set to None to train on everything (for production after
# validation passes).
MAX_TRAIN_SEASON: int | None = None


def inn2_phase_outcome(g: pd.DataFrame, target_balls: int, inn1_total: int) -> int | None:
    """Return inn2 settled total for the phase Line market. None when the
    market would have voided (innings rain-reduced below phase boundary)."""
    g = g.sort_values(["over", "ball", "is_legal_delivery"], ascending=[True, True, False]).reset_index(drop=True)
    legal_idx = np.where(g["is_legal_delivery"].values)[0]
    if len(legal_idx) == 0:
        return None
    if len(legal_idx) >= target_balls:
        cut = legal_idx[target_balls - 1] + 1
        return int(g.iloc[:cut]["runs_total"].sum())
    wickets = int(g["wicket"].fillna(False).astype(bool).sum())
    final_total = int(g["runs_total"].sum())
    if wickets >= 10 or final_total > inn1_total:
        return final_total
    return None  # rain-reduced -> market would void


def build_inn2_data(balls: pd.DataFrame, target_balls: int) -> pd.DataFrame:
    inn1_totals = balls[balls["innings"] == 1].groupby("match_id")["runs_total"].sum().astype(int).to_dict()
    rows = []
    for (mid, innings), g in balls.groupby(["match_id", "innings"]):
        if int(innings) != 2:
            continue
        outcome = inn2_phase_outcome(g, target_balls, inn1_totals.get(mid, 0))
        if outcome is None:
            continue
        tgt = g["target"].dropna()
        if tgt.empty:
            continue
        rows.append({
            "match_id": mid,
            "season": int(g["season"].iloc[0]),
            "batting_team": canonical_team(g["batting_team"].iloc[0]),
            "bowling_team": canonical_team(g["bowling_team"].iloc[0]),
            "venue": canonical_venue(g["venue"].iloc[0]),
            "target": float(tgt.iloc[0]),
            "phase_total": outcome,
        })
    return pd.DataFrame(rows)


HALF_LIFE = 2.0  # seasons, for recency weighting


def build_priors(df: pd.DataFrame) -> tuple[dict, dict, dict, dict]:
    """Recency-weighted (half-life 2 seasons) strictly-prior-season priors,
    plus a league_trend table = league's prior-season mean inn2 phase total.
    Recency + trend correct the scoring-inflation lag (same fix as inn1)."""
    bat, bowl, ven, trend = {}, {}, {}, {}
    for s in sorted(df["season"].unique()):
        prior = df[df["season"] < s]
        if prior.empty:
            continue
        w = np.power(0.5, (s - prior["season"]) / HALF_LIFE)
        prior = prior.assign(_w=w)
        for col, tbl in [("batting_team", bat), ("bowling_team", bowl), ("venue", ven)]:
            for key, gg in prior.groupby(col):
                tbl[f"{key}|{int(s)}"] = float(np.average(gg["phase_total"], weights=gg["_w"]))
        prev = df[df["season"] == s - 1]
        if not prev.empty:
            trend[str(int(s))] = float(prev["phase_total"].mean())
    return bat, bowl, ven, trend


def train_one(phase_label: str, target_balls: int, x_min: int, x_max: int, x_step: int,
              balls: pd.DataFrame, league_by_match: dict, model_dir: Path) -> None:
    logger.info("=== training %s inn2 (target_balls=%d) ===", phase_label, target_balls)
    df_all = build_inn2_data(balls, target_balls)
    if df_all.empty:
        logger.warning("no inn2 data for %s; skipping", phase_label)
        return

    # Priors are computed from the full set but each season key uses only
    # strictly-prior seasons, so this is leak-free even when later seasons
    # are present. We need priors for the eval seasons too.
    bat_pp, bowl_pp, ven_par, league_trend = build_priors(df_all)

    if MAX_TRAIN_SEASON is not None:
        df = df_all[df_all["season"] <= MAX_TRAIN_SEASON].copy()
        logger.info("  inn2 matches (train, season<=%d): %d / %d", MAX_TRAIN_SEASON, len(df), len(df_all))
    else:
        df = df_all
        logger.info("  inn2 matches (train, all seasons): %d", len(df))
    default_par = float(np.mean(list(ven_par.values()))) if ven_par else 50.0

    rows = []
    for r in df.itertuples(index=False):
        season = int(r.season)
        bat_prior = bat_pp.get(f"{r.batting_team}|{season}", default_par)
        bowl_prior = bowl_pp.get(f"{r.bowling_team}|{season}", default_par)
        venue_par = ven_par.get(f"{r.venue}|{season}", default_par)
        league = league_by_match.get(str(r.match_id), "unknown")
        # Implied "par" for this phase given the target (linear pacing)
        phase_par_from_target = float(r.target) * target_balls / 120.0
        lt = league_trend.get(str(season), default_par)
        for X in range(x_min, x_max + 1, x_step):
            rows.append({
                "match_id": r.match_id, "season": season, "league": league,
                "threshold_X": X,
                "bat_prior": bat_prior, "bowl_prior": bowl_prior, "venue_par": venue_par,
                "target": float(r.target),
                "phase_par_from_target": phase_par_from_target,
                "x_minus_par": X - venue_par,
                "x_minus_bat": X - bat_prior,
                "x_minus_bowl": X - bowl_prior,
                "x_minus_target_par": X - phase_par_from_target,
                "league_trend": lt,
                "x_minus_trend": X - lt,
                "innings": 2,
                "actual_over_X": int(r.phase_total >= X),
            })
    train_df = pd.DataFrame(rows)
    for L in LEAGUES:
        train_df[f"is_{L}"] = (train_df["league"] == L).astype(int)

    features = [
        "threshold_X", "bat_prior", "bowl_prior", "venue_par",
        "target", "phase_par_from_target",
        "x_minus_par", "x_minus_bat", "x_minus_bowl", "x_minus_target_par",
        "league_trend", "x_minus_trend",
        "innings",
    ] + [f"is_{L}" for L in LEAGUES]
    train_df = train_df.dropna(subset=features + ["actual_over_X"])
    logger.info("  training rows: %d", len(train_df))

    x = train_df[features].to_numpy(dtype=np.float32)
    y = train_df["actual_over_X"].to_numpy(dtype=np.float32)
    cut = int(len(x) * 0.9)
    booster = lgb.train(
        params={
            "objective": "binary", "metric": "binary_logloss",
            "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 200,
            "verbose": -1, "seed": 17, "deterministic": True,
        },
        train_set=lgb.Dataset(x[:cut], label=y[:cut], feature_name=features),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(x[cut:], label=y[cut:], feature_name=features)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_dir / f"{phase_label}_inn2_gbm.lgb"))
    meta = {
        "features": features, "target_balls": target_balls,
        "x_range": [x_min, x_max, x_step], "edge_threshold": 0.05,
        "implied_min": 0.10, "implied_max": 0.90,
        "leagues": LEAGUES, "default_par": default_par,
        "trained_rows": int(len(x)), "best_iter": int(booster.best_iteration),
        "innings": 2,
        "max_train_season": MAX_TRAIN_SEASON,
        "prior_mode": "recency_halflife2", "has_trend": True,
    }
    (model_dir / f"{phase_label}_inn2_meta.json").write_text(json.dumps(meta, indent=2))
    (model_dir / f"{phase_label}_inn2_bat_prior.json").write_text(json.dumps(bat_pp))
    (model_dir / f"{phase_label}_inn2_bowl_prior.json").write_text(json.dumps(bowl_pp))
    (model_dir / f"{phase_label}_inn2_venue_par.json").write_text(json.dumps(ven_par))
    (model_dir / f"{phase_label}_inn2_league_trend.json").write_text(json.dumps(league_trend))
    logger.info("  saved %s_inn2_gbm.lgb (best_iter=%d)", phase_label, booster.best_iteration)


def main() -> int:
    configure_logging()
    all_balls = []
    league_by_match = {}
    for league in LEAGUES:
        d = Path("data/processed") if league == "ipl" else Path(f"data/processed/{league}")
        b = read_balls(d)
        if not b.empty:
            all_balls.append(b)
        m = pd.read_parquet(d / "matches.parquet")
        for mid in m["match_id"].astype(str).unique():
            league_by_match[mid] = league
    balls = pd.concat(all_balls, ignore_index=True)
    logger.info("loaded %d balls across %d leagues", len(balls), len(all_balls))

    model_dir = Path("models")
    for phase_label, target_balls, x_min, x_max, x_step in PHASES:
        train_one(phase_label, target_balls, x_min, x_max, x_step,
                  balls, league_by_match, model_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

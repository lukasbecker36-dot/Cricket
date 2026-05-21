"""Train phase-specific models for 6, 10, and 15 over Innings Runs Line markets.

These mirror the full_innings model but predict P(phase_total >= X) at the
relevant ball count, with phase-appropriate team and venue priors. Cricsheet
outcomes only (no market data needed in training -- at inference, Line markets
use implied = 1/1.92 directly).

For each phase produces:
  models/phase_{N}_gbm.lgb
  models/phase_{N}_meta.json
  models/phase_{N}_bat_prior.json
  models/phase_{N}_bowl_prior.json
  models/phase_{N}_venue_par.json
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

# (phase_label, target_balls, X-range_min, X-range_max, X-step)
PHASES = [
    ("phase_6",  36,  20, 110, 5),    # powerplay (6 overs)
    ("phase_10", 60,  40, 175, 5),
    ("phase_15", 90,  60, 250, 5),
]

LEAGUES = ["ipl", "bbl", "psl", "cpl", "ntb"]


def phase_total(g: pd.DataFrame, target_balls: int) -> int | None:
    g = g.sort_values(["over", "ball", "is_legal_delivery"], ascending=[True, True, False]).reset_index(drop=True)
    legal_idx = np.where(g["is_legal_delivery"].values)[0]
    if len(legal_idx) < target_balls:
        # Allow short innings only for full-T20 case; phase models need full phase
        return None
    cut = legal_idx[target_balls - 1] + 1
    return int(g.iloc[:cut]["runs_total"].sum())


def build_phase_priors(balls: pd.DataFrame, target_balls: int) -> tuple[dict, dict, dict]:
    """Per-(team, season) and (venue, season) rolling averages using only
    strictly-prior seasons."""
    rows = []
    for (mid, innings), g in balls.groupby(["match_id", "innings"]):
        total = phase_total(g, target_balls)
        if total is None:
            continue
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
    return bat, bowl, ven, pp_df


def build_training_rows(
    pp_df: pd.DataFrame,
    league_by_match: dict,
    bat_pp: dict, bowl_pp: dict, ven_par: dict,
    default_par: float,
    target_balls: int,
    x_min: int, x_max: int, x_step: int,
) -> pd.DataFrame:
    """For each (match, X) sweep, build a row with phase-appropriate features."""
    rows = []
    for r in pp_df.itertuples(index=False):
        season = int(r.season)
        bat_prior = bat_pp.get(f"{r.batting_team}|{season}", default_par)
        bowl_prior = bowl_pp.get(f"{r.bowling_team}|{season}", default_par)
        venue_par = ven_par.get(f"{r.venue}|{season}", default_par)
        league = league_by_match.get(str(r.match_id), "unknown")
        for X in range(x_min, x_max + 1, x_step):
            rows.append({
                "match_id": r.match_id,
                "season": season,
                "league": league,
                "threshold_X": X,
                "bat_prior": bat_prior,
                "bowl_prior": bowl_prior,
                "venue_par": venue_par,
                "x_minus_par": X - venue_par,
                "x_minus_bat": X - bat_prior,
                "x_minus_bowl": X - bowl_prior,
                "innings": 1,  # innings-1 only for training; phase totals for innings 2 are noisy
                "actual_over_X": int(r.total >= X),
            })
    df = pd.DataFrame(rows)
    df = df[df["innings"] == 1].copy()
    return df


def train_one_phase(
    phase_label: str, target_balls: int, x_min: int, x_max: int, x_step: int,
    balls: pd.DataFrame, league_by_match: dict, model_dir: Path,
) -> None:
    logger.info("=== training %s (target_balls=%d) ===", phase_label, target_balls)
    bat_pp, bowl_pp, ven_par, pp_df = build_phase_priors(balls, target_balls)
    if pp_df.empty:
        logger.warning("no phase data for %s; skipping", phase_label)
        return

    # Filter to first innings only -- phase prediction in innings 2 is dominated
    # by chase-end effects (target met -> innings stops) and isn't a clean
    # P(total>=X) signal.
    rows_inn1 = []
    for (mid, innings), g in balls.groupby(["match_id", "innings"]):
        if int(innings) != 1:
            continue
        total = phase_total(g, target_balls)
        if total is None:
            continue
        rows_inn1.append({
            "match_id": mid,
            "season": int(g["season"].iloc[0]),
            "batting_team": g["batting_team"].iloc[0],
            "bowling_team": g["bowling_team"].iloc[0],
            "venue": g["venue"].iloc[0],
            "total": total,
        })
    pp_df = pd.DataFrame(rows_inn1)
    logger.info("  innings-1 matches with complete phase: %d", len(pp_df))

    default_par = float(np.mean(list(ven_par.values()))) if ven_par else 50.0

    train_df = build_training_rows(
        pp_df, league_by_match, bat_pp, bowl_pp, ven_par,
        default_par, target_balls, x_min, x_max, x_step,
    )
    for L in LEAGUES:
        train_df[f"is_{L}"] = (train_df["league"] == L).astype(int)

    features = [
        "threshold_X", "bat_prior", "bowl_prior", "venue_par",
        "x_minus_par", "x_minus_bat", "x_minus_bowl", "innings",
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
    booster.save_model(str(model_dir / f"{phase_label}_gbm.lgb"))
    meta = {
        "features": features,
        "target_balls": target_balls,
        "x_range": [x_min, x_max, x_step],
        "edge_threshold": 0.05,
        "implied_min": 0.10,
        "implied_max": 0.90,
        "leagues": LEAGUES,
        "default_par": default_par,
        "trained_rows": int(len(x)),
        "best_iter": int(booster.best_iteration),
    }
    (model_dir / f"{phase_label}_meta.json").write_text(json.dumps(meta, indent=2))
    (model_dir / f"{phase_label}_bat_prior.json").write_text(json.dumps(bat_pp))
    (model_dir / f"{phase_label}_bowl_prior.json").write_text(json.dumps(bowl_pp))
    (model_dir / f"{phase_label}_venue_par.json").write_text(json.dumps(ven_par))
    logger.info("  saved model %s_gbm.lgb (best_iter=%d)", phase_label, booster.best_iteration)


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
        train_one_phase(phase_label, target_balls, x_min, x_max, x_step,
                        balls, league_by_match, model_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

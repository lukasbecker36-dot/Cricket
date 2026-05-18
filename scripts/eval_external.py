"""Train on IPL, test on another T20 league. Pure cross-domain generalization test.

Reuses the same feature pipeline. Player stats for the external league are
computed from that league's own historical data (leakage-free per season),
so the model is asked to map league-agnostic numeric features to a win
probability that has never seen this league's player pool.

Usage:
    python -m scripts.eval_external --league bbl
    python -m scripts.eval_external --league bbl --test-seasons 2022 2023 2024
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.features.engineering import FEATURE_COLUMNS
from src.ingestion.storage import read_balls
from src.logging_setup import configure_logging
from src.model.calibration import fit_isotonic
from src.model.train import train_lightgbm
from src.validation.metrics import evaluate
from src.validation.walk_forward import build_dataset

logger = logging.getLogger(__name__)


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--league", required=True, help="league code (bbl, psl, cpl, ntb, ...)")
    parser.add_argument(
        "--test-seasons", type=int, nargs="+", default=None,
        help="defaults to the league's three most recent complete seasons",
    )
    parser.add_argument(
        "--ipl-train-through", type=int, default=2022,
        help="IPL seasons up to and including this used for training (with the next two as calibration)",
    )
    args = parser.parse_args()

    cfg = Config.load(args.config)

    # --- IPL training data ---
    ipl_balls = read_balls(cfg.data.processed_dir)
    if ipl_balls.empty:
        logger.error("no IPL parquet; run scripts.ingest first")
        return 1
    ipl_matches = pd.read_parquet(cfg.data.processed_dir / "matches.parquet")

    fit_seasons = [s for s in sorted(ipl_balls["season"].unique()) if s <= args.ipl_train_through]
    fit_seasons = fit_seasons[cfg.validation.warmup_seasons_to_skip:]
    cal_seasons = [args.ipl_train_through + 1, args.ipl_train_through + 2]

    ipl_pool = sorted(set(fit_seasons) | set(cal_seasons))
    ipl_balls_pool = ipl_balls[ipl_balls["season"].isin(ipl_pool)]
    ipl_matches_pool = ipl_matches[ipl_matches["season"].isin(ipl_pool)]
    logger.info(
        "IPL training: fit %s, calibrate %s",
        fit_seasons, cal_seasons,
    )

    # Build IPL feature dataset (use up_to_season just above latest cal_season so
    # all rows get full prior history). build_dataset is leak-free per-row-season.
    ipl_dataset = build_dataset(
        ipl_balls_pool, ipl_matches_pool,
        up_to_season_for_stats=max(ipl_pool) + 1,
        min_balls_into_chase=cfg.validation.min_balls_into_chase,
    )
    fit_df = ipl_dataset[ipl_dataset["season"].isin(fit_seasons)]
    cal_df = ipl_dataset[ipl_dataset["season"].isin(cal_seasons)]
    cut = int(len(fit_df) * 0.9)
    train_part = fit_df.iloc[:cut]
    early_stop_part = fit_df.iloc[cut:]

    logger.info(
        "IPL fit rows=%d, early-stop rows=%d, calibration rows=%d",
        len(train_part), len(early_stop_part), len(cal_df),
    )
    model = train_lightgbm(train_part, early_stop_part, cfg.model)
    raw_cal = model.booster.predict(cal_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
    calibrator = fit_isotonic(np.asarray(raw_cal), cal_df["label"].to_numpy())

    # --- External league test data ---
    external_dir = Path(f"data/processed/{args.league}")
    external_balls = read_balls(external_dir)
    if external_balls.empty:
        logger.error("no parquet at %s; run scripts.ingest --league %s first", external_dir, args.league)
        return 1
    external_matches = pd.read_parquet(external_dir / "matches.parquet")

    available_seasons = sorted(external_balls["season"].unique())
    test_seasons = args.test_seasons or available_seasons[-3:]
    test_seasons = [s for s in test_seasons if s in available_seasons]
    logger.info("%s test seasons: %s", args.league.upper(), test_seasons)

    # Build external dataset. Stats are per-row-season from external history only,
    # so the model is forced to use the structural features for unknown players.
    external_dataset = build_dataset(
        external_balls, external_matches,
        up_to_season_for_stats=max(test_seasons) + 1,
        min_balls_into_chase=cfg.validation.min_balls_into_chase,
    )
    test_df = external_dataset[external_dataset["season"].isin(test_seasons)]
    if test_df.empty:
        logger.error("no test rows generated for %s seasons %s", args.league, test_seasons)
        return 1

    x_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    raw_test = model.booster.predict(x_test)
    p_test = calibrator.predict(np.asarray(raw_test))
    y_test = test_df["label"].to_numpy()

    overall = evaluate(p_test, y_test)
    logger.info(
        "=== %s OVERALL: n=%d logloss=%.4f brier=%.4f acc@.5=%.3f ECE=%.4f",
        args.league.upper(), overall.n, overall.log_loss, overall.brier,
        overall.accuracy_at_50, overall.calibration.ece,
    )
    for season, group in test_df.groupby("season"):
        idx = group.index
        local_idx = test_df.index.get_indexer(idx)
        p = p_test[local_idx]
        y = y_test[local_idx]
        m = evaluate(p, y)
        logger.info(
            "  season=%d n=%d logloss=%.4f brier=%.4f acc@.5=%.3f ECE=%.4f",
            int(season), m.n, m.log_loss, m.brier, m.accuracy_at_50, m.calibration.ece,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

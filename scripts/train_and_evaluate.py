"""End-to-end: load parquet, walk-forward train+evaluate, run backtest with costs.

Usage:
    python -m scripts.train_and_evaluate [--config configs/default.yaml]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.ingestion.storage import read_balls
from src.logging_setup import configure_logging
from src.validation.backtest import run_backtest
from src.validation.walk_forward import walk_forward

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    configure_logging()
    cfg = Config.load(args.config)

    balls = read_balls(cfg.data.processed_dir)
    if balls.empty:
        logger.error("no parquet data at %s; run scripts.ingest first", cfg.data.processed_dir)
        return 1
    matches_path = cfg.data.processed_dir / "matches.parquet"
    if not matches_path.exists():
        logger.error("no matches.parquet; run scripts.ingest first")
        return 1
    matches = pd.read_parquet(matches_path)

    all_predictions: list[pd.DataFrame] = []
    for fold in walk_forward(
        balls=balls,
        matches=matches,
        train_seasons=cfg.validation.train_seasons,
        test_seasons=cfg.validation.test_seasons,
        cfg=cfg.model,
        min_balls_into_chase=cfg.validation.min_balls_into_chase,
    ):
        m = fold.metrics
        logger.info(
            "season=%d n=%d logloss=%.4f brier=%.4f acc@.5=%.3f ECE=%.4f",
            fold.test_season, m.n, m.log_loss, m.brier, m.accuracy_at_50, m.calibration.ece,
        )
        all_predictions.append(fold.predictions)
        out_path = Path("models") / f"booster_{fold.test_season}.lgb"
        fold.model.save(out_path)

    if not all_predictions:
        logger.error("no folds completed")
        return 1
    combined = pd.concat(all_predictions, ignore_index=True)

    rng = np.random.default_rng(cfg.model.seed)
    bt = run_backtest(combined, cfg.backtest, rng=rng)
    logger.info(
        "BACKTEST (SIMULATED MARKET — see backtest.py): n_trades=%d pnl=%.2f roi=%.3f maxDD=%.2f",
        bt.n_trades, bt.total_pnl, bt.roi, bt.max_drawdown,
    )
    logger.warning(
        "ROI above uses a placeholder market. Replace simulate_market with real Betfair "
        "tick data before drawing any conclusions."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

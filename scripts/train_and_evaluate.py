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
from src.ingestion.market_alignment import build_market_lookup
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
        warmup_seasons_to_skip=cfg.validation.warmup_seasons_to_skip,
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

    # Try to load real Betfair prices. If the join file isn't there, fall back to
    # the simulated market with a loud warning so we never silently use fake data.
    betfair_join_path = cfg.data.processed_dir / "betfair_join.parquet"
    betfair_dir = cfg.data.processed_dir / "betfair"
    use_real_market = betfair_join_path.exists() and betfair_dir.exists()

    if use_real_market:
        join_df = pd.read_parquet(betfair_join_path)
        combined = build_market_lookup(combined, balls, betfair_dir, join_df)
        n_with_market = combined["market_prob"].notna().sum()
        logger.info(
            "real Betfair coverage: %d / %d rows (%.1f%%)",
            n_with_market, len(combined), 100.0 * n_with_market / len(combined),
        )
        # Backtest on the subset where we actually have prices
        bt_input = combined.dropna(subset=["market_prob"]).copy()
        market_implied = bt_input["market_prob"].astype(float)
        bt = run_backtest(bt_input, cfg.backtest, market_implied=market_implied.reset_index(drop=True))
        logger.info(
            "BACKTEST (REAL BETFAIR LTP, BASIC plan): n_trades=%d pnl=%.2f roi=%.3f maxDD=%.2f",
            bt.n_trades, bt.total_pnl, bt.roi, bt.max_drawdown,
        )
        logger.warning(
            "ROI is UPPER BOUND. Ball-index -> wall-time alignment is linear and "
            "can put the model's decision time minutes ahead of the real ball, so "
            "the market price used at decision T may reflect balls the model hadn't "
            "seen yet. Tighter timing anchoring is the next required step before "
            "trusting these numbers."
        )
        # Per-season ROI break-down so we can see where the edge (if any) lives
        bt_input = bt_input.reset_index(drop=True)
        trades = bt.trades.merge(
            bt_input[["match_id", "ball_index", "season"]],
            on=["match_id", "ball_index"], how="left",
        )
        for season, group in trades.groupby("season"):
            stake_total = 100.0 * len(group)
            logger.info(
                "  season=%d n=%d pnl=%.2f roi=%.3f",
                int(season), len(group), group["pnl"].sum(),
                group["pnl"].sum() / stake_total if stake_total else 0.0,
            )
    else:
        rng = np.random.default_rng(cfg.model.seed)
        bt = run_backtest(combined, cfg.backtest, rng=rng)
        logger.info(
            "BACKTEST (SIMULATED MARKET): n_trades=%d pnl=%.2f roi=%.3f maxDD=%.2f",
            bt.n_trades, bt.total_pnl, bt.roi, bt.max_drawdown,
        )
        logger.warning(
            "ROI above uses a placeholder market. Run scripts.ingest_betfair first "
            "for real prices."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

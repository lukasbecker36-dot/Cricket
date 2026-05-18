"""Slice walk-forward errors for one test season to find where the model is failing.

Usage: python -m scripts.diagnose_season --season 2023
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from src.config import Config
from src.ingestion.storage import read_balls
from src.logging_setup import configure_logging
from src.validation.metrics import evaluate
from src.validation.walk_forward import walk_forward

logger = logging.getLogger(__name__)


def slice_metrics(df: pd.DataFrame, by: str) -> pd.DataFrame:
    out = []
    for key, group in df.groupby(by):
        if len(group) < 50:
            continue
        m = evaluate(group["p"].to_numpy(), group["y"].to_numpy())
        out.append(
            {
                by: key,
                "n": m.n,
                "logloss": round(m.log_loss, 4),
                "brier": round(m.brier, 4),
                "acc@.5": round(m.accuracy_at_50, 3),
                "ece": round(m.calibration.ece, 4),
                "base_rate": round(group["y"].mean(), 3),
            }
        )
    return pd.DataFrame(out).sort_values("logloss", ascending=False)


def add_slices(preds: pd.DataFrame, balls: pd.DataFrame) -> pd.DataFrame:
    """Join venue + chase context onto each prediction row."""
    inn2 = balls[balls["innings"] == 2].copy()
    # one row per (match_id, legal-ball-cumulative-index)
    inn2 = inn2.sort_values(["match_id", "over", "ball", "is_legal_delivery"], ascending=[True, True, True, False])
    inn2["legal_so_far"] = inn2.groupby("match_id")["is_legal_delivery"].cumsum().astype(int) - inn2["is_legal_delivery"].astype(int)
    inn2["ball_index_from_chase_start"] = inn2.groupby("match_id").cumcount()
    keep = inn2[["match_id", "ball_index_from_chase_start", "venue", "over", "target"]].rename(
        columns={"ball_index_from_chase_start": "ball_index"}
    )
    merged = preds.merge(keep, on=["match_id", "ball_index"], how="left")
    merged["phase"] = pd.cut(
        merged["over"], bins=[-1, 5, 14, 19, 99], labels=["pp", "mid", "death", "extra"]
    )
    merged["target_band"] = pd.cut(
        merged["target"], bins=[0, 140, 160, 180, 200, 400], labels=["<=140", "141-160", "161-180", "181-200", "200+"]
    )
    return merged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    configure_logging()
    cfg = Config.load(args.config)
    balls = read_balls(cfg.data.processed_dir)
    matches = pd.read_parquet(cfg.data.processed_dir / "matches.parquet")

    train_seasons = [s for s in cfg.validation.train_seasons + cfg.validation.test_seasons if s < args.season]
    folds = list(
        walk_forward(
            balls=balls,
            matches=matches,
            train_seasons=train_seasons,
            test_seasons=[args.season],
            cfg=cfg.model,
            min_balls_into_chase=cfg.validation.min_balls_into_chase,
        )
    )
    if not folds:
        logger.error("no fold produced")
        return 1
    fold = folds[0]
    m = fold.metrics
    print(f"\nSeason {args.season}: n={m.n} logloss={m.log_loss:.4f} brier={m.brier:.4f} "
          f"acc@.5={m.accuracy_at_50:.3f} ECE={m.calibration.ece:.4f}\n")

    enriched = add_slices(fold.predictions, balls)

    print("--- BY VENUE (worst log loss first) ---")
    print(slice_metrics(enriched, "venue").to_string(index=False))

    print("\n--- BY CHASE PHASE ---")
    print(slice_metrics(enriched, "phase").to_string(index=False))

    print("\n--- BY TARGET BAND ---")
    print(slice_metrics(enriched, "target_band").to_string(index=False))

    print("\n--- HIGH-CONFIDENCE WRONG (predicted > .8 but lost, or < .2 but won) ---")
    wrong = enriched[((enriched["p"] > 0.8) & (enriched["y"] == 0)) | ((enriched["p"] < 0.2) & (enriched["y"] == 1))]
    by_match = wrong.groupby("match_id").size().sort_values(ascending=False).head(10)
    print(by_match.to_string())

    print("\n--- CALIBRATION BY BIN ---")
    cal = m.calibration
    for i, (mp, fp, c) in enumerate(zip(cal.bin_mean_pred, cal.bin_frac_pos, cal.bin_counts)):
        if c == 0:
            continue
        print(f"  bin {i}: mean_pred={mp:.3f}  actual={fp:.3f}  n={c}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

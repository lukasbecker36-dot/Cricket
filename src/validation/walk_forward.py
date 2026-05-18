"""Walk-forward validation. Train on seasons <= N, test on N+1, roll forward.

Never random splits — temporal leakage is the most common failure mode here.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import ModelConfig
from ..features.engineering import (
    FEATURE_COLUMNS,
    compute_venue_pars,
    features_from_state,
    replay_chase,
)
from ..features.player_quality import compute_player_stats
from ..ingestion.schema import MatchMeta
from ..model.calibration import fit_isotonic
from ..model.train import TrainedModel, train_lightgbm
from .metrics import MetricsReport, evaluate

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    test_season: int
    metrics: MetricsReport
    predictions: pd.DataFrame  # match_id, season, ball_index, p, y
    model: TrainedModel


def build_dataset(
    balls: pd.DataFrame,
    matches: pd.DataFrame,
    up_to_season_for_stats: int,
    min_balls_into_chase: int,
) -> pd.DataFrame:
    """Generate (features, label) rows for every legal point in every chase."""
    stats = compute_player_stats(balls, up_to_season=up_to_season_for_stats)
    venue_pars = compute_venue_pars(balls, up_to_season=up_to_season_for_stats)

    match_labels = matches.set_index("match_id")
    rows: list[dict] = []

    for match_id, group in balls.groupby("match_id"):
        if match_id not in match_labels.index:
            continue
        meta = match_labels.loc[match_id]
        winner = meta.get("winner")
        chasing_team = group.loc[group["innings"] == 2, "batting_team"]
        if chasing_team.empty or pd.isna(winner):
            continue
        label = int(winner == chasing_team.iloc[0])

        for i, state in enumerate(replay_chase(group, label=label)):
            if i < min_balls_into_chase:
                continue
            feats = features_from_state(state, stats, venue_pars)
            feats.update(
                match_id=match_id, season=state.season, ball_index=i, label=state.label
            )
            rows.append(feats)

    if not rows:
        return pd.DataFrame(columns=FEATURE_COLUMNS + ["match_id", "season", "ball_index", "label"])
    return pd.DataFrame(rows)


def walk_forward(
    balls: pd.DataFrame,
    matches: pd.DataFrame,
    train_seasons: list[int],
    test_seasons: list[int],
    cfg: ModelConfig,
    min_balls_into_chase: int = 6,
) -> Iterator[FoldResult]:
    """Yield one FoldResult per test season. Stats computed using only prior data."""
    train_pool = sorted(set(train_seasons))
    for test_season in sorted(test_seasons):
        train_mask = balls["season"].isin(train_pool)
        train_balls = balls[train_mask]
        train_matches = matches[matches["season"].isin(train_pool)]

        test_balls = balls[balls["season"] == test_season]
        test_matches = matches[matches["season"] == test_season]
        if test_balls.empty:
            logger.warning("no data for test season %d; skipping", test_season)
            continue

        train_df = build_dataset(train_balls, train_matches, test_season, min_balls_into_chase)
        test_df = build_dataset(
            pd.concat([train_balls, test_balls], ignore_index=True),
            pd.concat([train_matches, test_matches], ignore_index=True),
            test_season,
            min_balls_into_chase,
        )
        test_df = test_df[test_df["season"] == test_season]

        if train_df.empty or test_df.empty:
            logger.warning("empty dataset for season %d", test_season)
            continue

        # carve a tail of training data as a validation set for early stopping + calibration
        cut = int(len(train_df) * 0.9)
        train_part = train_df.iloc[:cut]
        valid_part = train_df.iloc[cut:]

        # recency weighting: weight = decay^(test_season - 1 - season)
        # so the most recent training season has weight 1.0 and older seasons decay.
        ref_season = test_season - 1
        weights = np.power(
            cfg.recency_decay,
            np.maximum(ref_season - train_part["season"].to_numpy(), 0),
        ).astype(np.float32)

        model = train_lightgbm(train_part, valid_part, cfg, train_weights=weights)
        raw_valid = model.booster.predict(valid_part[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
        calibrator = fit_isotonic(np.asarray(raw_valid), valid_part["label"].to_numpy())

        raw_test = model.booster.predict(test_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
        p_test = calibrator.predict(np.asarray(raw_test))
        y_test = test_df["label"].to_numpy()

        preds = test_df[["match_id", "season", "ball_index"]].copy()
        preds["p"] = p_test
        preds["y"] = y_test

        train_pool.append(test_season)
        yield FoldResult(
            test_season=test_season,
            metrics=evaluate(p_test, y_test),
            predictions=preds,
            model=model,
        )

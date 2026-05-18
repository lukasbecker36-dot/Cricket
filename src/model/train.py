"""LightGBM training for the v1 baseline. Training path; inference lives elsewhere."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..config import ModelConfig
from ..features.engineering import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


@dataclass
class TrainedModel:
    booster: lgb.Booster
    feature_columns: list[str]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path))

    @classmethod
    def load(cls, path: Path) -> "TrainedModel":
        booster = lgb.Booster(model_file=str(path))
        return cls(booster=booster, feature_columns=FEATURE_COLUMNS)


def train_lightgbm(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    cfg: ModelConfig,
    train_weights: np.ndarray | None = None,
) -> TrainedModel:
    """Train a binary GBM on chase win probability."""
    if "label" not in train_df.columns:
        raise ValueError("train_df missing 'label'")

    x_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train = train_df["label"].to_numpy(dtype=np.int8)
    x_valid = valid_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_valid = valid_df["label"].to_numpy(dtype=np.int8)

    dtrain = lgb.Dataset(
        x_train, label=y_train, weight=train_weights, feature_name=FEATURE_COLUMNS
    )
    dvalid = lgb.Dataset(x_valid, label=y_valid, feature_name=FEATURE_COLUMNS, reference=dtrain)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "min_data_in_leaf": cfg.min_data_in_leaf,
        "feature_pre_filter": False,
        "verbose": -1,
        "seed": cfg.seed,
        "deterministic": True,
    }

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=cfg.n_estimators,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
    )
    logger.info("trained GBM: best_iter=%d", booster.best_iteration)
    return TrainedModel(booster=booster, feature_columns=FEATURE_COLUMNS)

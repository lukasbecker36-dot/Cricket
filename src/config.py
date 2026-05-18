"""Project configuration. Every model run must be reproducible from a config snapshot."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    cache_dir: Path = Path("data/cache")
    cricsheet_url: str = "https://cricsheet.org/downloads/ipl_json.zip"


class ModelConfig(BaseModel):
    seed: int = 17
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_data_in_leaf: int = 200
    n_estimators: int = 1000
    early_stopping_rounds: int = 50
    # Recency weighting: each older season's sample weight decays by this factor per year.
    # 1.0 = uniform; 0.92 = the most recent training season carries ~2.4x the weight of one
    # 10 years older. Justified by IPL regime drift (rules, par scores).
    recency_decay: float = 0.92


class ValidationConfig(BaseModel):
    train_seasons: list[int] = Field(default_factory=lambda: list(range(2008, 2021)))
    test_seasons: list[int] = Field(default_factory=lambda: [2021, 2022, 2023, 2024])
    min_balls_into_chase: int = 6
    # Drop the first N seasons from training: stats for those rows are computed from
    # too few prior seasons (or none), creating training-feature distribution heterogeneity
    # vs test rows that always see ~14 prior seasons. Costs us a few hundred matches.
    warmup_seasons_to_skip: int = 3


class BacktestConfig(BaseModel):
    commission: float = 0.05  # 5% Betfair commission, conservative
    slippage_ticks: int = 1
    decision_lag_seconds: int = 5
    edge_threshold: float = 0.05  # only trade when |model - market| > threshold
    min_liquidity_gbp: float = 100.0


class Config(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        if path is None:
            return cls()
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

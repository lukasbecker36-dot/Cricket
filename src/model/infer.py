"""Inference path. Loads trained booster + calibrator and predicts on a feature row."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..features.engineering import FEATURE_COLUMNS
from .calibration import Calibrator
from .train import TrainedModel


class WinProbabilityModel:
    def __init__(self, model: TrainedModel, calibrator: Calibrator | None = None):
        self.model = model
        self.calibrator = calibrator

    @classmethod
    def load(cls, model_path: Path, calibrator_path: Path | None = None) -> "WinProbabilityModel":
        model = TrainedModel.load(model_path)
        calibrator = Calibrator.load(calibrator_path) if calibrator_path else None
        return cls(model, calibrator)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        missing = [c for c in FEATURE_COLUMNS if c not in features.columns]
        if missing:
            raise ValueError(f"features missing columns: {missing}")
        x = features[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        raw = self.model.booster.predict(x)
        raw = np.asarray(raw, dtype=np.float64)
        if (raw < 0).any() or (raw > 1).any():
            raise ValueError("model produced probability outside [0,1] — fail loud")
        if self.calibrator is not None:
            return self.calibrator.predict(raw)
        return np.clip(raw, 1e-4, 1 - 1e-4)

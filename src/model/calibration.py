"""Isotonic calibration. Apply after GBM training on held-out data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass
class Calibrator:
    iso: IsotonicRegression

    def predict(self, p: np.ndarray) -> np.ndarray:
        out = self.iso.predict(p)
        return np.clip(out, 1e-4, 1 - 1e-4)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.iso, path)

    @classmethod
    def load(cls, path: Path) -> "Calibrator":
        return cls(iso=joblib.load(path))


def fit_isotonic(p: np.ndarray, y: np.ndarray) -> Calibrator:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p, y)
    return Calibrator(iso=iso)

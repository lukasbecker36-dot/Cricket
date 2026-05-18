"""Required metrics for every model version: log loss, Brier, ECE, accuracy."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss


@dataclass
class CalibrationReport:
    bin_edges: np.ndarray
    bin_mean_pred: np.ndarray
    bin_frac_pos: np.ndarray
    bin_counts: np.ndarray
    ece: float  # expected calibration error


def expected_calibration_error(
    p: np.ndarray, y: np.ndarray, n_bins: int = 10
) -> CalibrationReport:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    counts = np.zeros(n_bins, dtype=np.int64)
    mean_pred = np.zeros(n_bins)
    frac_pos = np.zeros(n_bins)
    for i in range(n_bins):
        mask = bin_idx == i
        counts[i] = mask.sum()
        if counts[i] == 0:
            continue
        mean_pred[i] = p[mask].mean()
        frac_pos[i] = y[mask].mean()
    weights = counts / max(counts.sum(), 1)
    ece = float(np.sum(weights * np.abs(mean_pred - frac_pos)))
    return CalibrationReport(edges, mean_pred, frac_pos, counts, ece)


@dataclass
class MetricsReport:
    n: int
    log_loss: float
    brier: float
    accuracy_at_50: float
    accuracy_at_70: float
    calibration: CalibrationReport


def evaluate(p: np.ndarray, y: np.ndarray) -> MetricsReport:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    ll = float(log_loss(y, p, labels=[0, 1]))
    br = float(brier_score_loss(y, p))
    acc50 = float(((p >= 0.5) == y.astype(bool)).mean())
    # accuracy at high-confidence predictions only
    high_conf = (p >= 0.7) | (p <= 0.3)
    if high_conf.any():
        pred = (p[high_conf] >= 0.5).astype(int)
        acc70 = float((pred == y[high_conf]).mean())
    else:
        acc70 = float("nan")
    cal = expected_calibration_error(p, y)
    return MetricsReport(
        n=len(y),
        log_loss=ll,
        brier=br,
        accuracy_at_50=acc50,
        accuracy_at_70=acc70,
        calibration=cal,
    )

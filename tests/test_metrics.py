import numpy as np

from src.validation.metrics import evaluate, expected_calibration_error


def test_perfect_predictions_have_zero_logloss_and_brier():
    p = np.array([0.999, 0.001, 0.999, 0.001])
    y = np.array([1, 0, 1, 0])
    r = evaluate(p, y)
    assert r.log_loss < 0.01
    assert r.brier < 0.01
    assert r.accuracy_at_50 == 1.0


def test_calibrated_uniform_predictions_have_low_ece():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=10_000)
    y = (rng.uniform(0, 1, size=10_000) < p).astype(int)
    rep = expected_calibration_error(p, y, n_bins=10)
    assert rep.ece < 0.05


def test_constant_predictions_get_accuracy_equal_to_base_rate():
    p = np.full(100, 0.7)
    y = np.zeros(100, dtype=int)
    y[:30] = 1
    r = evaluate(p, y)
    # everything predicted positive, base rate is 0.3 -> accuracy is 0.3
    assert abs(r.accuracy_at_50 - 0.3) < 1e-9

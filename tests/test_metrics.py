import numpy as np
import pandas as pd

from uncertainty_benchmark.metrics.calibration import (
    compute_ece,
    confidence_from_uncertainty,
)
from uncertainty_benchmark.metrics.ranking import (
    auprc_error_detection,
    safe_roc_auc,
)
from uncertainty_benchmark.metrics.risk_coverage import compute_rc_metrics
from uncertainty_benchmark.metrics.suite import (
    METRIC_NAMES,
    compute_metrics_per_method_with_timing,
    compute_single_method_metrics,
    metrics_to_long,
)


def test_confidence_from_uncertainty():
    uncertainty = np.array([0.0, 0.5, 1.0])
    u_norm, conf = confidence_from_uncertainty(uncertainty)

    assert np.allclose(u_norm, [0.0, 0.5, 1.0])
    assert np.allclose(conf, [1.0, 0.5, 0.0])


def test_confidence_from_constant_uncertainty():
    uncertainty = np.array([2.0, 2.0, 2.0])
    u_norm, conf = confidence_from_uncertainty(uncertainty)

    assert np.allclose(u_norm, [0.0, 0.0, 0.0])
    assert np.allclose(conf, [1.0, 1.0, 1.0])


def test_compute_ece_returns_float():
    confidence = np.array([0.9, 0.8, 0.3, 0.2])
    correct = np.array([1, 1, 0, 0])

    ece = compute_ece(confidence, correct, n_bins=2)
    assert isinstance(ece, float)
    assert ece >= 0.0


def test_ranking_metrics():
    correct = np.array([1, 1, 0, 0])
    confidence = np.array([0.9, 0.8, 0.3, 0.2])
    uncertainty = 1.0 - confidence

    roc = safe_roc_auc(correct, confidence)
    auprc = auprc_error_detection(correct, uncertainty)

    assert roc == 1.0
    assert auprc >= 0.0


def test_risk_coverage_metrics():
    uncertainty = np.array([0.1, 0.2, 0.8, 0.9])
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])

    out = compute_rc_metrics(uncertainty, y_true, y_pred)

    assert "E-AUoptRC" in out
    assert "TI" in out
    assert "TI@95" in out
    assert "Optimal Coverage" in out


def test_single_method_metrics():
    uncertainty = np.array([0.1, 0.2, 0.8, 0.9])
    correct = np.array([1, 1, 0, 0])
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])

    out = compute_single_method_metrics(
        uncertainty=uncertainty,
        y_true_bin=correct,
        y_true_idx=y_true,
        y_pred_idx=y_pred,
        bins=2,
    )

    for metric in METRIC_NAMES:
        assert metric in out


def test_metric_suite_with_timing():
    df = pd.DataFrame(
        {
            "correct": [True, True, False, False],
            "y_true_idx": [0, 1, 0, 1],
            "y_pred_idx": [0, 1, 1, 0],
            "SR": [0.1, 0.2, 0.8, 0.9],
            "ENT": [0.2, 0.3, 0.7, 0.95],
        }
    )

    metrics_df, method_times = compute_metrics_per_method_with_timing(
        df,
        methods=["SR", "ENT"],
        bins=2,
    )

    assert set(metrics_df.columns) == {"SR", "ENT"}
    assert set(method_times.keys()) == {"SR", "ENT"}

    long_df = metrics_to_long(metrics_df, fold=0)
    assert {"fold", "metric", "method", "value"}.issubset(long_df.columns)

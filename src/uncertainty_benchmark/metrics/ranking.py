"""Backward-compatible ranking and discrimination metric wrappers.

Historically, this repository used ``ranking.py`` for two slightly different
metrics:

1. ROC-AUC with correctness as the positive class:
       y_true_bin: 1 = correct, 0 = incorrect
       score: confidence

2. AU-PRC for error detection:
       positive class: error
       score: uncertainty

Newer code should use ``uncertainty_benchmark.metrics.discrimination`` for
explicit error-discrimination metrics.
"""

from __future__ import annotations

from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

from uncertainty_benchmark.metrics.discrimination import (
    compute_discrimination_for_methods,
    compute_discrimination_from_errors,
    compute_discrimination_metrics,
)
from uncertainty_benchmark.metrics.utils import (
    ArrayLike,
    finite_mask,
    has_two_classes,
    to_numpy_1d,
)


def safe_roc_auc(
    y_true_bin: ArrayLike,
    confidence: ArrayLike,
) -> float:
    """ROC-AUC with correctness as the positive class.

    Parameters
    ----------
    y_true_bin:
        Binary labels where 1 = correct and 0 = incorrect.
    confidence:
        Confidence scores where larger means more confident.

    Returns
    -------
    float
        ROC-AUC value, or NaN if undefined.
    """
    y = to_numpy_1d(y_true_bin, dtype=float)
    scores = to_numpy_1d(confidence, dtype=float)

    mask = finite_mask(y, scores)
    y = y[mask].astype(int)
    scores = scores[mask]

    if y.size == 0 or not has_two_classes(y):
        return float("nan")

    try:
        return float(roc_auc_score(y, scores))
    except Exception:
        return float("nan")


def auprc_error_detection(
    y_true_bin: ArrayLike,
    uncertainty_norm: ArrayLike,
) -> float:
    """AU-PRC for error detection.

    Parameters
    ----------
    y_true_bin:
        Binary labels where 1 = correct and 0 = incorrect.
    uncertainty_norm:
        Normalised uncertainty scores where larger means more uncertain.

    Notes
    -----
    Error is the positive class:

        1 = incorrect prediction
        0 = correct prediction
    """
    y = to_numpy_1d(y_true_bin, dtype=float)
    scores = to_numpy_1d(uncertainty_norm, dtype=float)

    mask = finite_mask(y, scores)
    y = y[mask].astype(int)
    scores = scores[mask]

    if y.size == 0:
        return float("nan")

    error = (y == 0).astype(int)

    if not has_two_classes(error):
        return float("nan")

    try:
        precision, recall, _ = precision_recall_curve(error, scores)
        return float(auc(recall, precision))
    except Exception:
        return float("nan")


__all__ = [
    "safe_roc_auc",
    "auprc_error_detection",
    "compute_discrimination_from_errors",
    "compute_discrimination_metrics",
    "compute_discrimination_for_methods",
]
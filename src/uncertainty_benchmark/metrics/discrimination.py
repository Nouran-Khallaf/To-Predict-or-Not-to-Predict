"""Discrimination metrics for uncertainty estimation.

These metrics evaluate whether an uncertainty score can distinguish correct
predictions from incorrect predictions.

Convention
----------
The uncertainty score should follow this direction:

    larger score = more uncertain = more likely to be wrong

The positive class for discrimination is therefore an incorrect prediction.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from uncertainty_benchmark.metrics.utils import (
    ArrayLike,
    finite_mask,
    has_two_classes,
    prediction_error_labels,
    to_numpy_1d,
)


# ---------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------


def roc_auc_uncertainty(
    y_error: ArrayLike,
    uncertainty_scores: ArrayLike,
) -> float:
    """Compute ROC-AUC for uncertainty-based error detection.

    Parameters
    ----------
    y_error:
        Binary labels where 1 = incorrect prediction and 0 = correct prediction.
    uncertainty_scores:
        Uncertainty scores where larger means more uncertain.

    Returns
    -------
    float
        ROC-AUC value, or NaN when undefined.
    """
    y = to_numpy_1d(y_error, dtype=float)
    scores = to_numpy_1d(uncertainty_scores, dtype=float)

    mask = finite_mask(y, scores)
    y = y[mask].astype(int)
    scores = scores[mask]

    if y.size == 0 or not has_two_classes(y):
        return float("nan")

    try:
        return float(roc_auc_score(y, scores))
    except Exception:
        return float("nan")


def auprc_uncertainty(
    y_error: ArrayLike,
    uncertainty_scores: ArrayLike,
) -> float:
    """Compute AU-PRC / average precision for uncertainty-based error detection.

    Parameters
    ----------
    y_error:
        Binary labels where 1 = incorrect prediction and 0 = correct prediction.
    uncertainty_scores:
        Uncertainty scores where larger means more uncertain.

    Returns
    -------
    float
        AU-PRC value, or NaN when undefined.
    """
    y = to_numpy_1d(y_error, dtype=float)
    scores = to_numpy_1d(uncertainty_scores, dtype=float)

    mask = finite_mask(y, scores)
    y = y[mask].astype(int)
    scores = scores[mask]

    if y.size == 0 or not has_two_classes(y):
        return float("nan")

    try:
        return float(average_precision_score(y, scores))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------
# Combined APIs
# ---------------------------------------------------------------------


def compute_discrimination_from_errors(
    y_error: ArrayLike,
    uncertainty_scores: ArrayLike,
    prefix: str = "",
) -> Dict[str, float]:
    """Compute discrimination metrics from precomputed error labels.

    Parameters
    ----------
    y_error:
        Binary labels where 1 = incorrect and 0 = correct.
    uncertainty_scores:
        Uncertainty scores where larger means more uncertain.
    prefix:
        Optional prefix added to metric names.

    Returns
    -------
    dict
        Dictionary containing ROC-AUC and AU-PRC.
    """
    return {
        f"{prefix}ROC-AUC": roc_auc_uncertainty(y_error, uncertainty_scores),
        f"{prefix}AU-PRC": auprc_uncertainty(y_error, uncertainty_scores),
    }


def compute_discrimination_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    uncertainty_scores: ArrayLike,
    prefix: str = "",
) -> Dict[str, float]:
    """Compute uncertainty discrimination metrics from true and predicted labels.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    y_pred:
        Predicted labels.
    uncertainty_scores:
        Uncertainty scores where larger means more uncertain.
    prefix:
        Optional prefix added to metric names.

    Returns
    -------
    dict
        Dictionary containing ROC-AUC and AU-PRC.
    """
    y_error = prediction_error_labels(y_true, y_pred)

    return compute_discrimination_from_errors(
        y_error=y_error,
        uncertainty_scores=uncertainty_scores,
        prefix=prefix,
    )


def compute_discrimination_for_methods(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    method_to_scores: Dict[str, ArrayLike],
) -> Dict[str, Dict[str, float]]:
    """Compute discrimination metrics for multiple uncertainty methods.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    y_pred:
        Predicted labels.
    method_to_scores:
        Mapping from method name to uncertainty score vector.

    Returns
    -------
    dict
        Nested dictionary: method -> metric -> value.
    """
    results: Dict[str, Dict[str, float]] = {}

    for method, scores in method_to_scores.items():
        results[method] = compute_discrimination_metrics(
            y_true=y_true,
            y_pred=y_pred,
            uncertainty_scores=scores,
        )

    return results


__all__ = [
    "roc_auc_uncertainty",
    "auprc_uncertainty",
    "compute_discrimination_from_errors",
    "compute_discrimination_metrics",
    "compute_discrimination_for_methods",
]
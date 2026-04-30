"""Backward-compatible wrappers for uncertainty ranking metrics.

Historically, this repository used ``ranking.py`` for ROC-AUC and AU-PRC
error-detection metrics. New code should use:

    uncertainty_benchmark.metrics.discrimination

This module is kept only so older imports, tests, and scripts do not break.
"""

from __future__ import annotations

from typing import Dict

from uncertainty_benchmark.metrics.discrimination import (
    auprc_uncertainty,
    compute_discrimination_for_methods,
    compute_discrimination_from_errors,
    compute_discrimination_metrics,
    roc_auc_uncertainty,
)
from uncertainty_benchmark.metrics.utils import ArrayLike


def safe_roc_auc(
    y_true_bin: ArrayLike,
    scores: ArrayLike,
) -> float:
    """Backward-compatible wrapper for ROC-AUC error detection.

    Parameters
    ----------
    y_true_bin:
        Binary labels where 1 = incorrect prediction and 0 = correct prediction.
    scores:
        Uncertainty scores where larger means more uncertain.

    Returns
    -------
    float
        ROC-AUC value, or NaN if undefined.
    """
    return roc_auc_uncertainty(
        y_error=y_true_bin,
        uncertainty_scores=scores,
    )


def auprc_error_detection(
    y_true_bin: ArrayLike,
    scores: ArrayLike,
) -> float:
    """Backward-compatible wrapper for AU-PRC error detection.

    Parameters
    ----------
    y_true_bin:
        Binary labels where 1 = incorrect prediction and 0 = correct prediction.
    scores:
        Uncertainty scores where larger means more uncertain.

    Returns
    -------
    float
        AU-PRC value, or NaN if undefined.
    """
    return auprc_uncertainty(
        y_error=y_true_bin,
        uncertainty_scores=scores,
    )


def compute_ranking_from_errors(
    y_error: ArrayLike,
    uncertainty_scores: ArrayLike,
    prefix: str = "",
) -> Dict[str, float]:
    """Backward-compatible wrapper for discrimination metrics from errors."""
    return compute_discrimination_from_errors(
        y_error=y_error,
        uncertainty_scores=uncertainty_scores,
        prefix=prefix,
    )


def compute_ranking_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    uncertainty_scores: ArrayLike,
    prefix: str = "",
) -> Dict[str, float]:
    """Backward-compatible wrapper for discrimination metrics."""
    return compute_discrimination_metrics(
        y_true=y_true,
        y_pred=y_pred,
        uncertainty_scores=uncertainty_scores,
        prefix=prefix,
    )


def compute_ranking_for_methods(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    method_to_scores: Dict[str, ArrayLike],
) -> Dict[str, Dict[str, float]]:
    """Backward-compatible wrapper for multiple-method discrimination metrics."""
    return compute_discrimination_for_methods(
        y_true=y_true,
        y_pred=y_pred,
        method_to_scores=method_to_scores,
    )


__all__ = [
    "safe_roc_auc",
    "auprc_error_detection",
    "compute_ranking_from_errors",
    "compute_ranking_metrics",
    "compute_ranking_for_methods",
]
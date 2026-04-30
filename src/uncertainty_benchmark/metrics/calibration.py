"""Calibration metrics for uncertainty/confidence estimates.

Conventions
-----------
Most uncertainty methods in this repository use this direction:

    larger uncertainty = more uncertain

Calibration metrics, however, use confidence scores:

    larger confidence = more likely to be correct

Use :func:`confidence_from_uncertainty` or :func:`uncertainty_to_confidence`
when you need to convert uncertainty scores before computing calibration.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from uncertainty_benchmark.metrics.utils import (
    ArrayLike,
    clip_probabilities,
    finite_mask,
    prediction_correct_labels,
    safe_logit,
    to_numpy_1d,
)


# ---------------------------------------------------------------------
# Uncertainty-to-confidence conversion
# ---------------------------------------------------------------------


def confidence_from_uncertainty(uncertainty: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Normalise uncertainty to [0, 1] and convert it to confidence.

    This function preserves the original public API used by the metric suite:

    - input: uncertainty scores where larger means more uncertain
    - output: ``(normalised_uncertainty, confidence)``
    - conversion: ``confidence = 1 - normalised_uncertainty``

    For constant uncertainty scores, the normalised uncertainty is set to 0 and
    confidence is set to 1. This keeps backward compatibility with the existing
    tests and previous metric outputs.
    """
    raw = to_numpy_1d(uncertainty, dtype=float)

    if raw.size == 0:
        return raw, raw

    uncertainty_norm = np.full_like(raw, np.nan, dtype=float)
    finite = np.isfinite(raw)

    if not np.any(finite):
        return uncertainty_norm, uncertainty_norm.copy()

    values = raw[finite]
    vmin = np.min(values)
    vmax = np.max(values)

    if np.isclose(vmax, vmin):
        uncertainty_norm[finite] = 0.0
    else:
        uncertainty_norm[finite] = (values - vmin) / (vmax - vmin)

    confidence = 1.0 - uncertainty_norm
    confidence = np.clip(confidence, 0.0, 1.0)

    return uncertainty_norm, confidence


def uncertainty_to_confidence(
    uncertainty_scores: ArrayLike,
    method: str = "minmax_inverse",
    eps: float = 1e-6,
) -> np.ndarray:
    """Convert uncertainty scores to confidence scores.

    Parameters
    ----------
    uncertainty_scores:
        Scores where larger means more uncertain.
    method:
        Conversion strategy:

        - ``"minmax_inverse"``: confidence = 1 - minmax(uncertainty)
        - ``"rank_inverse"``: confidence = 1 - percentile_rank(uncertainty)
        - ``"negative"``: confidence = -uncertainty, then min-max scaled

    eps:
        Clipping value for the final finite confidence values.

    Returns
    -------
    np.ndarray
        Confidence scores in ``[eps, 1 - eps]`` where possible.
    """
    uncertainty = to_numpy_1d(uncertainty_scores, dtype=float)
    confidence = np.full_like(uncertainty, np.nan, dtype=float)
    finite = np.isfinite(uncertainty)

    if not np.any(finite):
        return confidence

    values = uncertainty[finite]

    if method == "minmax_inverse":
        lo = np.min(values)
        hi = np.max(values)

        if np.isclose(lo, hi):
            scaled = np.zeros_like(values, dtype=float)
        else:
            scaled = (values - lo) / (hi - lo)

        confidence[finite] = 1.0 - scaled

    elif method == "rank_inverse":
        order = np.argsort(values)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(values), dtype=float)

        if len(values) == 1:
            scaled = np.zeros_like(values, dtype=float)
        else:
            scaled = ranks / (len(values) - 1)

        confidence[finite] = 1.0 - scaled

    elif method == "negative":
        negative_values = -values
        lo = np.min(negative_values)
        hi = np.max(negative_values)

        if np.isclose(lo, hi):
            scaled = np.ones_like(negative_values, dtype=float)
        else:
            scaled = (negative_values - lo) / (hi - lo)

        confidence[finite] = scaled

    else:
        raise ValueError(f"Unknown uncertainty-to-confidence method: {method}")

    confidence[finite] = np.clip(confidence[finite], eps, 1.0 - eps)
    return confidence


# ---------------------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------------------


def expected_calibration_error(
    y_correct: ArrayLike,
    confidence_scores: ArrayLike,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Compute Expected Calibration Error.

    Parameters
    ----------
    y_correct:
        Binary correctness labels where 1 = correct and 0 = incorrect.
    confidence_scores:
        Confidence scores in [0, 1], where larger means more confident.
    n_bins:
        Number of bins.
    strategy:
        ``"uniform"`` for equal-width bins or ``"quantile"`` for approximately
        equal-count bins.
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    y = to_numpy_1d(y_correct, dtype=float)
    confidence = to_numpy_1d(confidence_scores, dtype=float)

    mask = finite_mask(y, confidence)
    y = y[mask]
    confidence = confidence[mask]

    if y.size == 0:
        return float("nan")

    confidence = np.clip(confidence, 0.0, 1.0)

    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    elif strategy == "quantile":
        bin_edges = np.quantile(confidence, np.linspace(0.0, 1.0, n_bins + 1))
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
        bin_edges = np.unique(bin_edges)

        if bin_edges.size < 2:
            return 0.0

    else:
        raise ValueError("strategy must be either 'uniform' or 'quantile'.")

    ece = 0.0
    n = y.size

    for i in range(len(bin_edges) - 1):
        left = bin_edges[i]
        right = bin_edges[i + 1]

        if i == len(bin_edges) - 2:
            in_bin = (confidence >= left) & (confidence <= right)
        else:
            in_bin = (confidence >= left) & (confidence < right)

        count = int(np.sum(in_bin))

        if count == 0:
            continue

        bin_accuracy = float(np.mean(y[in_bin]))
        bin_confidence = float(np.mean(confidence[in_bin]))
        ece += (count / n) * abs(bin_accuracy - bin_confidence)

    return float(ece)


def compute_ece(confidence: ArrayLike, y_true_bin: ArrayLike, n_bins: int = 15) -> float:
    """Backward-compatible wrapper for Expected Calibration Error.

    The existing metric suite calls this as:

    ``compute_ece(confidence, y_true_bin)``

    Newer code may prefer :func:`expected_calibration_error`, whose argument
    order is:

    ``expected_calibration_error(y_correct, confidence_scores)``
    """
    return expected_calibration_error(
        y_correct=y_true_bin,
        confidence_scores=confidence,
        n_bins=n_bins,
        strategy="uniform",
    )


def calibration_bins(
    y_correct: ArrayLike,
    confidence_scores: ArrayLike,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> np.ndarray:
    """Return bin-level calibration statistics.

    The returned array has these columns:

    ``bin_left, bin_right, n, accuracy, confidence, gap``

    where ``gap = accuracy - confidence``.
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    y = to_numpy_1d(y_correct, dtype=float)
    confidence = to_numpy_1d(confidence_scores, dtype=float)

    mask = finite_mask(y, confidence)
    y = y[mask]
    confidence = np.clip(confidence[mask], 0.0, 1.0)

    if y.size == 0:
        return np.empty((0, 6), dtype=float)

    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    elif strategy == "quantile":
        bin_edges = np.quantile(confidence, np.linspace(0.0, 1.0, n_bins + 1))
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
        bin_edges = np.unique(bin_edges)

        if bin_edges.size < 2:
            return np.empty((0, 6), dtype=float)

    else:
        raise ValueError("strategy must be either 'uniform' or 'quantile'.")

    rows = []

    for i in range(len(bin_edges) - 1):
        left = float(bin_edges[i])
        right = float(bin_edges[i + 1])

        if i == len(bin_edges) - 2:
            in_bin = (confidence >= left) & (confidence <= right)
        else:
            in_bin = (confidence >= left) & (confidence < right)

        count = int(np.sum(in_bin))

        if count == 0:
            rows.append([left, right, 0, np.nan, np.nan, np.nan])
            continue

        accuracy = float(np.mean(y[in_bin]))
        avg_confidence = float(np.mean(confidence[in_bin]))

        rows.append(
            [
                left,
                right,
                count,
                accuracy,
                avg_confidence,
                accuracy - avg_confidence,
            ]
        )

    return np.asarray(rows, dtype=float)


# ---------------------------------------------------------------------
# Logistic calibration: CITL and C-Slope
# ---------------------------------------------------------------------


def _negative_log_likelihood(logits: np.ndarray, y: np.ndarray) -> float:
    """Binary logistic negative log-likelihood for logits."""
    return float(np.sum(np.logaddexp(0.0, logits) - y * logits))


def calibration_in_the_large(
    y_correct: ArrayLike,
    confidence_scores: ArrayLike,
    eps: float = 1e-6,
) -> float:
    """Compute Calibration-in-the-Large.

    CITL is fitted as the intercept ``alpha`` in:

    ``logit(P(correct)) = alpha + logit(confidence)``

    with slope fixed to 1. The ideal value is 0.
    """
    y = to_numpy_1d(y_correct, dtype=float)
    confidence = clip_probabilities(confidence_scores, eps=eps)

    mask = finite_mask(y, confidence)
    y = y[mask]
    confidence = confidence[mask]

    if y.size < 2:
        return float("nan")

    x = safe_logit(confidence, eps=eps)

    def objective(alpha: float) -> float:
        logits = alpha + x
        return _negative_log_likelihood(logits, y)

    try:
        result = minimize_scalar(objective, bounds=(-20.0, 20.0), method="bounded")

        if not result.success or not np.isfinite(result.fun):
            return float("nan")

        return float(result.x)

    except Exception:
        return float("nan")


def calibration_slope_and_intercept(
    y_correct: ArrayLike,
    confidence_scores: ArrayLike,
    eps: float = 1e-6,
) -> Tuple[float, float]:
    """Fit logistic calibration intercept and slope.

    Fits:

    ``logit(P(correct)) = alpha + beta * logit(confidence)``

    Returns ``(intercept, slope)``. The ideal slope is 1.
    """
    y = to_numpy_1d(y_correct, dtype=float)
    confidence = clip_probabilities(confidence_scores, eps=eps)

    mask = finite_mask(y, confidence)
    y = y[mask]
    confidence = confidence[mask]

    if y.size < 2:
        return float("nan"), float("nan")

    x = safe_logit(confidence, eps=eps)

    if np.isclose(np.nanstd(x), 0.0):
        return float("nan"), float("nan")

    def objective(params: np.ndarray) -> float:
        alpha, beta = params
        logits = alpha + beta * x
        return _negative_log_likelihood(logits, y)

    try:
        result = minimize(objective, x0=np.array([0.0, 1.0]), method="BFGS")

        if not result.success or not np.isfinite(result.fun):
            return float("nan"), float("nan")

        alpha, beta = result.x
        return float(alpha), float(beta)

    except Exception:
        return float("nan"), float("nan")


def calibration_slope(
    y_correct: ArrayLike,
    confidence_scores: ArrayLike,
    eps: float = 1e-6,
) -> float:
    """Compute the logistic calibration slope. The ideal value is 1."""
    _, slope = calibration_slope_and_intercept(y_correct, confidence_scores, eps=eps)
    return slope


def mean_calibration_bias(
    y_correct: ArrayLike,
    confidence_scores: ArrayLike,
) -> float:
    """Compute mean confidence minus empirical accuracy.

    Positive values indicate overconfidence.
    Negative values indicate underconfidence.
    """
    y = to_numpy_1d(y_correct, dtype=float)
    confidence = to_numpy_1d(confidence_scores, dtype=float)

    mask = finite_mask(y, confidence)
    y = y[mask]
    confidence = confidence[mask]

    if y.size == 0:
        return float("nan")

    return float(np.mean(confidence) - np.mean(y))


# ---------------------------------------------------------------------
# Combined APIs
# ---------------------------------------------------------------------


def compute_calibration_from_correctness(
    y_correct: ArrayLike,
    confidence_scores: ArrayLike,
    n_bins: int = 10,
    bin_strategy: str = "uniform",
    prefix: str = "",
) -> Dict[str, float]:
    """Compute calibration metrics from correctness labels and confidence scores."""
    return {
        f"{prefix}CITL": calibration_in_the_large(y_correct, confidence_scores),
        f"{prefix}C-Slope": calibration_slope(y_correct, confidence_scores),
        f"{prefix}ECE": expected_calibration_error(
            y_correct,
            confidence_scores,
            n_bins=n_bins,
            strategy=bin_strategy,
        ),
        f"{prefix}Mean Calibration Bias": mean_calibration_bias(
            y_correct,
            confidence_scores,
        ),
    }


def compute_calibration_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    confidence_scores: ArrayLike,
    n_bins: int = 10,
    bin_strategy: str = "uniform",
    prefix: str = "",
) -> Dict[str, float]:
    """Compute calibration metrics from true labels, predictions, and confidence."""
    y_correct = prediction_correct_labels(y_true, y_pred)

    return compute_calibration_from_correctness(
        y_correct=y_correct,
        confidence_scores=confidence_scores,
        n_bins=n_bins,
        bin_strategy=bin_strategy,
        prefix=prefix,
    )


def compute_calibration_for_methods(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    method_to_confidence: Dict[str, ArrayLike],
    n_bins: int = 10,
    bin_strategy: str = "uniform",
) -> Dict[str, Dict[str, float]]:
    """Compute calibration metrics for multiple confidence methods."""
    results: Dict[str, Dict[str, float]] = {}

    for method, confidence in method_to_confidence.items():
        results[method] = compute_calibration_metrics(
            y_true=y_true,
            y_pred=y_pred,
            confidence_scores=confidence,
            n_bins=n_bins,
            bin_strategy=bin_strategy,
        )

    return results


__all__ = [
    "confidence_from_uncertainty",
    "uncertainty_to_confidence",
    "expected_calibration_error",
    "compute_ece",
    "calibration_bins",
    "calibration_in_the_large",
    "calibration_slope_and_intercept",
    "calibration_slope",
    "mean_calibration_bias",
    "compute_calibration_from_correctness",
    "compute_calibration_metrics",
    "compute_calibration_for_methods",
]
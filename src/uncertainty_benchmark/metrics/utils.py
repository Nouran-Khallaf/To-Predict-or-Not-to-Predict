"""Shared helper utilities for uncertainty metrics.

This module contains small validation and conversion helpers used by several
metric modules. Keeping them here avoids repeating the same helper functions in
calibration, discrimination, selective prediction, and rejection metrics.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np


ArrayLike = Iterable[object]


def to_numpy_1d(values: ArrayLike, dtype: Optional[type] = None) -> np.ndarray:
    """Convert values to a one-dimensional NumPy array.

    Parameters
    ----------
    values:
        Input values. Can be a list, tuple, pandas Series, NumPy array, or other
        iterable.
    dtype:
        Optional dtype to cast the array to.

    Returns
    -------
    np.ndarray
        One-dimensional NumPy array.
    """
    arr = np.asarray(values)

    if arr.ndim == 0:
        arr = arr.reshape(1)
    else:
        arr = np.ravel(arr)

    if dtype is not None:
        arr = arr.astype(dtype)

    return arr


def validate_same_length(*arrays: np.ndarray) -> None:
    """Raise ValueError if arrays do not have the same first dimension."""
    if not arrays:
        return

    n = arrays[0].shape[0]

    for arr in arrays:
        if arr.shape[0] != n:
            raise ValueError(
                "All arrays must have the same length. "
                f"Expected {n}, got {arr.shape[0]}."
            )


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    """Return a mask where all arrays have finite numeric values."""
    if not arrays:
        raise ValueError("At least one array is required.")

    validate_same_length(*arrays)

    mask = np.ones(arrays[0].shape[0], dtype=bool)

    for arr in arrays:
        mask &= np.isfinite(arr.astype(float))

    return mask


def prediction_error_labels(y_true: ArrayLike, y_pred: ArrayLike) -> np.ndarray:
    """Return binary error labels.

    Returns
    -------
    np.ndarray
        Array where:

        - 1 = incorrect prediction
        - 0 = correct prediction
    """
    true_arr = to_numpy_1d(y_true)
    pred_arr = to_numpy_1d(y_pred)

    validate_same_length(true_arr, pred_arr)

    return (true_arr != pred_arr).astype(int)


def prediction_correct_labels(y_true: ArrayLike, y_pred: ArrayLike) -> np.ndarray:
    """Return binary correctness labels.

    Returns
    -------
    np.ndarray
        Array where:

        - 1 = correct prediction
        - 0 = incorrect prediction
    """
    true_arr = to_numpy_1d(y_true)
    pred_arr = to_numpy_1d(y_pred)

    validate_same_length(true_arr, pred_arr)

    return (true_arr == pred_arr).astype(int)


def has_two_classes(binary_labels: ArrayLike) -> bool:
    """Return True if binary labels contain both 0 and 1."""
    labels = to_numpy_1d(binary_labels, dtype=float)
    labels = labels[np.isfinite(labels)]

    if labels.size == 0:
        return False

    return np.unique(labels).size == 2


def clip_probabilities(values: ArrayLike, eps: float = 1e-6) -> np.ndarray:
    """Clip probability/confidence values to [eps, 1 - eps]."""
    arr = to_numpy_1d(values, dtype=float)
    return np.clip(arr, eps, 1.0 - eps)


def safe_logit(values: ArrayLike, eps: float = 1e-6) -> np.ndarray:
    """Compute logit values after probability clipping."""
    p = clip_probabilities(values, eps=eps)
    return np.log(p / (1.0 - p))


def normalise_rejection_rates(rejection_rates: Sequence[float]) -> list[float]:
    """Validate and return rejection rates as floats."""
    rates = [float(rate) for rate in rejection_rates]

    for rate in rates:
        if rate < 0.0 or rate > 1.0:
            raise ValueError(
                f"Invalid rejection rate {rate}. "
                "Rejection rates must be between 0 and 1."
            )

    return rates


def safe_mean(values: ArrayLike) -> float:
    """Return the mean of finite values, or NaN if no finite values exist."""
    arr = to_numpy_1d(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return float("nan")

    return float(np.mean(arr))


__all__ = [
    "ArrayLike",
    "to_numpy_1d",
    "validate_same_length",
    "finite_mask",
    "prediction_error_labels",
    "prediction_correct_labels",
    "has_two_classes",
    "clip_probabilities",
    "safe_logit",
    "normalise_rejection_rates",
    "safe_mean",
]
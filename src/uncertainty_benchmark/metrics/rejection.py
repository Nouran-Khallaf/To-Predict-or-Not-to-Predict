#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed-rate rejection metrics for uncertainty estimation.

These functions evaluate what happens when we reject the most uncertain examples
at predefined rejection rates, such as 1%, 5%, 10%, and 15%.

Convention
----------
Uncertainty scores follow this convention:

    larger score = more uncertain = rejected earlier

Main outputs
------------
For each rejection rate, the module can compute:

- Macro ΔF1:
    macro-F1 on retained examples minus original macro-F1

- Rejected error rate:
    percentage of rejected examples that were originally misclassified

- Rejected count:
    number of rejected examples

- Incorrect rejected count:
    number of rejected examples that were originally misclassified
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import f1_score

from uncertainty_benchmark.metrics.utils import (
    ArrayLike,
    normalise_rejection_rates,
    prediction_error_labels,
    to_numpy_1d,
    validate_same_length,
)


DEFAULT_REJECTION_RATES = [0.01, 0.05, 0.10, 0.15]
DEFAULT_REVERSE_SCORE_METHODS = {"LOF", "ISOF"}


# ---------------------------------------------------------------------
# Helpers specific to fixed-rate rejection
# ---------------------------------------------------------------------


def finite_score_mask(scores: np.ndarray) -> np.ndarray:
    """Return mask for finite uncertainty scores."""
    return np.isfinite(scores.astype(float))


def rejection_count(
    n_examples: int,
    rejection_rate: float,
    use_ceil: bool = True,
) -> int:
    """Compute the number of examples to reject at a given rate.

    Parameters
    ----------
    n_examples:
        Dataset size.
    rejection_rate:
        Fraction of examples to reject.
    use_ceil:
        If True, uses ceil(rate * n). This ensures that 1% of 1299 gives 13
        rather than 12. If False, uses round(rate * n).
    """
    if n_examples < 0:
        raise ValueError("n_examples must be non-negative.")

    if rejection_rate < 0.0 or rejection_rate > 1.0:
        raise ValueError("rejection_rate must be between 0 and 1.")

    if use_ceil:
        n_reject = int(math.ceil(rejection_rate * n_examples))
    else:
        n_reject = int(round(rejection_rate * n_examples))

    return max(0, min(n_reject, n_examples))


def sorted_indices_by_uncertainty(
    uncertainty_scores: ArrayLike,
    reverse_score: bool = False,
) -> np.ndarray:
    """Return indices sorted from least uncertain to most uncertain.

    Parameters
    ----------
    uncertainty_scores:
        Uncertainty scores.
    reverse_score:
        If True, scores are multiplied by -1 before sorting. This is useful for
        methods such as LOF/ISOF when lower raw values indicate higher
        uncertainty.

    Notes
    -----
    NaN values are treated as very low uncertainty so that they are not rejected
    first. Users should normally avoid NaNs in score columns.
    """
    scores = to_numpy_1d(uncertainty_scores, dtype=float)

    if reverse_score:
        scores = -scores

    clean_scores = np.where(np.isfinite(scores), scores, -np.inf)

    return np.argsort(clean_scores, kind="mergesort")


def split_rejected_kept_indices(
    uncertainty_scores: ArrayLike,
    rejection_rate: float,
    reverse_score: bool = False,
    use_ceil: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split indices into rejected and kept sets at a fixed rejection rate.

    Returns
    -------
    rejected_idx, kept_idx
    """
    scores = to_numpy_1d(uncertainty_scores, dtype=float)

    order = sorted_indices_by_uncertainty(
        scores,
        reverse_score=reverse_score,
    )

    n_reject = rejection_count(
        n_examples=len(scores),
        rejection_rate=rejection_rate,
        use_ceil=use_ceil,
    )

    if n_reject == 0:
        return np.array([], dtype=int), order

    rejected_idx = order[-n_reject:]
    kept_idx = order[:-n_reject]

    return rejected_idx, kept_idx


# ---------------------------------------------------------------------
# Core fixed-rate rejection metrics
# ---------------------------------------------------------------------


def baseline_macro_f1(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: str = "macro",
) -> float:
    """Compute baseline F1 before rejection."""
    true_arr = to_numpy_1d(y_true)
    pred_arr = to_numpy_1d(y_pred)

    validate_same_length(true_arr, pred_arr)

    if true_arr.size == 0:
        return float("nan")

    return float(f1_score(true_arr, pred_arr, average=average))


def baseline_error_rate_pct(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> float:
    """Compute original error rate as a percentage."""
    errors = prediction_error_labels(y_true, y_pred)

    if errors.size == 0:
        return float("nan")

    return float(100.0 * np.mean(errors))


def macro_f1_after_rejection(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    uncertainty_scores: ArrayLike,
    rejection_rate: float,
    reverse_score: bool = False,
    average: str = "macro",
    use_ceil: bool = True,
) -> float:
    """Compute F1 on retained examples after rejecting the most uncertain items."""
    true_arr = to_numpy_1d(y_true)
    pred_arr = to_numpy_1d(y_pred)
    scores = to_numpy_1d(uncertainty_scores, dtype=float)

    validate_same_length(true_arr, pred_arr, scores)

    _, kept_idx = split_rejected_kept_indices(
        uncertainty_scores=scores,
        rejection_rate=rejection_rate,
        reverse_score=reverse_score,
        use_ceil=use_ceil,
    )

    if kept_idx.size == 0:
        return float("nan")

    return float(
        f1_score(
            true_arr[kept_idx],
            pred_arr[kept_idx],
            average=average,
        )
    )


def macro_f1_delta_after_rejection(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    uncertainty_scores: ArrayLike,
    rejection_rate: float,
    reverse_score: bool = False,
    average: str = "macro",
    use_ceil: bool = True,
) -> float:
    """Compute Macro ΔF1 after rejection.

    Macro ΔF1 = F1(retained examples) - F1(all examples)
    """
    base = baseline_macro_f1(
        y_true=y_true,
        y_pred=y_pred,
        average=average,
    )

    kept = macro_f1_after_rejection(
        y_true=y_true,
        y_pred=y_pred,
        uncertainty_scores=uncertainty_scores,
        rejection_rate=rejection_rate,
        reverse_score=reverse_score,
        average=average,
        use_ceil=use_ceil,
    )

    if not np.isfinite(base) or not np.isfinite(kept):
        return float("nan")

    return float(kept - base)


def rejected_error_counts(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    uncertainty_scores: ArrayLike,
    rejection_rate: float,
    reverse_score: bool = False,
    use_ceil: bool = True,
) -> Tuple[int, int]:
    """Return number rejected and number incorrect among rejected."""
    true_arr = to_numpy_1d(y_true)
    pred_arr = to_numpy_1d(y_pred)
    scores = to_numpy_1d(uncertainty_scores, dtype=float)

    validate_same_length(true_arr, pred_arr, scores)

    rejected_idx, _ = split_rejected_kept_indices(
        uncertainty_scores=scores,
        rejection_rate=rejection_rate,
        reverse_score=reverse_score,
        use_ceil=use_ceil,
    )

    n_rejected = int(rejected_idx.size)

    if n_rejected == 0:
        return 0, 0

    n_incorrect = int(np.sum(true_arr[rejected_idx] != pred_arr[rejected_idx]))

    return n_rejected, n_incorrect


def pct_incorrect_rejected(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    uncertainty_scores: ArrayLike,
    rejection_rate: float,
    reverse_score: bool = False,
    use_ceil: bool = True,
) -> float:
    """Compute percentage of rejected examples that were originally incorrect."""
    n_rejected, n_incorrect = rejected_error_counts(
        y_true=y_true,
        y_pred=y_pred,
        uncertainty_scores=uncertainty_scores,
        rejection_rate=rejection_rate,
        reverse_score=reverse_score,
        use_ceil=use_ceil,
    )

    if n_rejected == 0:
        return float("nan")

    return float(100.0 * n_incorrect / n_rejected)


def compute_rejection_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    uncertainty_scores: ArrayLike,
    rejection_rates: Sequence[float] = DEFAULT_REJECTION_RATES,
    reverse_score: bool = False,
    average: str = "macro",
    use_ceil: bool = True,
) -> Dict[float, Dict[str, float]]:
    """Compute fixed-rate rejection metrics for one uncertainty method.

    Returns
    -------
    dict
        Mapping from rejection rate to metric dictionary.
    """
    rates = normalise_rejection_rates(rejection_rates)

    base_f1 = baseline_macro_f1(
        y_true=y_true,
        y_pred=y_pred,
        average=average,
    )

    base_error_pct = baseline_error_rate_pct(
        y_true=y_true,
        y_pred=y_pred,
    )

    results: Dict[float, Dict[str, float]] = {}

    for rate in rates:
        kept_f1 = macro_f1_after_rejection(
            y_true=y_true,
            y_pred=y_pred,
            uncertainty_scores=uncertainty_scores,
            rejection_rate=rate,
            reverse_score=reverse_score,
            average=average,
            use_ceil=use_ceil,
        )

        if np.isfinite(kept_f1) and np.isfinite(base_f1):
            delta = kept_f1 - base_f1
        else:
            delta = float("nan")

        n_rejected, n_incorrect = rejected_error_counts(
            y_true=y_true,
            y_pred=y_pred,
            uncertainty_scores=uncertainty_scores,
            rejection_rate=rate,
            reverse_score=reverse_score,
            use_ceil=use_ceil,
        )

        if n_rejected:
            pct_bad = float(100.0 * n_incorrect / n_rejected)
        else:
            pct_bad = float("nan")

        results[rate] = {
            "baseline_macro_f1": float(base_f1),
            "macro_f1_after_rejection": float(kept_f1),
            "macro_delta": float(delta),
            "baseline_error_pct": float(base_error_pct),
            "pct_incorrect_rejected": float(pct_bad),
            "n_rejected": int(n_rejected),
            "n_incorrect_rejected": int(n_incorrect),
        }

    return results


# ---------------------------------------------------------------------
# Multi-method APIs
# ---------------------------------------------------------------------


def compute_rejection_for_methods(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    method_to_scores: Mapping[str, ArrayLike],
    rejection_rates: Sequence[float] = DEFAULT_REJECTION_RATES,
    reverse_score_methods: Iterable[str] = DEFAULT_REVERSE_SCORE_METHODS,
    average: str = "macro",
    use_ceil: bool = True,
) -> Dict[str, Dict[float, Dict[str, float]]]:
    """Compute rejection metrics for multiple uncertainty methods.

    Parameters
    ----------
    method_to_scores:
        Mapping from method name to uncertainty score vector.
    reverse_score_methods:
        Methods whose raw scores should be multiplied by -1 before sorting.
    """
    reverse_set = set(reverse_score_methods)
    results: Dict[str, Dict[float, Dict[str, float]]] = {}

    for method, scores in method_to_scores.items():
        results[method] = compute_rejection_metrics(
            y_true=y_true,
            y_pred=y_pred,
            uncertainty_scores=scores,
            rejection_rates=rejection_rates,
            reverse_score=(method in reverse_set),
            average=average,
            use_ceil=use_ceil,
        )

    return results


def rejection_results_to_rows(
    results: Mapping[str, Mapping[float, Mapping[str, float]]],
) -> List[Dict[str, float | str]]:
    """Convert nested rejection results into tidy row dictionaries."""
    rows: List[Dict[str, float | str]] = []

    for method, per_rate in results.items():
        for rate, metrics in per_rate.items():
            row: Dict[str, float | str] = {
                "method": method,
                "rejection_rate": float(rate),
            }
            row.update(metrics)
            rows.append(row)

    return rows


def compute_rejection_summary_arrays(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    method_to_scores: Mapping[str, ArrayLike],
    rejection_rates: Sequence[float] = DEFAULT_REJECTION_RATES,
    reverse_score_methods: Iterable[str] = DEFAULT_REVERSE_SCORE_METHODS,
    average: str = "macro",
    use_ceil: bool = True,
) -> Dict[str, np.ndarray]:
    """Return compact arrays for downstream aggregation.

    Returns arrays with shape:

        n_methods x n_rejection_rates

    Keys
    ----
    - methods
    - rejection_rates
    - macro_delta
    - pct_incorrect_rejected
    - n_rejected
    - n_incorrect_rejected
    - baseline_error_pct
    """
    methods = list(method_to_scores.keys())
    rates = normalise_rejection_rates(rejection_rates)
    reverse_set = set(reverse_score_methods)

    macro_delta = np.full((len(methods), len(rates)), np.nan, dtype=float)
    pct_bad = np.full((len(methods), len(rates)), np.nan, dtype=float)
    n_rejected = np.full((len(methods), len(rates)), np.nan, dtype=float)
    n_incorrect = np.full((len(methods), len(rates)), np.nan, dtype=float)

    baseline_error = baseline_error_rate_pct(
        y_true=y_true,
        y_pred=y_pred,
    )

    for method_idx, method in enumerate(methods):
        per_rate = compute_rejection_metrics(
            y_true=y_true,
            y_pred=y_pred,
            uncertainty_scores=method_to_scores[method],
            rejection_rates=rates,
            reverse_score=(method in reverse_set),
            average=average,
            use_ceil=use_ceil,
        )

        for rate_idx, rate in enumerate(rates):
            metrics = per_rate[rate]

            macro_delta[method_idx, rate_idx] = metrics["macro_delta"]
            pct_bad[method_idx, rate_idx] = metrics["pct_incorrect_rejected"]
            n_rejected[method_idx, rate_idx] = metrics["n_rejected"]
            n_incorrect[method_idx, rate_idx] = metrics["n_incorrect_rejected"]

    return {
        "methods": np.asarray(methods, dtype=object),
        "rejection_rates": np.asarray(rates, dtype=float),
        "macro_delta": macro_delta,
        "pct_incorrect_rejected": pct_bad,
        "n_rejected": n_rejected,
        "n_incorrect_rejected": n_incorrect,
        "baseline_error_pct": np.asarray([baseline_error], dtype=float),
    }


__all__ = [
    "DEFAULT_REJECTION_RATES",
    "DEFAULT_REVERSE_SCORE_METHODS",
    "finite_score_mask",
    "rejection_count",
    "sorted_indices_by_uncertainty",
    "split_rejected_kept_indices",
    "baseline_macro_f1",
    "baseline_error_rate_pct",
    "macro_f1_after_rejection",
    "macro_f1_delta_after_rejection",
    "rejected_error_counts",
    "pct_incorrect_rejected",
    "compute_rejection_metrics",
    "compute_rejection_for_methods",
    "rejection_results_to_rows",
    "compute_rejection_summary_arrays",
]
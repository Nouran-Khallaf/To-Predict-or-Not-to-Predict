"""Selective prediction metrics for uncertainty estimation.

These metrics evaluate how model performance changes when we keep only the most
confident predictions or reject the most uncertain predictions.

Convention
----------
Most functions use uncertainty scores with this convention:

    larger score = more uncertain = reject earlier

For selective prediction, examples are sorted from least uncertain to most
uncertain. Coverage decreases as increasingly uncertain examples are rejected.

Paper/reporting convention
--------------------------
This module reports the selective-prediction metrics using the convention used
in the original paper tables:

- ``RC-AUC`` is on a retained-performance scale, so higher is better. Internally
  this is computed as ``1 - AURC``, where AURC is the mean retained risk over
  the risk-coverage curve.
- ``Norm RC-AUC`` is also higher-is-better: 1.0 is optimal, 0.0 is random, and
  values below 0.0 are worse than random.
- ``E-AUoptRC`` remains a lower-is-better excess-risk-area metric:
  ``AURC - optimal_AURC``.
- ``TI`` and ``TI@coverage`` are reported on a retained-accuracy scale, not as
  risk reduction. This keeps them comparable to the older high-valued tables
  where values are around 0.8--0.9 rather than around 0.0--0.2.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from uncertainty_benchmark.metrics.utils import (
    ArrayLike,
    finite_mask,
    prediction_error_labels,
    to_numpy_1d,
)


# ---------------------------------------------------------------------
# Internal preparation
# ---------------------------------------------------------------------


def _prepare_errors_and_scores(
    y_error: ArrayLike,
    uncertainty_scores: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate, clean, and sort binary errors plus uncertainty scores.

    Returned arrays are sorted from least uncertain to most uncertain.
    Selective prediction retains prefixes of this order.
    """
    errors = to_numpy_1d(y_error, dtype=float)
    scores = to_numpy_1d(uncertainty_scores, dtype=float)

    mask = finite_mask(errors, scores)
    errors = errors[mask].astype(int)
    scores = scores[mask]

    if errors.size == 0:
        return errors, scores

    order = np.argsort(scores, kind="mergesort")
    return errors[order], scores[order]


def _clean_errors(y_error: ArrayLike) -> np.ndarray:
    """Return finite binary error labels as a 1D int array."""
    errors = to_numpy_1d(y_error, dtype=float)
    errors = errors[np.isfinite(errors)].astype(int)
    return errors


# ---------------------------------------------------------------------
# Risk-coverage curves
# ---------------------------------------------------------------------


def risk_coverage_curve_from_errors(
    y_error: ArrayLike,
    uncertainty_scores: ArrayLike,
    include_zero_coverage: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the risk-coverage curve from binary error labels.

    Parameters
    ----------
    y_error:
        Binary labels where 1 = incorrect and 0 = correct.
    uncertainty_scores:
        Uncertainty scores where larger means more uncertain.
    include_zero_coverage:
        If True, append coverage=0 with risk=0. This is mainly useful for
        plotting. Metric integration usually uses coverage > 0.

    Returns
    -------
    coverages, risks:
        Arrays ordered from high coverage to low coverage.
    """
    sorted_errors, _ = _prepare_errors_and_scores(y_error, uncertainty_scores)
    n = sorted_errors.size

    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    cumulative_errors = np.cumsum(sorted_errors)

    # For coverage k/n, keep the k least uncertain examples.
    # We produce k = n, n-1, ..., 1.
    ks = np.arange(n, 0, -1)

    retained_errors = cumulative_errors[ks - 1]
    risks = retained_errors / ks
    coverages = ks / n

    if include_zero_coverage:
        coverages = np.concatenate([coverages, np.array([0.0])])
        risks = np.concatenate([risks, np.array([0.0])])

    return coverages.astype(float), risks.astype(float)


def risk_coverage_curve(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    uncertainty_scores: ArrayLike,
    include_zero_coverage: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the risk-coverage curve from true and predicted labels."""
    y_error = prediction_error_labels(y_true, y_pred)

    return risk_coverage_curve_from_errors(
        y_error=y_error,
        uncertainty_scores=uncertainty_scores,
        include_zero_coverage=include_zero_coverage,
    )


def risk_at_coverage_from_errors(
    y_error: ArrayLike,
    uncertainty_scores: ArrayLike,
    target_coverage: float,
) -> float:
    """Compute retained risk at a target coverage.

    The number of retained examples is ceil(target_coverage * n), with at least
    one retained example when target_coverage > 0.
    """
    if not 0.0 <= target_coverage <= 1.0:
        raise ValueError("target_coverage must be between 0 and 1.")

    sorted_errors, _ = _prepare_errors_and_scores(y_error, uncertainty_scores)
    n = sorted_errors.size

    if n == 0:
        return float("nan")

    if target_coverage == 0:
        return 0.0

    k = int(np.ceil(target_coverage * n))
    k = max(1, min(k, n))

    return float(np.mean(sorted_errors[:k]))


def risk_at_coverage(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    uncertainty_scores: ArrayLike,
    target_coverage: float,
) -> float:
    """Compute retained risk at target coverage from true and predicted labels."""
    y_error = prediction_error_labels(y_true, y_pred)

    return risk_at_coverage_from_errors(
        y_error=y_error,
        uncertainty_scores=uncertainty_scores,
        target_coverage=target_coverage,
    )


def retained_accuracy_at_coverage_from_errors(
    y_error: ArrayLike,
    uncertainty_scores: ArrayLike,
    target_coverage: float,
) -> float:
    """Compute retained accuracy at a target coverage.

    This is ``1 - risk_at_coverage`` and is therefore higher-is-better.
    """
    risk = risk_at_coverage_from_errors(
        y_error=y_error,
        uncertainty_scores=uncertainty_scores,
        target_coverage=target_coverage,
    )

    if not np.isfinite(risk):
        return float("nan")

    return float(1.0 - risk)


# ---------------------------------------------------------------------
# AURC / RC-AUC and optimal curves
# ---------------------------------------------------------------------


def aurc_from_errors(
    y_error: ArrayLike,
    uncertainty_scores: ArrayLike,
    normalise_by_step_count: bool = True,
) -> float:
    """Compute AURC: area under the risk-coverage curve.

    AURC is a risk-area quantity, so lower is better. It is kept as an internal
    helper because the paper table reports ``RC-AUC = 1 - AURC``.
    """
    coverages, risks = risk_coverage_curve_from_errors(y_error, uncertainty_scores)

    if risks.size == 0:
        return float("nan")

    if normalise_by_step_count:
        return float(np.mean(risks))

    # np.trapz expects x in increasing order.
    order = np.argsort(coverages)
    return float(np.trapz(risks[order], coverages[order]))


def rc_auc_from_errors(
    y_error: ArrayLike,
    uncertainty_scores: ArrayLike,
    normalise_by_step_count: bool = True,
) -> float:
    """Compute paper-style RC-AUC on a retained-performance scale.

    This returns ``1 - AURC``. Higher is better. This is the convention used in
    the older LaTeX tables where RC-AUC values are close to 0.8--0.95.
    """
    aurc = aurc_from_errors(
        y_error=y_error,
        uncertainty_scores=uncertainty_scores,
        normalise_by_step_count=normalise_by_step_count,
    )

    if not np.isfinite(aurc):
        return float("nan")

    return float(1.0 - aurc)


def optimal_aurc_from_errors(y_error: ArrayLike) -> float:
    """Compute optimal AURC by rejecting incorrect predictions first.

    Lower is better. This is the best possible risk-coverage curve given the
    observed correct/incorrect labels.
    """
    errors = _clean_errors(y_error)

    if errors.size == 0:
        return float("nan")

    # Least uncertain retained first should be all correct examples, then errors.
    optimal_order = np.sort(errors)  # 0s first, 1s last
    dummy_scores = np.arange(errors.size, dtype=float)

    return aurc_from_errors(
        y_error=optimal_order,
        uncertainty_scores=dummy_scores,
        normalise_by_step_count=True,
    )


def optimal_rc_auc_from_errors(y_error: ArrayLike) -> float:
    """Compute paper-style optimal RC-AUC.

    This returns ``1 - optimal_AURC``. Higher is better.
    """
    opt_aurc = optimal_aurc_from_errors(y_error)

    if not np.isfinite(opt_aurc):
        return float("nan")

    return float(1.0 - opt_aurc)


def random_aurc_from_errors(y_error: ArrayLike) -> float:
    """Expected AURC for a non-informative uncertainty ranking.

    With a random ranking, the expected retained risk at each coverage equals
    the baseline error rate.
    """
    errors = _clean_errors(y_error)

    if errors.size == 0:
        return float("nan")

    return float(np.mean(errors))


def random_rc_auc_from_errors(y_error: ArrayLike) -> float:
    """Expected paper-style RC-AUC for a random uncertainty ranking.

    This returns ``1 - baseline_error_rate``. Higher is better.
    """
    rand_aurc = random_aurc_from_errors(y_error)

    if not np.isfinite(rand_aurc):
        return float("nan")

    return float(1.0 - rand_aurc)


def normalised_rc_auc_from_errors(
    y_error: ArrayLike,
    uncertainty_scores: ArrayLike,
) -> float:
    """Compute paper-style normalised RC-AUC.

    This is higher-is-better:

        Norm RC-AUC = (AURC_random - AURC) / (AURC_random - AURC_opt)

    Interpretation:

    - 1.0 = optimal ranking
    - 0.0 = random ranking
    - <0.0 = worse than random
    """
    errors = to_numpy_1d(y_error, dtype=float)
    scores = to_numpy_1d(uncertainty_scores, dtype=float)

    mask = finite_mask(errors, scores)
    errors = errors[mask].astype(int)
    scores = scores[mask]

    if errors.size == 0:
        return float("nan")

    aurc = aurc_from_errors(errors, scores)
    opt = optimal_aurc_from_errors(errors)
    rand = random_aurc_from_errors(errors)

    if not np.isfinite(aurc) or not np.isfinite(opt) or not np.isfinite(rand):
        return float("nan")

    denom = rand - opt

    if np.isclose(denom, 0.0):
        return float("nan")

    return float((rand - aurc) / denom)


def e_auopt_rc_from_errors(
    y_error: ArrayLike,
    uncertainty_scores: ArrayLike,
) -> float:
    """Compute excess area over the optimal risk-coverage curve.

    E-AUoptRC = AURC - optimal AURC.

    Lower is better. Zero means optimal ranking. This metric intentionally stays
    on the risk-area scale even though ``RC-AUC`` is reported as ``1 - AURC``.
    """
    errors = to_numpy_1d(y_error, dtype=float)
    scores = to_numpy_1d(uncertainty_scores, dtype=float)

    mask = finite_mask(errors, scores)
    errors = errors[mask].astype(int)
    scores = scores[mask]

    if errors.size == 0:
        return float("nan")

    aurc = aurc_from_errors(errors, scores)
    opt = optimal_aurc_from_errors(errors)

    if not np.isfinite(aurc) or not np.isfinite(opt):
        return float("nan")

    return float(aurc - opt)


# ---------------------------------------------------------------------
# Threshold/coverage metrics
# ---------------------------------------------------------------------


def optimal_coverage_from_errors(
    y_error: ArrayLike,
    uncertainty_scores: ArrayLike,
) -> float:
    """Return the coverage that gives the highest retained accuracy.

    If there are ties, the largest coverage is returned. This avoids selecting a
    tiny retained set when several thresholds achieve the same retained
    accuracy.
    """
    coverages, risks = risk_coverage_curve_from_errors(y_error, uncertainty_scores)

    if risks.size == 0:
        return float("nan")

    accuracies = 1.0 - risks
    best = np.nanmax(accuracies)
    tied = np.where(np.isclose(accuracies, best, equal_nan=False))[0]

    if tied.size == 0:
        return float("nan")

    return float(np.nanmax(coverages[tied]))


def threshold_improvement_from_errors(
    y_error: ArrayLike,
    uncertainty_scores: ArrayLike,
    target_coverage: Optional[float] = None,
) -> float:
    """Compute paper-style TI on a retained-accuracy scale.

    If ``target_coverage`` is None, this returns the best retained accuracy over
    all empirical coverage thresholds:

        TI = max_c accuracy(c) = max_c [1 - risk(c)]

    If ``target_coverage`` is provided, this returns retained accuracy at that
    coverage:

        TI@c = 1 - risk(c)

    Higher is better.
    """
    errors = to_numpy_1d(y_error, dtype=float)
    scores = to_numpy_1d(uncertainty_scores, dtype=float)

    mask = finite_mask(errors, scores)
    errors = errors[mask].astype(int)
    scores = scores[mask]

    if errors.size == 0:
        return float("nan")

    if target_coverage is not None:
        return retained_accuracy_at_coverage_from_errors(
            y_error=errors,
            uncertainty_scores=scores,
            target_coverage=target_coverage,
        )

    _, risks = risk_coverage_curve_from_errors(errors, scores)

    if risks.size == 0:
        return float("nan")

    accuracies = 1.0 - risks
    return float(np.nanmax(accuracies))


def threshold_improvement(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    uncertainty_scores: ArrayLike,
    target_coverage: Optional[float] = None,
) -> float:
    """Compute paper-style TI from true and predicted labels."""
    y_error = prediction_error_labels(y_true, y_pred)

    return threshold_improvement_from_errors(
        y_error=y_error,
        uncertainty_scores=uncertainty_scores,
        target_coverage=target_coverage,
    )


# ---------------------------------------------------------------------
# Combined APIs
# ---------------------------------------------------------------------


def compute_selective_prediction_from_errors(
    y_error: ArrayLike,
    uncertainty_scores: ArrayLike,
    ti_coverage: float = 0.95,
    prefix: str = "",
) -> Dict[str, float]:
    """Compute selective prediction metrics from binary error labels.

    Parameters
    ----------
    y_error:
        Binary labels where 1 = incorrect and 0 = correct.
    uncertainty_scores:
        Uncertainty scores where larger means more uncertain.
    ti_coverage:
        Coverage used for TI@coverage. Default 0.95 gives TI@95.
    prefix:
        Optional prefix for metric names.
    """
    ti_name = f"TI@{int(round(ti_coverage * 100))}"

    return {
        f"{prefix}RC-AUC": rc_auc_from_errors(y_error, uncertainty_scores),
        f"{prefix}Norm RC-AUC": normalised_rc_auc_from_errors(
            y_error,
            uncertainty_scores,
        ),
        f"{prefix}E-AUoptRC": e_auopt_rc_from_errors(
            y_error,
            uncertainty_scores,
        ),
        f"{prefix}TI": threshold_improvement_from_errors(
            y_error,
            uncertainty_scores,
        ),
        f"{prefix}{ti_name}": threshold_improvement_from_errors(
            y_error,
            uncertainty_scores,
            target_coverage=ti_coverage,
        ),
        f"{prefix}Optimal Coverage": optimal_coverage_from_errors(
            y_error,
            uncertainty_scores,
        ),
    }


def compute_selective_prediction_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    uncertainty_scores: ArrayLike,
    ti_coverage: float = 0.95,
    prefix: str = "",
) -> Dict[str, float]:
    """Compute selective prediction metrics from true and predicted labels."""
    y_error = prediction_error_labels(y_true, y_pred)

    return compute_selective_prediction_from_errors(
        y_error=y_error,
        uncertainty_scores=uncertainty_scores,
        ti_coverage=ti_coverage,
        prefix=prefix,
    )


def compute_selective_prediction_for_methods(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    method_to_scores: Dict[str, ArrayLike],
    ti_coverage: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """Compute selective prediction metrics for multiple uncertainty methods."""
    results: Dict[str, Dict[str, float]] = {}

    for method, scores in method_to_scores.items():
        results[method] = compute_selective_prediction_metrics(
            y_true=y_true,
            y_pred=y_pred,
            uncertainty_scores=scores,
            ti_coverage=ti_coverage,
        )

    return results

def _selective_prediction_metrics_from_correctness(
    correct: np.ndarray,
    uncertainty_scores: np.ndarray,
    ti_fixed_coverage: float = 0.95,
) -> dict[str, float]:
    """Compute paper-style selective-prediction metrics.

    Conventions:
    - larger uncertainty = reject earlier
    - RC-AUC is reported on retained-performance scale: 1 - AURC
    - Norm RC-AUC is higher-is-better, where 1 is optimal and 0 is random
    - E-AUoptRC is still lower-is-better
    - TI is best retained accuracy across coverages
    - TI@95 is retained accuracy at 95% coverage
    """
    correct_arr = np.asarray(correct, dtype=float).ravel()
    scores = np.asarray(uncertainty_scores, dtype=float).ravel()

    mask = np.isfinite(correct_arr) & np.isfinite(scores)
    correct_arr = correct_arr[mask].astype(int)
    scores = scores[mask]

    if correct_arr.size == 0:
        return {name: float("nan") for name in SELECTIVE_METRIC_ORDER + ["Optimal Coverage"]}

    y_error = (correct_arr == 0).astype(int)

    # Sort from least uncertain to most uncertain.
    order = np.argsort(scores, kind="mergesort")
    sorted_errors = y_error[order]

    n = int(sorted_errors.size)
    ks = np.arange(n, 0, -1)

    cumulative_errors = np.cumsum(sorted_errors)
    risks = cumulative_errors[ks - 1] / ks
    accuracies = 1.0 - risks

    # AURC is risk area. Paper-style RC-AUC is retained-performance area.
    aurc = float(np.mean(risks))
    rc_auc = float(1.0 - aurc)

    # Optimal curve: all correct examples first, all errors last.
    optimal_errors = np.sort(y_error)
    optimal_cumulative_errors = np.cumsum(optimal_errors)
    optimal_risks = optimal_cumulative_errors[ks - 1] / ks
    optimal_aurc = float(np.mean(optimal_risks))

    # Random/non-informative ranking has expected risk equal to baseline error.
    random_aurc = float(np.mean(y_error))

    denom = random_aurc - optimal_aurc
    if np.isfinite(denom) and not np.isclose(denom, 0.0):
        norm_rc_auc = float((random_aurc - aurc) / denom)
    else:
        norm_rc_auc = float("nan")

    e_auopt_rc = float(aurc - optimal_aurc)

    # TI = best retained accuracy and its coverage.
    best_idx = int(np.nanargmax(accuracies))
    ti = float(accuracies[best_idx])
    optimal_coverage = float(ks[best_idx] / n)

    if not 0.0 <= ti_fixed_coverage <= 1.0:
        raise ValueError("metrics.ti_fixed_coverage must be between 0 and 1.")

    if ti_fixed_coverage == 0:
        ti_fixed = float("nan")
    else:
        k = int(np.ceil(ti_fixed_coverage * n))
        k = max(1, min(k, n))
        retained_errors = sorted_errors[:k]
        ti_fixed = float(1.0 - np.mean(retained_errors))

    return {
        "RC-AUC": rc_auc,
        "Norm RC-AUC": norm_rc_auc,
        "E-AUoptRC": e_auopt_rc,
        "TI": ti,
        "TI@95": ti_fixed,
        "Optimal Coverage": optimal_coverage,
    }
    
__all__ = [
    "prediction_error_labels",
    "risk_coverage_curve_from_errors",
    "risk_coverage_curve",
    "risk_at_coverage_from_errors",
    "risk_at_coverage",
    "retained_accuracy_at_coverage_from_errors",
    "aurc_from_errors",
    "rc_auc_from_errors",
    "optimal_aurc_from_errors",
    "optimal_rc_auc_from_errors",
    "random_aurc_from_errors",
    "random_rc_auc_from_errors",
    "normalised_rc_auc_from_errors",
    "e_auopt_rc_from_errors",
    "optimal_coverage_from_errors",
    "threshold_improvement_from_errors",
    "threshold_improvement",
    "compute_selective_prediction_from_errors",
    "compute_selective_prediction_metrics",
    "compute_selective_prediction_for_methods",
]

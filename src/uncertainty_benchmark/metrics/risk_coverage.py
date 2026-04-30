"""Backward-compatible risk-coverage and trust-index metrics.

This module keeps the original macro-F1 based risk-coverage calculation used by
the metric suite.

Important
---------
This file is different from ``selective_prediction.py``.

``selective_prediction.py`` uses error-rate risk:

    risk = mean(error)

This compatibility module uses macro-F1 risk:

    risk = 1 - macro_f1

So we keep ``compute_rc_metrics`` here to avoid changing previous experiment
outputs.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import auc, f1_score

from uncertainty_benchmark.metrics.utils import (
    ArrayLike,
    finite_mask,
    to_numpy_1d,
    validate_same_length,
)


def _prepare_inputs(
    scores_uncert: ArrayLike,
    y_true_idx: ArrayLike,
    y_pred_idx: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate inputs and remove rows with non-finite uncertainty scores."""
    scores = to_numpy_1d(scores_uncert, dtype=float)
    y_true = to_numpy_1d(y_true_idx, dtype=int)
    y_pred = to_numpy_1d(y_pred_idx, dtype=int)

    validate_same_length(scores, y_true, y_pred)

    mask = finite_mask(scores)
    return scores[mask], y_true[mask], y_pred[mask]


def macro_f1_risk_curve(
    scores_uncert: ArrayLike,
    y_true_idx: ArrayLike,
    y_pred_idx: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute macro-F1 risk-coverage curve.

    Items are sorted from low uncertainty to high uncertainty.

    At each coverage level, the system keeps only the lowest-uncertainty items
    and computes:

        risk = 1 - macro_f1

    Returns
    -------
    coverages, risks
        Coverage and risk arrays ordered from low coverage to full coverage.
    """
    scores, y_true, y_pred = _prepare_inputs(
        scores_uncert=scores_uncert,
        y_true_idx=y_true_idx,
        y_pred_idx=y_pred_idx,
    )

    n = len(y_true)

    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    order = np.argsort(scores, kind="mergesort")
    coverages = np.arange(1, n + 1, dtype=float) / n

    risks = []

    for k in range(1, n + 1):
        covered = order[:k]

        f1 = f1_score(
            y_true[covered],
            y_pred[covered],
            average="macro",
            zero_division=0,
        )

        risks.append(1.0 - f1)

    return coverages, np.asarray(risks, dtype=float)


def compute_rc_metrics(
    scores_uncert: ArrayLike,
    y_true_idx: ArrayLike,
    y_pred_idx: ArrayLike,
    ti_fixed_cov: float = 0.95,
) -> Dict[str, float]:
    """Compute macro-F1 based risk-coverage and trust-index metrics.

    Parameters
    ----------
    scores_uncert:
        Uncertainty scores where larger means more uncertain.
    y_true_idx:
        Gold class labels.
    y_pred_idx:
        Predicted class labels.
    ti_fixed_cov:
        Fixed coverage used for ``TI@95`` by default.

    Returns
    -------
    dict
        Dictionary containing:

        - ``E-AUoptRC``
        - ``TI``
        - ``TI@95``
        - ``Optimal Coverage``
    """
    scores, y_true, y_pred = _prepare_inputs(
        scores_uncert=scores_uncert,
        y_true_idx=y_true_idx,
        y_pred_idx=y_pred_idx,
    )

    n = len(y_true)

    if n == 0:
        return {
            "E-AUoptRC": float("nan"),
            "TI": float("nan"),
            "TI@95": float("nan"),
            "Optimal Coverage": float("nan"),
        }

    if not 0.0 <= ti_fixed_cov <= 1.0:
        raise ValueError("ti_fixed_cov must be between 0 and 1.")

    coverages, risks = macro_f1_risk_curve(
        scores_uncert=scores,
        y_true_idx=y_true,
        y_pred_idx=y_pred,
    )

    full_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    # Original compatibility definition:
    # optimal coverage is approximated using full macro-F1.
    k_star = max(1, min(n, int(np.floor(full_f1 * n))))
    ti_cstar = 1.0 - risks[k_star - 1]

    if k_star > 1:
        e_auopt_rc = auc(coverages[:k_star], risks[:k_star])
    else:
        e_auopt_rc = 0.0

    k_fixed = max(1, min(n, int(np.floor(ti_fixed_cov * n))))
    ti_fixed = 1.0 - risks[k_fixed - 1]

    return {
        "E-AUoptRC": float(e_auopt_rc),
        "TI": float(ti_cstar),
        "TI@95": float(ti_fixed),
        "Optimal Coverage": float(full_f1),
    }


__all__ = [
    "macro_f1_risk_curve",
    "compute_rc_metrics",
]
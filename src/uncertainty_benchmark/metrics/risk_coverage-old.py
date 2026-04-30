"""Risk-coverage and trust-index metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import auc, f1_score


def compute_rc_metrics(
    scores_uncert,
    y_true_idx,
    y_pred_idx,
    ti_fixed_cov: float = 0.95,
) -> dict:
    """Compute risk-coverage and trust-index metrics.

    Items are sorted from low uncertainty to high uncertainty.
    At each coverage level, the system keeps only the lowest-uncertainty
    items and computes macro-F1 risk:

        risk = 1 - macro_f1

    Returns
    -------
    dict
        - E-AUoptRC
        - TI
        - TI@95
        - Optimal Coverage
    """
    scores_uncert = np.asarray(scores_uncert, dtype=float)
    y_true_idx = np.asarray(y_true_idx, dtype=int)
    y_pred_idx = np.asarray(y_pred_idx, dtype=int)

    if not (len(scores_uncert) == len(y_true_idx) == len(y_pred_idx)):
        raise ValueError("scores_uncert, y_true_idx, and y_pred_idx must have same length.")

    n = len(y_true_idx)

    if n == 0:
        return {
            "E-AUoptRC": float("nan"),
            "TI": float("nan"),
            "TI@95": float("nan"),
            "Optimal Coverage": float("nan"),
        }

    order = np.argsort(scores_uncert)
    coverages = (np.arange(1, n + 1)) / n

    risks = []
    for k in range(1, n + 1):
        covered = order[:k]
        f1 = f1_score(
            y_true_idx[covered],
            y_pred_idx[covered],
            average="macro",
            zero_division=0,
        )
        risks.append(1.0 - f1)

    full_f1 = f1_score(y_true_idx, y_pred_idx, average="macro", zero_division=0)

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

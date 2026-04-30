"""Hybrid uncertainty methods."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from uncertainty_benchmark.methods.base import UncertaintyMethod


def total_uncertainty_huq(
    epistemic: np.ndarray,
    aleatoric: np.ndarray,
    threshold_min: float = 0.1,
    threshold_max: float = 0.9,
    alpha: float = 0.2,
) -> np.ndarray:
    """Hybrid rank-based uncertainty.

    This follows the logic in the original experiment:

    - epistemic uncertainty comes from a distance-based score such as MD
    - aleatoric uncertainty comes from a predictive score such as SR
    - ranks are combined, with extra handling for low epistemic and
      high aleatoric regions
    """
    epistemic = np.asarray(epistemic, dtype=float)
    aleatoric = np.asarray(aleatoric, dtype=float)

    if len(epistemic) != len(aleatoric):
        raise ValueError(
            "epistemic and aleatoric arrays must have the same length. "
            f"Got {len(epistemic)} and {len(aleatoric)}."
        )

    n_preds = len(aleatoric)
    n_lowest = int(n_preds * threshold_min)
    n_max = int(n_preds * threshold_max)

    aleatoric_rank = rankdata(aleatoric)
    epistemic_rank = rankdata(epistemic)

    total_rank = (1.0 - alpha) * epistemic_rank + alpha * aleatoric_rank

    low_epistemic = epistemic_rank <= n_lowest
    total_rank[low_epistemic] = rankdata(aleatoric[low_epistemic])

    mask = (aleatoric_rank > n_max) & low_epistemic
    total_rank[mask] = aleatoric_rank[mask]

    return total_rank


class HUQMahalanobis(UncertaintyMethod):
    """Hybrid uncertainty using MD as epistemic and SR as aleatoric."""

    name = "HUQ-MD"
    requires = ["MD", "SR"]
    higher_is_uncertain = True

    def score(self, context):
        self.check_requirements(context)
        return total_uncertainty_huq(
            epistemic=context["MD"],
            aleatoric=context["SR"],
            threshold_min=context.get("huq_threshold_min", 0.1),
            threshold_max=context.get("huq_threshold_max", 0.9),
            alpha=context.get("huq_alpha", 0.2),
        )

"""Deterministic softmax-based uncertainty methods."""

from __future__ import annotations

import numpy as np

from uncertainty_benchmark.methods.base import UncertaintyMethod


def entropy_probs(probs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Compute entropy over probability distributions.

    Parameters
    ----------
    probs:
        Array of shape [n_examples, n_classes], or any array where the
        final dimension is the class probability dimension.

    Returns
    -------
    np.ndarray
        Entropy for each item.
    """
    probs = np.asarray(probs, dtype=float)
    return -np.sum(probs * np.log(np.clip(probs, eps, 1.0)), axis=-1)


class SoftmaxResponse(UncertaintyMethod):
    """Softmax response uncertainty.

    SR = 1 - max_c p(y=c | x)

    Larger values mean the model's top class probability is lower.
    """

    name = "SR"
    requires = ["eval_probs"]
    higher_is_uncertain = True

    def score(self, context):
        self.check_requirements(context)
        probs = np.asarray(context["eval_probs"], dtype=float)
        return 1.0 - np.max(probs, axis=-1)


class PredictiveEntropy(UncertaintyMethod):
    """Predictive entropy over one deterministic softmax distribution."""

    name = "ENT"
    requires = ["eval_probs"]
    higher_is_uncertain = True

    def score(self, context):
        self.check_requirements(context)
        probs = np.asarray(context["eval_probs"], dtype=float)
        return entropy_probs(probs)

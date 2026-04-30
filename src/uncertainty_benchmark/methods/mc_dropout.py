"""MC-dropout uncertainty methods."""

from __future__ import annotations

import numpy as np

from uncertainty_benchmark.methods.base import UncertaintyMethod
from uncertainty_benchmark.methods.deterministic import entropy_probs


def sampled_max_probability(sampled_probs: np.ndarray) -> np.ndarray:
    """Sampled maximum probability uncertainty.

    sampled_probs shape:
        [n_examples, n_samples, n_classes]

    SMP = 1 - max_c mean_t p_t(y=c | x)
    """
    sampled_probs = np.asarray(sampled_probs, dtype=float)
    mean_prob = np.mean(sampled_probs, axis=1)
    return 1.0 - np.max(mean_prob, axis=-1)


def probability_variance(sampled_probs: np.ndarray) -> np.ndarray:
    """Probability variance across MC-dropout samples.

    The class-wise variance is summed across classes.
    """
    sampled_probs = np.asarray(sampled_probs, dtype=float)
    mean_prob = np.mean(sampled_probs, axis=1, keepdims=True)
    return ((sampled_probs - mean_prob) ** 2).mean(axis=1).sum(axis=-1)


def bald_score(sampled_probs: np.ndarray) -> np.ndarray:
    """BALD-style mutual information score.

    BALD = H[E[p(y|x,w)]] - E[H[p(y|x,w)]]
    """
    sampled_probs = np.asarray(sampled_probs, dtype=float)
    mean_prob = np.mean(sampled_probs, axis=1)
    predictive_entropy = entropy_probs(mean_prob)
    expected_entropy = np.mean(entropy_probs(sampled_probs), axis=1)
    return predictive_entropy - expected_entropy


def mc_predictive_entropy(sampled_probs: np.ndarray) -> np.ndarray:
    """Entropy of the mean MC-dropout predictive distribution."""
    sampled_probs = np.asarray(sampled_probs, dtype=float)
    mean_prob = np.mean(sampled_probs, axis=1)
    return entropy_probs(mean_prob)


class SampledMaxProbability(UncertaintyMethod):
    name = "SMP"
    requires = ["sampled_probs"]
    higher_is_uncertain = True

    def score(self, context):
        self.check_requirements(context)
        return sampled_max_probability(context["sampled_probs"])


class ProbabilityVariance(UncertaintyMethod):
    name = "PV"
    requires = ["sampled_probs"]
    higher_is_uncertain = True

    def score(self, context):
        self.check_requirements(context)
        return probability_variance(context["sampled_probs"])


class BALD(UncertaintyMethod):
    name = "BALD"
    requires = ["sampled_probs"]
    higher_is_uncertain = True

    def score(self, context):
        self.check_requirements(context)
        return bald_score(context["sampled_probs"])


class MCPredictiveEntropy(UncertaintyMethod):
    name = "ENT_MC"
    requires = ["sampled_probs"]
    higher_is_uncertain = True

    def score(self, context):
        self.check_requirements(context)
        return mc_predictive_entropy(context["sampled_probs"])

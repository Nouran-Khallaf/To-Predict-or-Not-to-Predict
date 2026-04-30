"""Registry of available uncertainty methods."""

from __future__ import annotations

from uncertainty_benchmark.methods.deterministic import (
    PredictiveEntropy,
    SoftmaxResponse,
)
from uncertainty_benchmark.methods.distance import MahalanobisDistance
from uncertainty_benchmark.methods.huq import HUQMahalanobis
from uncertainty_benchmark.methods.mc_dropout import (
    BALD,
    MCPredictiveEntropy,
    ProbabilityVariance,
    SampledMaxProbability,
)
from uncertainty_benchmark.methods.outlier import (
    IsolationForestUncertainty,
    LOFUncertainty,
)


METHOD_REGISTRY = {
    "SR": SoftmaxResponse,
    "ENT": PredictiveEntropy,
    "SMP": SampledMaxProbability,
    "PV": ProbabilityVariance,
    "BALD": BALD,
    "ENT_MC": MCPredictiveEntropy,
    "MD": MahalanobisDistance,
    "HUQ-MD": HUQMahalanobis,
    "LOF": LOFUncertainty,
    "ISOF": IsolationForestUncertainty,
}


def available_methods() -> list[str]:
    return sorted(METHOD_REGISTRY.keys())


def get_method_class(name: str):
    if name not in METHOD_REGISTRY:
        raise KeyError(
            f"Unknown uncertainty method: {name}. "
            f"Available methods: {available_methods()}"
        )
    return METHOD_REGISTRY[name]


def build_method(name: str):
    method_class = get_method_class(name)
    return method_class()


def build_methods(names: list[str]):
    return [build_method(name) for name in names]

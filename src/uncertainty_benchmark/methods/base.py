"""Base interface for uncertainty methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class UncertaintyMethod:
    """Base class for all uncertainty methods.

    Each method receives a context dictionary containing shared objects,
    such as probabilities, sampled probabilities, logits, labels, or embeddings.

    All methods should return a 1D array where larger values mean
    higher uncertainty.
    """

    name: ClassVar[str]
    requires: ClassVar[list[str]]
    higher_is_uncertain: ClassVar[bool] = True

    def score(self, context):
        raise NotImplementedError

    def check_requirements(self, context):
        missing = [key for key in self.requires if key not in context]
        if missing:
            raise KeyError(
                f"{self.name} requires missing context keys: {missing}. "
                f"Available keys: {sorted(context.keys())}"
            )

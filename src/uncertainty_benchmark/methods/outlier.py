"""Embedding/outlier-based uncertainty methods.

Important legacy convention
---------------------------
The original notebook stored raw sklearn ``decision_function`` scores for LOF
and ISOF. These scores are larger for more normal / more in-distribution
examples, so they are confidence-like, not uncertainty-like. The metric suite
flips them exactly once during preprocessing.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from uncertainty_benchmark.methods.base import UncertaintyMethod


class LOFUncertainty(UncertaintyMethod):
    """Raw Local Outlier Factor decision_function score.

    Matches the original notebook:
        lof_scores = lof.decision_function(eval_features)
        eval_results["LOF"] = lof_scores.tolist()

    Larger raw values mean more in-distribution / more confident. They are
    negated in ``metrics.suite`` before min-max normalisation.
    """

    name = "LOF"
    requires = ["train_emb", "eval_emb"]
    higher_is_uncertain = False

    def __init__(self, n_neighbors: int = 20):
        self.n_neighbors = n_neighbors

    def score(self, context):
        self.check_requirements(context)

        train_emb = np.asarray(context["train_emb"], dtype=float)
        eval_emb = np.asarray(context["eval_emb"], dtype=float)

        lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            novelty=True,
        )
        lof.fit(train_emb)

        return lof.decision_function(eval_emb)


class IsolationForestUncertainty(UncertaintyMethod):
    """Raw Isolation Forest decision_function score.

    Matches the original notebook:
        isof_scores = isof.decision_function(eval_features)
        eval_results["ISOF"] = isof_scores.tolist()

    Larger raw values mean more in-distribution / more confident. They are
    negated in ``metrics.suite`` before min-max normalisation.
    """

    name = "ISOF"
    requires = ["train_emb", "eval_emb"]
    higher_is_uncertain = False

    def __init__(self, contamination="auto", random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state

    def score(self, context):
        self.check_requirements(context)

        train_emb = np.asarray(context["train_emb"], dtype=float)
        eval_emb = np.asarray(context["eval_emb"], dtype=float)

        isof = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
        )
        isof.fit(train_emb)

        return isof.decision_function(eval_emb)

import numpy as np

from uncertainty_benchmark.methods.deterministic import (
    PredictiveEntropy,
    SoftmaxResponse,
    entropy_probs,
)
from uncertainty_benchmark.methods.distance import mahalanobis_distance
from uncertainty_benchmark.methods.huq import total_uncertainty_huq
from uncertainty_benchmark.methods.mc_dropout import (
    bald_score,
    probability_variance,
    sampled_max_probability,
)
from uncertainty_benchmark.methods.registry import available_methods, build_method


def test_entropy_probs_shape():
    probs = np.array([[0.9, 0.1], [0.5, 0.5]])
    ent = entropy_probs(probs)
    assert ent.shape == (2,)
    assert ent[1] > ent[0]


def test_softmax_response():
    probs = np.array([[0.9, 0.1], [0.6, 0.4]])
    method = SoftmaxResponse()
    scores = method.score({"eval_probs": probs})
    assert np.allclose(scores, np.array([0.1, 0.4]))


def test_predictive_entropy():
    probs = np.array([[0.9, 0.1], [0.5, 0.5]])
    method = PredictiveEntropy()
    scores = method.score({"eval_probs": probs})
    assert scores[1] > scores[0]


def test_mc_methods_shapes():
    sampled_probs = np.array(
        [
            [[0.9, 0.1], [0.8, 0.2], [0.85, 0.15]],
            [[0.5, 0.5], [0.6, 0.4], [0.4, 0.6]],
        ]
    )

    assert sampled_max_probability(sampled_probs).shape == (2,)
    assert probability_variance(sampled_probs).shape == (2,)
    assert bald_score(sampled_probs).shape == (2,)


def test_mahalanobis_distance_shape():
    train_features = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [2.0, 2.0],
            [2.1, 2.1],
        ]
    )
    train_labels = np.array([0, 0, 1, 1])
    eval_features = np.array([[0.0, 0.1], [3.0, 3.0]])

    scores = mahalanobis_distance(train_features, train_labels, eval_features)
    assert scores.shape == (2,)
    assert scores[1] > scores[0]


def test_huq_shape():
    md = np.array([0.1, 0.5, 0.9])
    sr = np.array([0.2, 0.4, 0.8])
    scores = total_uncertainty_huq(md, sr)
    assert scores.shape == (3,)


def test_registry_contains_all_methods():
    expected = {
        "SR",
        "ENT",
        "SMP",
        "PV",
        "BALD",
        "ENT_MC",
        "MD",
        "HUQ-MD",
        "LOF",
        "ISOF",
    }
    assert expected.issubset(set(available_methods()))

    for name in expected:
        method = build_method(name)
        assert method.name == name

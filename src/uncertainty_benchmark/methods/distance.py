"""Distance-based uncertainty methods."""

from __future__ import annotations

import numpy as np

from uncertainty_benchmark.methods.base import UncertaintyMethod


def compute_centroids(train_features: np.ndarray, train_labels: np.ndarray) -> np.ndarray:
    """Compute one centroid per class."""
    train_features = np.asarray(train_features, dtype=float)
    train_labels = np.asarray(train_labels)

    centroids = []
    for label in np.sort(np.unique(train_labels)):
        centroids.append(train_features[train_labels == label].mean(axis=0))

    return np.asarray(centroids)


def compute_covariance(
    centroids: np.ndarray,
    train_features: np.ndarray,
    train_labels: np.ndarray,
) -> np.ndarray:
    """Compute shared covariance around class centroids."""
    train_features = np.asarray(train_features, dtype=float)
    train_labels = np.asarray(train_labels)

    cov = np.zeros((train_features.shape[1], train_features.shape[1]), dtype=float)

    for class_idx, mu_c in enumerate(centroids):
        class_features = train_features[train_labels == class_idx]
        for x in class_features:
            diff = (x - mu_c)[:, None]
            cov += diff @ diff.T

    return cov / train_features.shape[0]


def mahalanobis_distance(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    eval_features: np.ndarray,
) -> np.ndarray:
    """Minimum Mahalanobis distance to class centroids.

    Larger values indicate that the item is farther from the training
    class distributions and therefore more uncertain.
    """
    train_features = np.asarray(train_features, dtype=float)
    train_labels = np.asarray(train_labels)
    eval_features = np.asarray(eval_features, dtype=float)

    centroids = compute_centroids(train_features, train_labels)
    sigma = compute_covariance(centroids, train_features, train_labels)

    try:
        sigma_inv = np.linalg.inv(sigma)
    except np.linalg.LinAlgError:
        sigma_inv = np.linalg.pinv(sigma)

    diff = eval_features[:, None, :] - centroids[None, :, :]
    dists = np.matmul(np.matmul(diff, sigma_inv), diff.transpose(0, 2, 1))
    dists = np.asarray([np.diag(dist) for dist in dists])

    return np.min(dists, axis=1)


class MahalanobisDistance(UncertaintyMethod):
    """Mahalanobis distance over logits or features."""

    name = "MD"
    requires = ["train_logits", "train_labels", "eval_logits"]
    higher_is_uncertain = True

    def score(self, context):
        self.check_requirements(context)
        return mahalanobis_distance(
            train_features=context["train_logits"],
            train_labels=context["train_labels"],
            eval_features=context["eval_logits"],
        )

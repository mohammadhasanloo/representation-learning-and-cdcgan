"""Classical dimensionality reduction, and the probe that scores every method."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


@dataclass(frozen=True)
class ProbeResult:
    method: str
    latent_dim: int
    n_neighbors: int
    accuracy: float


def knn_probe(
    latent_train: np.ndarray,
    latent_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_neighbors: int,
) -> float:
    """Accuracy of a KNN classifier fitted on one representation.

    Every method in this project is judged the same way: reduce, then ask how
    separable the classes are in the reduced space. It keeps a linear projection
    and a trained encoder on directly comparable footing.
    """
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors).fit(latent_train, y_train)
    return float(accuracy_score(y_test, classifier.predict(latent_test)))


def pca_representation(
    x_train: np.ndarray, x_test: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA on train only, then project both splits with it."""
    pca = PCA(n_components=n_components, random_state=0).fit(x_train)
    return pca.transform(x_train), pca.transform(x_test)


def isomap_representation(
    x_train: np.ndarray, x_test: np.ndarray, n_components: int, n_neighbors: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Isomap embedding.

    Roughly O(n^2 log n), so this is only practical on a subsample; the caller is
    expected to pass one rather than the full 50,000 training images.
    """
    isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors).fit(x_train)
    return isomap.transform(x_train), isomap.transform(x_test)

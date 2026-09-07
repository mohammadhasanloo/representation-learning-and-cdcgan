"""CIFAR-10 loading and normalisation."""

from __future__ import annotations

import math

import numpy as np

IMAGE_SHAPE = (32, 32, 3)
FLAT_IMAGE_SIZE = math.prod(IMAGE_SHAPE)
NUM_CLASSES = 10
CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def load_cifar10() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_train, y_train, x_test, y_test) with images in [0, 1].

    Labels come back as a flat vector rather than the (n, 1) column Keras hands
    out, which otherwise has to be squeezed at every call site.
    """
    from keras.datasets import cifar10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    return (
        x_train.astype("float32") / 255.0,
        y_train.reshape(-1),
        x_test.astype("float32") / 255.0,
        y_test.reshape(-1),
    )


def flatten(images: np.ndarray) -> np.ndarray:
    """(n, 32, 32, 3) -> (n, 3072), for the methods that take vectors."""
    return images.reshape(len(images), -1)

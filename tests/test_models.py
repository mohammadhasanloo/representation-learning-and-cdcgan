"""Shape and wiring tests.

These build every model and push a small random batch through it. They do not
train anything. They check that the architectures are connected correctly and
that the tensor shapes line up end to end.
"""

from __future__ import annotations

import numpy as np
import pytest

from representation_learning.autoencoders import (
    build_convolutional_autoencoder,
    build_convolutional_vae,
    build_dense_autoencoder,
    build_dense_vae,
)
from representation_learning.baselines import knn_probe, pca_representation
from representation_learning.cdcgan import build_discriminator, build_generator
from representation_learning.data import IMAGE_SHAPE, NUM_CLASSES, flatten

BATCH = 4
LATENT = 16


@pytest.fixture
def images():
    return np.random.default_rng(0).random((BATCH, *IMAGE_SHAPE)).astype("float32")


@pytest.mark.parametrize(
    "builder", [build_dense_autoencoder, build_convolutional_autoencoder]
)
def test_autoencoder_round_trips_to_the_input_shape(builder, images):
    encoder, decoder, autoencoder = builder(LATENT)
    assert encoder.predict(images, verbose=0).shape == (BATCH, LATENT)
    assert autoencoder.predict(images, verbose=0).shape == (BATCH, *IMAGE_SHAPE)


@pytest.mark.parametrize("builder", [build_dense_vae, build_convolutional_vae])
def test_vae_encoder_returns_mean_log_var_and_sample(builder, images):
    encoder, decoder, vae = builder(LATENT)
    z_mean, z_log_var, z = encoder.predict(images, verbose=0)
    for tensor in (z_mean, z_log_var, z):
        assert tensor.shape == (BATCH, LATENT)
    assert decoder.predict(z, verbose=0).shape == (BATCH, *IMAGE_SHAPE)


@pytest.mark.parametrize("builder", [build_dense_vae, build_convolutional_vae])
def test_vae_sampling_is_stochastic(builder, images):
    """z is drawn, so two passes must differ; the mean must not."""
    encoder, _, _ = builder(LATENT)
    first_mean, _, first_z = encoder.predict(images, verbose=0)
    second_mean, _, second_z = encoder.predict(images, verbose=0)
    assert np.allclose(first_mean, second_mean)
    assert not np.allclose(first_z, second_z)


def test_generator_is_conditioned_on_the_label():
    generator = build_generator(LATENT)
    noise = np.random.default_rng(1).normal(size=(BATCH, LATENT)).astype("float32")
    labels = np.zeros((BATCH, 1), dtype="int32")
    first = generator.predict([noise, labels], verbose=0)
    second = generator.predict([noise, labels + 1], verbose=0)

    assert first.shape == (BATCH, *IMAGE_SHAPE)
    assert first.min() >= -1.0 and first.max() <= 1.0  # tanh output
    # Same noise, different label must give a different image, or the label is
    # not reaching the generator at all.
    assert not np.allclose(first, second)


def test_discriminator_scores_an_image_label_pair(images):
    discriminator = build_discriminator()
    labels = np.random.default_rng(2).integers(0, NUM_CLASSES, (BATCH, 1)).astype("int32")
    assert discriminator.predict([images, labels], verbose=0).shape == (BATCH, 1)


def test_pca_fits_on_train_only_and_projects_both(images):
    train, test = flatten(images), flatten(images[:2])
    latent_train, latent_test = pca_representation(train, test, n_components=2)
    assert latent_train.shape == (BATCH, 2)
    assert latent_test.shape == (2, 2)


def test_knn_probe_recovers_separable_classes():
    rng = np.random.default_rng(3)
    train = np.vstack([rng.normal(-5, 0.1, (20, 2)), rng.normal(5, 0.1, (20, 2))])
    labels = np.array([0] * 20 + [1] * 20)
    assert knn_probe(train, train, labels, labels, n_neighbors=3) == 1.0

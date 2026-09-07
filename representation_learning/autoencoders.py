"""Autoencoders and variational autoencoders over CIFAR-10.

Each builder returns (encoder, decoder, model). The encoder maps an image to its
latent code, which for the VAEs is the mean of the posterior and serves as the
embedding. The full model is what you train.
"""

from __future__ import annotations

import keras
import tensorflow as tf
from keras import layers, ops

from representation_learning.data import FLAT_IMAGE_SIZE, IMAGE_SHAPE

LEAK = 0.05


def _dense_block(x, units: int, dropout: float):
    x = layers.Dense(units)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=LEAK)(x)
    return layers.Dropout(dropout)(x)


def _conv_block(x, filters: int):
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=LEAK)(x)
    return layers.MaxPooling2D(2, strides=2, padding="same")(x)


def build_dense_autoencoder(latent_dim: int) -> tuple[keras.Model, keras.Model, keras.Model]:
    """Fully connected autoencoder, 1024 -> 512 -> latent and back."""
    encoder_input = keras.Input(shape=IMAGE_SHAPE)
    x = layers.Flatten()(encoder_input)
    x = _dense_block(x, 1024, 0.2)
    x = _dense_block(x, 512, 0.1)
    code = layers.Dense(latent_dim, name="code")(x)
    encoder = keras.Model(encoder_input, code, name="dense_encoder")

    decoder_input = keras.Input(shape=(latent_dim,))
    y = _dense_block(decoder_input, 512, 0.1)
    y = _dense_block(y, 1024, 0.2)
    y = layers.Dense(FLAT_IMAGE_SIZE)(y)
    y = layers.ReLU(max_value=1.0)(y)
    y = layers.Reshape(IMAGE_SHAPE)(y)
    decoder = keras.Model(decoder_input, y, name="dense_decoder")

    autoencoder = keras.Model(encoder_input, decoder(encoder(encoder_input)), name="dense_ae")
    return encoder, decoder, autoencoder


def build_convolutional_autoencoder(
    latent_dim: int,
) -> tuple[keras.Model, keras.Model, keras.Model]:
    """Convolutional autoencoder, three downsampling blocks to a dense code."""
    encoder_input = keras.Input(shape=IMAGE_SHAPE)
    x = encoder_input
    for filters in (16, 32, 64):
        x = _conv_block(x, filters)
    x = layers.Flatten()(x)
    code = layers.Dense(latent_dim, name="code")(x)
    encoder = keras.Model(encoder_input, code, name="conv_encoder")

    decoder_input = keras.Input(shape=(latent_dim,))
    y = layers.Dense(4 * 4 * 64)(decoder_input)
    y = layers.LeakyReLU(negative_slope=LEAK)(y)
    y = layers.Reshape((4, 4, 64))(y)
    for filters in (32, 16):
        y = layers.Conv2DTranspose(filters, 2, strides=2)(y)
        y = layers.Conv2D(filters, 3, padding="same")(y)
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU(negative_slope=LEAK)(y)
    y = layers.Conv2DTranspose(16, 2, strides=2)(y)
    y = layers.Conv2D(3, 3, padding="same")(y)
    y = layers.ReLU(max_value=1.0)(y)
    decoder = keras.Model(decoder_input, y, name="conv_decoder")

    autoencoder = keras.Model(encoder_input, decoder(encoder(encoder_input)), name="conv_ae")
    return encoder, decoder, autoencoder


class Sampling(layers.Layer):
    """Draw z from the posterior via the reparameterisation trick.

    A Layer rather than a Lambda, so the model survives being saved and reloaded:
    a Lambda holding a closure does not serialise.
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = keras.random.normal(shape=ops.shape(z_mean))
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    """Wraps encoder and decoder so the KL term is part of the training step."""

    def __init__(self, encoder: keras.Model, decoder: keras.Model, kl_weight: float = 1.0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = kl_weight
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.loss_tracker, self.reconstruction_tracker, self.kl_tracker]

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

    def _losses(self, images):
        z_mean, z_log_var, z = self.encoder(images)
        reconstruction = self.decoder(z)
        reconstruction_loss = ops.mean(
            ops.sum(ops.square(images - reconstruction), axis=(1, 2, 3))
        )
        kl_loss = ops.mean(
            -0.5 * ops.sum(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=1)
        )
        return reconstruction_loss, kl_loss

    def train_step(self, data):
        images = data[0] if isinstance(data, tuple) else data
        with tf.GradientTape() as tape:
            reconstruction_loss, kl_loss = self._losses(images)
            loss = reconstruction_loss + self.kl_weight * kl_loss
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        return self._report(loss, reconstruction_loss, kl_loss)

    def test_step(self, data):
        images = data[0] if isinstance(data, tuple) else data
        reconstruction_loss, kl_loss = self._losses(images)
        return self._report(reconstruction_loss + self.kl_weight * kl_loss,
                            reconstruction_loss, kl_loss)

    def _report(self, loss, reconstruction_loss, kl_loss):
        self.loss_tracker.update_state(loss)
        self.reconstruction_tracker.update_state(reconstruction_loss)
        self.kl_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}


def _vae_encoder(encoder_input, features, latent_dim: int, name: str) -> keras.Model:
    z_mean = layers.Dense(latent_dim, name="z_mean")(features)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(features)
    z = Sampling(name="z")([z_mean, z_log_var])
    return keras.Model(encoder_input, [z_mean, z_log_var, z], name=name)


def build_dense_vae(latent_dim: int) -> tuple[keras.Model, keras.Model, VAE]:
    """Fully connected VAE, 1024 -> 512 -> 256 before the latent parameters.

    Every normalisation and dropout block follows a dense layer, so each stage
    adds capacity rather than only reshaping the previous one.
    """
    encoder_input = keras.Input(shape=IMAGE_SHAPE)
    x = layers.Flatten()(encoder_input)
    x = _dense_block(x, 1024, 0.2)
    x = _dense_block(x, 512, 0.2)
    x = _dense_block(x, 256, 0.1)
    encoder = _vae_encoder(encoder_input, x, latent_dim, "dense_vae_encoder")

    decoder_input = keras.Input(shape=(latent_dim,))
    y = _dense_block(decoder_input, 256, 0.1)
    y = _dense_block(y, 512, 0.2)
    y = _dense_block(y, 1024, 0.2)
    y = layers.Dense(FLAT_IMAGE_SIZE)(y)
    y = layers.ReLU(max_value=1.0)(y)
    y = layers.Reshape(IMAGE_SHAPE)(y)
    decoder = keras.Model(decoder_input, y, name="dense_vae_decoder")

    return encoder, decoder, VAE(encoder, decoder)


def build_convolutional_vae(latent_dim: int) -> tuple[keras.Model, keras.Model, VAE]:
    """Convolutional VAE, the one the latent traversal figure comes from."""
    encoder_input = keras.Input(shape=IMAGE_SHAPE)
    x = encoder_input
    for filters in (16, 32, 64):
        x = _conv_block(x, filters)
    x = layers.Flatten()(x)
    encoder = _vae_encoder(encoder_input, x, latent_dim, "conv_vae_encoder")

    decoder_input = keras.Input(shape=(latent_dim,))
    y = layers.Dense(4 * 4 * 64)(decoder_input)
    y = layers.LeakyReLU(negative_slope=LEAK)(y)
    y = layers.Reshape((4, 4, 64))(y)
    for filters in (32, 16):
        y = layers.Conv2DTranspose(filters, 2, strides=2)(y)
        y = layers.Conv2D(filters, 3, padding="same")(y)
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU(negative_slope=LEAK)(y)
    y = layers.Conv2DTranspose(16, 2, strides=2)(y)
    y = layers.Conv2D(3, 3, padding="same")(y)
    y = layers.ReLU(max_value=1.0)(y)
    decoder = keras.Model(decoder_input, y, name="conv_vae_decoder")

    return encoder, decoder, VAE(encoder, decoder)

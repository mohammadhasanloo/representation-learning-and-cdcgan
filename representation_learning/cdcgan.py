"""Conditional DCGAN: generator and discriminator both see the class label."""

from __future__ import annotations

import keras
from keras import layers

from representation_learning.data import IMAGE_SHAPE, NUM_CLASSES

LEAK = 0.2


def build_generator(latent_dim: int, num_classes: int = NUM_CLASSES) -> keras.Model:
    """Map (noise, label) to a 32x32 image.

    The label is embedded and concatenated with the noise vector, which is what
    makes the GAN conditional: the same noise gives a different class of image
    depending on the label it is paired with.
    """
    noise = keras.Input(shape=(latent_dim,), name="noise")
    label = keras.Input(shape=(1,), dtype="int32", name="label")

    embedded = layers.Embedding(num_classes, latent_dim)(label)
    embedded = layers.Flatten()(embedded)
    x = layers.Concatenate()([noise, embedded])

    x = layers.Dense(4 * 4 * 256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=LEAK)(x)
    x = layers.Reshape((4, 4, 256))(x)

    for filters in (128, 64):
        x = layers.Conv2DTranspose(filters, 4, strides=2, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(negative_slope=LEAK)(x)

    image = layers.Conv2DTranspose(
        IMAGE_SHAPE[-1], 4, strides=2, padding="same", activation="tanh", name="image"
    )(x)
    return keras.Model([noise, label], image, name="generator")


def build_discriminator(num_classes: int = NUM_CLASSES) -> keras.Model:
    """Score an (image, label) pair.

    The label is broadcast to a full feature map and stacked onto the image as an
    extra channel, so the discriminator can reject an image that is realistic but
    does not match the class it was asked for.
    """
    image = keras.Input(shape=IMAGE_SHAPE, name="image")
    label = keras.Input(shape=(1,), dtype="int32", name="label")

    height, width, _ = IMAGE_SHAPE
    embedded = layers.Embedding(num_classes, height * width)(label)
    embedded = layers.Reshape((height, width, 1))(embedded)
    x = layers.Concatenate()([image, embedded])

    for filters in (64, 128):
        x = layers.Conv2D(filters, 4, strides=2, padding="same")(x)
        x = layers.LeakyReLU(negative_slope=LEAK)(x)
        x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    score = layers.Dense(1, name="score")(x)
    return keras.Model([image, label], score, name="discriminator")

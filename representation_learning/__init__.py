"""Representation learning on CIFAR-10: PCA, Isomap, autoencoders, VAEs, cDCGAN."""

from representation_learning.autoencoders import (
    VAE,
    build_convolutional_autoencoder,
    build_convolutional_vae,
    build_dense_autoencoder,
    build_dense_vae,
)
from representation_learning.baselines import knn_probe, pca_representation
from representation_learning.cdcgan import build_discriminator, build_generator
from representation_learning.data import CLASS_NAMES, IMAGE_SHAPE, load_cifar10

__all__ = [
    "CLASS_NAMES",
    "IMAGE_SHAPE",
    "VAE",
    "build_convolutional_autoencoder",
    "build_convolutional_vae",
    "build_dense_autoencoder",
    "build_dense_vae",
    "build_discriminator",
    "build_generator",
    "knn_probe",
    "load_cifar10",
    "pca_representation",
]

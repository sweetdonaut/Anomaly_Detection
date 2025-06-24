"""
Models Module
=============

Neural network architectures for anomaly detection.
"""

from .compact import CompactAutoencoder, CompactUNetAutoencoder
from .standard_compact import StandardCompactAutoencoder
from .c3k2 import C3k2Autoencoder
from .vae import VariationalAutoencoder, ConditionalVAE
from .difference_denoiser import DifferenceDenoiser

__all__ = [
    'CompactAutoencoder',
    'CompactUNetAutoencoder',
    'StandardCompactAutoencoder',
    'C3k2Autoencoder',
    'VariationalAutoencoder',
    'ConditionalVAE',
    'DifferenceDenoiser'
]
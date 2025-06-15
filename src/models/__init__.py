"""
Models Module
=============

Neural network architectures for anomaly detection.
"""

from .baseline import BaselineAutoencoder
from .enhanced import EnhancedAutoencoder
from .compact import CompactAutoencoder, CompactUNetAutoencoder

__all__ = [
    'BaselineAutoencoder',
    'EnhancedAutoencoder',
    'CompactAutoencoder',
    'CompactUNetAutoencoder'
]
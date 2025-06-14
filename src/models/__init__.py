"""
Models Module
=============

Neural network architectures for anomaly detection.
"""

from .baseline import BaselineAutoencoder
from .enhanced import EnhancedAutoencoder

__all__ = [
    'BaselineAutoencoder',
    'EnhancedAutoencoder'
]
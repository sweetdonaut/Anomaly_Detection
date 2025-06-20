"""
Loss Functions Module
====================

This module contains various loss functions for anomaly detection training.
"""

from .base import BaseLoss
from .mse import MSELoss
from .ssim import SSIMLoss
from .ms_ssim import MultiScaleSSIMLoss  # Using the simplified version for stability
from .sobel import SobelGradientLoss
from .focal_frequency import FocalFrequencyLoss
from .triplet_mse import TripletMSELoss
from .triplet_ssim import TripletSSIMLoss
from .vae_loss import VAELoss, AnnealedVAELoss, CyclicalVAELoss
from .manager import ModularLossManager

__all__ = [
    'BaseLoss',
    'MSELoss',
    'SSIMLoss',
    'MultiScaleSSIMLoss',
    'SobelGradientLoss',
    'FocalFrequencyLoss',
    'TripletMSELoss',
    'TripletSSIMLoss',
    'VAELoss',
    'AnnealedVAELoss',
    'CyclicalVAELoss',
    'ModularLossManager'
]
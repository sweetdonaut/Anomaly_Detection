"""
Mean Squared Error Loss
======================

Simple pixel-wise reconstruction loss.
"""

import torch
import torch.nn.functional as F
from .base import BaseLoss


class MSELoss(BaseLoss):
    """Mean Squared Error Loss"""
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)
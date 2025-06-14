"""
Base Loss Class
===============

Base class for all loss functions with weight support.
"""

import torch
import torch.nn as nn


class BaseLoss(nn.Module):
    """Base class for all loss functions with weight support"""
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate loss between prediction and target
        
        Args:
            pred: Predicted tensor
            target: Target tensor
            
        Returns:
            Loss value
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Apply weight to the calculated loss"""
        return self.weight * self.forward(pred, target)
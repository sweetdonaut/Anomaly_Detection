"""
Modular Loss Manager
===================

Manages multiple loss functions with automatic weight normalization.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any
from .base import BaseLoss


class ModularLossManager(nn.Module):
    """Manages multiple loss functions with automatic weight normalization"""
    def __init__(self, loss_configs: Optional[Dict[str, Dict[str, Any]]] = None, 
                 normalize_weights: bool = True):
        super().__init__()
        self.losses = nn.ModuleDict()
        self.normalize_weights = normalize_weights
        
        if loss_configs:
            self.configure_losses(loss_configs)
    
    def configure_losses(self, loss_configs: Dict[str, Dict[str, Any]]) -> None:
        """Configure losses from dictionary specification"""
        for name, config in loss_configs.items():
            loss_class = config['class']
            weight = config.get('weight', 1.0)
            params = config.get('params', {})
            
            # Create loss instance
            loss_instance = loss_class(weight=weight, **params)
            self.losses[name] = loss_instance
        
        if self.normalize_weights:
            self._normalize_weights()
    
    def add_loss(self, name: str, loss_instance: BaseLoss) -> None:
        """Add a single loss function"""
        self.losses[name] = loss_instance
        if self.normalize_weights:
            self._normalize_weights()
    
    def remove_loss(self, name: str) -> None:
        """Remove a loss function"""
        if name in self.losses:
            del self.losses[name]
            if self.normalize_weights:
                self._normalize_weights()
    
    def _normalize_weights(self) -> None:
        """Normalize all loss weights to sum to 1"""
        total_weight = sum(loss.weight for loss in self.losses.values())
        if total_weight > 0:
            for loss in self.losses.values():
                loss.weight /= total_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate all losses and return detailed results"""
        losses = {}
        total_loss = torch.tensor(0.0, device=pred.device)
        
        for name, loss_fn in self.losses.items():
            loss_value = loss_fn(pred, target)
            losses[name] = loss_value
            total_loss += loss_value
        
        losses['total'] = total_loss
        return losses
    
    def get_weights(self) -> Dict[str, float]:
        """Get current weights for all losses"""
        return {name: loss.weight for name, loss in self.losses.items()}
    
    def set_weights(self, weights: Dict[str, float]) -> None:
        """Set weights for specific losses"""
        for name, weight in weights.items():
            if name in self.losses:
                self.losses[name].weight = weight
        
        if self.normalize_weights:
            self._normalize_weights()
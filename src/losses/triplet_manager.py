"""
Triplet Loss Manager
====================

Loss manager specifically designed for triplet loss functions that return tuples.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Tuple, Union
from .manager import ModularLossManager


class TripletLossManager(ModularLossManager):
    """Loss manager that handles triplet loss functions returning (loss, dict) tuples"""
    
    def forward(self, pred: torch.Tensor, target: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate all losses and return total loss and detailed results
        
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with all loss components
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=pred.device)
        
        for name, loss_fn in self.losses.items():
            result = loss_fn(pred, target)
            
            # Handle triplet losses that return (loss, dict) tuple
            if isinstance(result, tuple):
                loss_value, component_dict = result
                losses[name] = loss_value
                # Add component losses to the output
                for k, v in component_dict.items():
                    if k != 'total' and isinstance(v, torch.Tensor):
                        losses[f"{name}_{k}"] = v
            else:
                # Handle regular losses that return single tensor
                loss_value = result
                losses[name] = loss_value
            
            total_loss += loss_value
        
        losses['total'] = total_loss
        return total_loss, losses
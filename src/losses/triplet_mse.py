"""
Triplet MSE Loss
================

MSE loss for triplet training where the target is reference images
instead of the input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
from .base import BaseLoss


class TripletMSELoss(BaseLoss):
    """
    MSE loss that compares output with reference images
    """
    
    def __call__(self, pred: torch.Tensor, target: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Override to handle tuple return"""
        loss, loss_dict = self.forward(pred, target)
        # Apply weight to main loss
        weighted_loss = self.weight * loss
        # Apply weight to loss dict values
        weighted_dict = {k: self.weight * v if isinstance(v, torch.Tensor) else v 
                        for k, v in loss_dict.items()}
        return weighted_loss, weighted_dict
    
    def __init__(
        self,
        weight: float = 1.0,  # Required by ModularLossManager but not used internally
        reference_mode: str = 'both',  # 'both', 'ref1', 'ref2', 'mean'
        reduction: str = 'mean'
    ):
        """
        Args:
            reference_mode: How to use the two reference images
                - 'both': Calculate loss against both references
                - 'ref1': Use only reference 1
                - 'ref2': Use only reference 2
                - 'mean': Use the mean of both references
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.reference_mode = reference_mode
        self.reduction = reduction
    
    def forward(
        self,
        output: torch.Tensor,
        target: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            output: Model output [B, C, H, W]
            target: For triplet mode, this should be a dict containing references
                   For compatibility, we check the type
        
        Returns:
            loss: The MSE loss
            loss_dict: Dictionary with loss details
        """
        # 檢查是否為 triplet 模式
        if isinstance(target, dict):
            # Triplet 模式：target 是包含 references 的字典
            return self._forward_triplet(output, target)
        else:
            # 相容模式：普通的 MSE loss
            loss = F.mse_loss(output, target, reduction=self.reduction)
            return loss, {'mse': loss}
    
    def _forward_triplet(
        self,
        output: torch.Tensor,
        batch_data: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Triplet mode forward pass
        """
        ref1 = batch_data['reference1']
        ref2 = batch_data['reference2']
        
        if self.reference_mode == 'ref1':
            loss = F.mse_loss(output, ref1, reduction=self.reduction)
            loss_dict = {'mse': loss, 'mse_ref1': loss}
            
        elif self.reference_mode == 'ref2':
            loss = F.mse_loss(output, ref2, reduction=self.reduction)
            loss_dict = {'mse': loss, 'mse_ref2': loss}
            
        elif self.reference_mode == 'mean':
            reference = (ref1 + ref2) / 2.0
            loss = F.mse_loss(output, reference, reduction=self.reduction)
            loss_dict = {'mse': loss, 'mse_mean': loss}
            
        elif self.reference_mode == 'both':
            # 計算對兩個 reference 的 loss
            loss1 = F.mse_loss(output, ref1, reduction=self.reduction)
            loss2 = F.mse_loss(output, ref2, reduction=self.reduction)
            loss = (loss1 + loss2) / 2.0
            
            loss_dict = {
                'mse': loss,
                'mse_ref1': loss1,
                'mse_ref2': loss2
            }
        else:
            raise ValueError(f"Unknown reference_mode: {self.reference_mode}")
        
        return loss, loss_dict
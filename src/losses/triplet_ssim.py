"""
Triplet SSIM Loss
=================

SSIM loss for triplet training where the target is reference images
instead of the input.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union
from .base import BaseLoss
from .ssim import SSIMLoss


class TripletSSIMLoss(BaseLoss):
    """
    SSIM loss that compares output with reference images
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
        window_size: int = 11,
        **ssim_kwargs
    ):
        """
        Args:
            reference_mode: How to use the two reference images
                - 'both': Calculate loss against both references
                - 'ref1': Use only reference 1
                - 'ref2': Use only reference 2
                - 'mean': Use the mean of both references
            window_size: Window size for SSIM calculation
            reduction: 'mean' or 'sum'
            **ssim_kwargs: Additional arguments for SSIMLoss
        """
        super().__init__()
        self.reference_mode = reference_mode
        
        # 使用現有的 SSIM 實作，傳入正確的參數
        self.ssim_loss = SSIMLoss(
            weight=1.0,  # SSIM 內部權重設為 1，外部權重由 LossManager 處理
            window_size=window_size,
            **ssim_kwargs
        )
    
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
            loss: The SSIM loss
            loss_dict: Dictionary with loss details
        """
        # 檢查是否為 triplet 模式
        if isinstance(target, dict):
            # Triplet 模式：target 是包含 references 的字典
            return self._forward_triplet(output, target)
        else:
            # 相容模式：普通的 SSIM loss
            loss, loss_dict = self.ssim_loss(output, target)
            return loss, loss_dict
    
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
            loss = self.ssim_loss(output, ref1)
            loss_dict = {'ssim': loss, 'ssim_ref1': loss}
            
        elif self.reference_mode == 'ref2':
            loss = self.ssim_loss(output, ref2)
            loss_dict = {'ssim': loss, 'ssim_ref2': loss}
            
        elif self.reference_mode == 'mean':
            reference = (ref1 + ref2) / 2.0
            loss = self.ssim_loss(output, reference)
            loss_dict = {'ssim': loss, 'ssim_mean': loss}
            
        elif self.reference_mode == 'both':
            # 計算對兩個 reference 的 loss
            loss1 = self.ssim_loss(output, ref1)
            loss2 = self.ssim_loss(output, ref2)
            loss = (loss1 + loss2) / 2.0
            
            loss_dict = {
                'ssim': loss,
                'ssim_ref1': loss1,
                'ssim_ref2': loss2
            }
        else:
            raise ValueError(f"Unknown reference_mode: {self.reference_mode}")
        
        return loss, loss_dict
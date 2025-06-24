"""
Difference Denoiser Loss Functions
==================================

Loss functions specifically designed for training the dual-output difference denoiser.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from .base import BaseLoss


class DifferenceDenoiserLoss(BaseLoss):
    """
    Combined loss for training the difference denoiser with dual outputs.
    
    Components:
    1. Reconstruction loss - ensures meaningful feature learning
    2. Anomaly regularization - encourages zero output for normal samples
    3. Smoothness regularization - prevents noisy anomaly maps
    """
    
    def __init__(
        self,
        weight: float = 1.0,
        recon_weight: float = 1.0,
        anomaly_weight: float = 0.5,
        smooth_weight: float = 0.01
    ):
        """
        Args:
            weight: Overall weight for this loss (used by loss manager)
            recon_weight: Weight for reconstruction loss
            anomaly_weight: Weight for anomaly regularization
            smooth_weight: Weight for smoothness regularization
        """
        super().__init__(weight)
        self.recon_weight = recon_weight
        self.anomaly_weight = anomaly_weight
        self.smooth_weight = smooth_weight
        
    def compute_reconstruction_loss(
        self,
        reconstructed_diffs: torch.Tensor,
        input_diffs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction loss between input and reconstructed differences.
        
        Args:
            reconstructed_diffs: Reconstructed difference images [B, 3, H, W]
            input_diffs: Original difference images [B, 3, H, W]
            
        Returns:
            Reconstruction loss value
        """
        # Compute individual losses for each difference image
        diff1_loss = F.mse_loss(reconstructed_diffs[:, 0], input_diffs[:, 0])
        diff2_loss = F.mse_loss(reconstructed_diffs[:, 1], input_diffs[:, 1])
        ref_diff_loss = F.mse_loss(reconstructed_diffs[:, 2], input_diffs[:, 2])
        
        # Weighted combination: diff1 and diff2 are more important (40% each)
        # ref_diff is auxiliary (20%)
        weighted_loss = 0.4 * diff1_loss + 0.4 * diff2_loss + 0.2 * ref_diff_loss
        
        return weighted_loss
    
    def compute_anomaly_loss(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly regularization loss.
        For normal samples, the anomaly map should be close to zero.
        
        Args:
            anomaly_map: Predicted anomaly heatmap [B, 1, H, W]
            
        Returns:
            Anomaly regularization loss
        """
        # L2 regularization to encourage zero output
        return torch.mean(anomaly_map ** 2)
    
    def compute_smoothness_loss(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """
        Compute total variation loss for smoothness.
        Prevents the anomaly map from being too noisy.
        
        Args:
            anomaly_map: Predicted anomaly heatmap [B, 1, H, W]
            
        Returns:
            Smoothness loss value
        """
        # Compute gradients
        diff_h = torch.abs(anomaly_map[:, :, 1:, :] - anomaly_map[:, :, :-1, :])
        diff_w = torch.abs(anomaly_map[:, :, :, 1:] - anomaly_map[:, :, :, :-1])
        
        # Total variation
        return torch.mean(diff_h) + torch.mean(diff_w)
    
    def forward(
        self,
        outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the combined loss.
        
        Args:
            outputs: Tuple of (anomaly_map, reconstructed_diffs, input_diffs)
            target: Not used, kept for compatibility with loss manager
            
        Returns:
            Weighted total loss
        """
        anomaly_map, reconstructed_diffs, input_diffs = outputs
        
        # Compute individual losses
        recon_loss = self.compute_reconstruction_loss(reconstructed_diffs, input_diffs)
        anomaly_loss = self.compute_anomaly_loss(anomaly_map)
        smooth_loss = self.compute_smoothness_loss(anomaly_map)
        
        # Combine with weights
        total_loss = (
            self.recon_weight * recon_loss +
            self.anomaly_weight * anomaly_loss +
            self.smooth_weight * smooth_loss
        )
        
        # Apply overall weight
        return self.weight * total_loss


class DifferenceDenoiserDetailedLoss(DifferenceDenoiserLoss):
    """
    Extended version that returns detailed loss components for monitoring.
    """
    
    def forward(
        self,
        outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss and return detailed components.
        
        Returns:
            total_loss: Weighted total loss
            loss_dict: Dictionary with individual loss components
        """
        anomaly_map, reconstructed_diffs, input_diffs = outputs
        
        # Compute individual losses
        recon_loss = self.compute_reconstruction_loss(reconstructed_diffs, input_diffs)
        anomaly_loss = self.compute_anomaly_loss(anomaly_map)
        smooth_loss = self.compute_smoothness_loss(anomaly_map)
        
        # Also compute individual reconstruction losses for monitoring
        diff1_loss = F.mse_loss(reconstructed_diffs[:, 0], input_diffs[:, 0])
        diff2_loss = F.mse_loss(reconstructed_diffs[:, 1], input_diffs[:, 1])
        ref_diff_loss = F.mse_loss(reconstructed_diffs[:, 2], input_diffs[:, 2])
        
        # Combine with weights
        total_loss = (
            self.recon_weight * recon_loss +
            self.anomaly_weight * anomaly_loss +
            self.smooth_weight * smooth_loss
        )
        
        # Apply overall weight
        weighted_total = self.weight * total_loss
        
        # Return detailed components
        loss_dict = {
            'reconstruction': recon_loss,
            'diff1_recon': diff1_loss,
            'diff2_recon': diff2_loss,
            'ref_diff_recon': ref_diff_loss,
            'anomaly': anomaly_loss,
            'smoothness': smooth_loss,
            'total': weighted_total
        }
        
        return weighted_total, loss_dict


class DifferenceDenoiserTripletLoss(nn.Module):
    """
    Wrapper to make DifferenceDenoiserLoss compatible with TripletLossManager.
    Expects a dictionary input containing target, reference1, and reference2.
    """
    
    def __init__(
        self,
        weight: float = 1.0,
        recon_weight: float = 1.0,
        anomaly_weight: float = 0.5,
        smooth_weight: float = 0.01
    ):
        super().__init__()
        self.weight = weight
        self.base_loss = DifferenceDenoiserDetailedLoss(
            weight=1.0,  # We handle weight here
            recon_weight=recon_weight,
            anomaly_weight=anomaly_weight,
            smooth_weight=smooth_weight
        )
    
    def __call__(
        self,
        pred: torch.Tensor,
        batch_data: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Make the loss callable.
        
        Args:
            pred: Model output (anomaly_map, reconstructed_diffs, input_diffs)
            batch_data: Dictionary with 'target', 'reference1', 'reference2'
            
        Returns:
            Tuple of (loss, loss_dict)
        """
        return self.forward(pred, batch_data)
    
    def forward(
        self,
        pred: torch.Tensor,
        batch_data: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass expecting triplet data format.
        
        Args:
            pred: Model output (anomaly_map, reconstructed_diffs, input_diffs)
            batch_data: Dictionary with 'target', 'reference1', 'reference2'
            
        Returns:
            Tuple of (loss, loss_dict)
        """
        # The model output is already a tuple
        if isinstance(pred, tuple):
            outputs = pred
        else:
            raise ValueError("Expected tuple output from DifferenceDenoiser model")
        
        # Compute loss - call forward directly instead of __call__
        total_loss, loss_dict = self.base_loss.forward(outputs, None)
        
        # Apply our weight
        weighted_loss = self.weight * total_loss
        loss_dict['total'] = weighted_loss
        
        return weighted_loss, loss_dict
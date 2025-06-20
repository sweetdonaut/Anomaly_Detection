"""
VAE Loss Functions
==================

Loss functions for Variational Autoencoder training.
Includes reconstruction loss and KL divergence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseLoss


class VAELoss(BaseLoss):
    """Combined VAE loss with reconstruction and KL divergence
    
    Loss = Reconstruction_Loss + β * KL_Divergence
    
    The β parameter controls the weight of KL divergence:
    - β < 1: Focus more on reconstruction quality
    - β = 1: Standard VAE
    - β > 1: β-VAE, encourages more disentangled representations
    """
    
    def __init__(self, reconstruction_loss='mse', beta=1.0, kl_weight=None, weight=1.0):
        """
        Initialize VAE loss
        
        Args:
            reconstruction_loss: Type of reconstruction loss ('mse' or 'bce')
            beta: Weight for KL divergence term (deprecated, use kl_weight)
            kl_weight: Weight for KL divergence term (overrides beta if provided)
            weight: Overall weight for the loss (for compatibility with loss manager)
        """
        super().__init__()
        self.reconstruction_loss = reconstruction_loss
        # Use kl_weight if provided, otherwise fall back to beta
        self.beta = kl_weight if kl_weight is not None else beta
        self.weight = weight  # Store but not used internally
        
        if reconstruction_loss == 'mse':
            self.recon_criterion = nn.MSELoss(reduction='none')
        elif reconstruction_loss == 'bce':
            # For binary cross entropy, outputs should be in [0,1]
            self.recon_criterion = nn.BCELoss(reduction='none')
        else:
            raise ValueError(f"Unknown reconstruction loss: {reconstruction_loss}")
    
    def __call__(self, recon_x, x, mu, logvar):
        """Allow VAELoss to be called with 4 arguments"""
        return self.forward(recon_x, x, mu, logvar)
    
    def forward(self, recon_x, x, mu, logvar):
        """
        Calculate VAE loss
        
        Args:
            recon_x: Reconstructed image
            x: Original image
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        
        Returns:
            Tuple of (total_loss, recon_loss, kl_loss)
        """
        # Reconstruction loss
        if self.reconstruction_loss == 'mse':
            recon_loss = self.recon_criterion(recon_x, x)
            # Average over all dimensions except batch
            recon_loss = recon_loss.view(recon_loss.size(0), -1).mean(dim=1)
        else:  # BCE
            recon_loss = self.recon_criterion(recon_x, x)
            recon_loss = recon_loss.view(recon_loss.size(0), -1).sum(dim=1)
        
        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        # Combine losses
        total_loss = recon_loss + self.beta * kl_loss
        
        # Return mean over batch
        return total_loss.mean(), recon_loss.mean(), kl_loss.mean()
    
    def get_anomaly_score(self, recon_x, x, mu, logvar):
        """
        Calculate anomaly score for each sample
        
        For anomaly detection, we typically use just the reconstruction error
        as the anomaly score, since KL divergence measures how well the
        latent distribution matches the prior (not directly related to anomaly).
        
        Args:
            recon_x: Reconstructed image
            x: Original image
            mu: Mean of latent distribution (not used for anomaly score)
            logvar: Log variance of latent distribution (not used for anomaly score)
        
        Returns:
            Anomaly scores for each sample in the batch
        """
        if self.reconstruction_loss == 'mse':
            # Use MSE as anomaly score
            anomaly_score = F.mse_loss(recon_x, x, reduction='none')
            # Average over spatial dimensions
            anomaly_score = anomaly_score.mean(dim=[1, 2, 3])
        else:  # BCE
            anomaly_score = F.binary_cross_entropy(recon_x, x, reduction='none')
            anomaly_score = anomaly_score.sum(dim=[1, 2, 3])
        
        return anomaly_score


class AnnealedVAELoss(VAELoss):
    """VAE loss with annealed β schedule
    
    Gradually increases β during training to help with training stability.
    Starts with low β (focus on reconstruction) and increases to target β.
    """
    
    def __init__(self, reconstruction_loss='mse', beta_start=0.0, beta_end=1.0, 
                 anneal_steps=10000):
        """
        Initialize annealed VAE loss
        
        Args:
            reconstruction_loss: Type of reconstruction loss
            beta_start: Starting value of β
            beta_end: Final value of β
            anneal_steps: Number of steps to anneal β
        """
        super().__init__(reconstruction_loss, beta_start)
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.anneal_steps = anneal_steps
        self.current_step = 0
    
    def update_beta(self):
        """Update β based on current step"""
        if self.current_step < self.anneal_steps:
            # Linear annealing
            progress = self.current_step / self.anneal_steps
            self.beta = self.beta_start + (self.beta_end - self.beta_start) * progress
        else:
            self.beta = self.beta_end
        
        self.current_step += 1
    
    def forward(self, recon_x, x, mu, logvar):
        """Forward with β annealing"""
        self.update_beta()
        return super().forward(recon_x, x, mu, logvar)


class CyclicalVAELoss(VAELoss):
    """VAE loss with cyclical β schedule
    
    Periodically varies β to prevent posterior collapse and improve
    disentanglement. Based on "Understanding disentangling in β-VAE".
    """
    
    def __init__(self, reconstruction_loss='mse', beta_min=0.0, beta_max=1.0,
                 cycle_length=10000, ratio=0.5):
        """
        Initialize cyclical VAE loss
        
        Args:
            reconstruction_loss: Type of reconstruction loss
            beta_min: Minimum value of β
            beta_max: Maximum value of β
            cycle_length: Length of one cycle
            ratio: Ratio of cycle for increasing β (vs. constant at max)
        """
        super().__init__(reconstruction_loss, beta_min)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.cycle_length = cycle_length
        self.ratio = ratio
        self.current_step = 0
    
    def update_beta(self):
        """Update β based on cyclical schedule"""
        cycle_position = self.current_step % self.cycle_length
        
        if cycle_position < self.cycle_length * self.ratio:
            # Increasing phase
            progress = cycle_position / (self.cycle_length * self.ratio)
            self.beta = self.beta_min + (self.beta_max - self.beta_min) * progress
        else:
            # Constant phase
            self.beta = self.beta_max
        
        self.current_step += 1
    
    def forward(self, recon_x, x, mu, logvar):
        """Forward with cyclical β"""
        self.update_beta()
        return super().forward(recon_x, x, mu, logvar)
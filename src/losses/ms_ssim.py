"""
Multi-Scale SSIM Loss
====================

Multi-scale structural similarity for capturing both fine details and global structure.

This module provides two implementations:
1. MultiScaleSSIMLoss - Standard implementation (recommended)
2. AdvancedMultiScaleSSIMLoss - Advanced version with more configuration options

For most use cases, MultiScaleSSIMLoss is recommended due to its stability and simplicity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from .base import BaseLoss


class AdvancedMultiScaleSSIMLoss(BaseLoss):
    """Advanced Multi-Scale Structural Similarity Index Loss
    
    This is an advanced implementation with more features and configuration options.
    It computes SSIM at multiple scales to capture both fine-grained details and 
    global structure. Features include custom scale weights, different downsampling
    methods, and more precise calculations.
    
    Args:
        weight (float): Overall weight for this loss component. Default: 1.0
        scales (int): Number of scales to use. Default: 3
        scale_weights (Optional[List[float]]): Weights for each scale. 
            If None, uses default weights based on scale count.
        sigma (float): Standard deviation for Gaussian kernel. Default: 1.5
        K1 (float): Stability constant for luminance. Default: 0.01
        K2 (float): Stability constant for contrast. Default: 0.03
        downsample_method (str): Method for downsampling ('avg_pool' or 'gaussian'). 
            Default: 'avg_pool'
    """
    
    def __init__(self, 
                 weight: float = 1.0,
                 scales: int = 3,
                 scale_weights: Optional[List[float]] = None,
                 sigma: float = 1.5,
                 K1: float = 0.01,
                 K2: float = 0.03,
                 downsample_method: str = 'avg_pool'):
        super().__init__(weight)
        
        self.scales = scales
        self.sigma = sigma
        self.K1 = K1
        self.K2 = K2
        self.downsample_method = downsample_method
        
        # Initialize scale weights
        if scale_weights is not None:
            assert len(scale_weights) == scales, \
                f"Number of scale weights ({len(scale_weights)}) must match scales ({scales})"
            self.scale_weights = torch.tensor(scale_weights, dtype=torch.float32)
        else:
            self.scale_weights = self._get_default_weights(scales)
        
        # Normalize weights
        self.scale_weights = self.scale_weights / self.scale_weights.sum()
        
        # Pre-compute window sizes for each scale
        self.window_sizes = [max(5, 11 - 2 * scale) for scale in range(scales)]
        
        # Cache for Gaussian windows
        self._window_cache = {}
    
    def _get_default_weights(self, scales: int) -> torch.Tensor:
        """Generate default weights for different numbers of scales
        
        These weights are based on empirical studies showing human visual
        system's sensitivity to different frequency bands.
        """
        weight_dict = {
            2: [0.4, 0.6],
            3: [0.0448, 0.2856, 0.3001],
            4: [0.0244, 0.1638, 0.2418, 0.3148],
            5: [0.0131, 0.0903, 0.1589, 0.2515, 0.3141]
        }
        
        if scales in weight_dict:
            return torch.tensor(weight_dict[scales], dtype=torch.float32)
        else:
            # For custom scale counts, use exponentially increasing weights
            weights = torch.exp(torch.linspace(0, 2, scales))
            return weights / weights.sum()
    
    def _create_gaussian_kernel(self, window_size: int, sigma: float) -> torch.Tensor:
        """Create 1D Gaussian kernel"""
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= window_size // 2
        
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        
        return g
    
    def _get_gaussian_window(self, window_size: int, channel: int, 
                           device: torch.device) -> torch.Tensor:
        """Get or create Gaussian window for convolution
        
        Uses caching to avoid recreating windows repeatedly.
        """
        cache_key = (window_size, channel, device)
        
        if cache_key not in self._window_cache:
            # Create 1D Gaussian
            gaussian_1d = self._create_gaussian_kernel(window_size, self.sigma)
            
            # Create 2D Gaussian window
            gaussian_2d = gaussian_1d.unsqueeze(1) @ gaussian_1d.unsqueeze(0)
            gaussian_2d = gaussian_2d.unsqueeze(0).unsqueeze(0)
            
            # Expand to match channel count
            window = gaussian_2d.expand(channel, 1, window_size, window_size)
            window = window.to(device)
            
            self._window_cache[cache_key] = window
        
        return self._window_cache[cache_key]
    
    def _compute_ssim_components(self, x: torch.Tensor, y: torch.Tensor, 
                                window_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute SSIM components: luminance * contrast_structure and contrast_structure
        
        Returns:
            Tuple of (l * cs, cs) where l is luminance and cs is contrast-structure
        """
        channel = x.size(1)
        window = self._get_gaussian_window(window_size, channel, x.device)
        
        # Constants for stability
        C1 = (self.K1 ** 2)
        C2 = (self.K2 ** 2)
        
        # Compute local means
        mu1 = F.conv2d(x, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(y, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Compute local variances and covariance
        sigma1_sq = F.conv2d(x * x, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(y * y, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(x * y, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        # Compute SSIM components
        luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
        contrast_structure = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        
        # Return mean values across spatial dimensions
        # Keep batch dimension, only average over spatial dimensions (H, W)
        ssim = (luminance * contrast_structure).mean(dim=[2, 3])
        cs = contrast_structure.mean(dim=[2, 3])
        
        # If single channel, squeeze the channel dimension
        if ssim.dim() == 2 and ssim.size(1) == 1:
            ssim = ssim.squeeze(1)
        if cs.dim() == 2 and cs.size(1) == 1:
            cs = cs.squeeze(1)
            
        return ssim, cs
    
    def _downsample(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample image by factor of 2
        
        Uses either average pooling or Gaussian filtering based on configuration.
        """
        if self.downsample_method == 'avg_pool':
            return F.avg_pool2d(x, kernel_size=2, stride=2)
        elif self.downsample_method == 'gaussian':
            # Apply Gaussian blur before downsampling
            kernel_size = 5
            sigma = 1.0
            channel = x.size(1)
            
            # Create Gaussian kernel
            gaussian_1d = self._create_gaussian_kernel(kernel_size, sigma)
            gaussian_2d = gaussian_1d.unsqueeze(1) @ gaussian_1d.unsqueeze(0)
            gaussian_kernel = gaussian_2d.unsqueeze(0).unsqueeze(0)
            gaussian_kernel = gaussian_kernel.expand(channel, 1, kernel_size, kernel_size)
            gaussian_kernel = gaussian_kernel.to(x.device)
            
            # Apply Gaussian blur
            x_blurred = F.conv2d(x, gaussian_kernel, padding=kernel_size//2, groups=channel)
            
            # Downsample
            return x_blurred[:, :, ::2, ::2]
        else:
            raise ValueError(f"Unknown downsample method: {self.downsample_method}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Multi-Scale SSIM Loss
        
        Args:
            pred: Predicted image tensor of shape (N, C, H, W)
            target: Target image tensor of shape (N, C, H, W)
            
        Returns:
            MS-SSIM loss value (1 - MS-SSIM)
        """
        device = pred.device
        
        # Move scale weights to correct device if needed
        if self.scale_weights.device != device:
            self.scale_weights = self.scale_weights.to(device)
        
        # Lists to store values at each scale
        mcs_values = []  # Contrast-structure values
        final_ssim = None  # Full SSIM at finest scale
        
        # Process each scale
        for scale in range(self.scales):
            # Use appropriate window size for this scale
            window_size = self.window_sizes[scale]
            
            # Compute SSIM components
            if scale == self.scales - 1:
                # At finest scale, we need the full SSIM
                ssim_val, _ = self._compute_ssim_components(pred, target, window_size)
                final_ssim = ssim_val
            else:
                # At other scales, we only need contrast-structure
                _, cs_val = self._compute_ssim_components(pred, target, window_size)
                mcs_values.append(cs_val)
            
            # Downsample for next scale (except at last scale)
            if scale < self.scales - 1:
                pred = self._downsample(pred)
                target = self._downsample(target)
                
                # Check if images are getting too small
                if pred.size(-1) < window_size or pred.size(-2) < window_size:
                    # If images are too small, stop early
                    break
        
        # Compute MS-SSIM
        if len(mcs_values) == 0:
            # Only one scale was used
            ms_ssim = final_ssim
        else:
            # Combine contrast-structure values with final SSIM
            mcs_stack = torch.stack(mcs_values)
            
            # Calculate actual number of scales used
            num_scales_used = len(mcs_values) + 1  # +1 for final SSIM
            
            # Normalize weights for actual scales used
            if num_scales_used < self.scales:
                # Redistribute weights proportionally
                weights_to_use = self.scale_weights[:num_scales_used].clone()
                weights_to_use = weights_to_use / weights_to_use.sum()
            else:
                weights_to_use = self.scale_weights
            
            # Apply weights to contrast-structure values
            # mcs_stack shape: [num_scales-1, batch_size]
            # weights shape: [num_scales-1] -> reshape to [num_scales-1, 1] for broadcasting
            cs_weights = weights_to_use[:len(mcs_values)].view(-1, 1)
            
            # Debug shapes
            if mcs_stack.dim() == 3:
                # If mcs_stack has an extra channel dimension, squeeze it
                mcs_stack = mcs_stack.squeeze(2)
            
            mcs_weighted = torch.prod(
                torch.pow(mcs_stack, cs_weights),
                dim=0  # Product over scales, keep batch dimension
            )
            
            # Combine with final SSIM
            final_weight = weights_to_use[-1]
            ms_ssim = mcs_weighted * (final_ssim ** final_weight)
        
        # Return loss (1 - MS-SSIM)
        # Average over batch dimension to get scalar loss
        return torch.mean(1.0 - ms_ssim)
    
    def __repr__(self) -> str:
        """String representation for debugging"""
        return (f"MultiScaleSSIMLoss(weight={self.weight}, scales={self.scales}, "
                f"scale_weights={self.scale_weights.tolist()}, "
                f"downsample_method={self.downsample_method})")


class MultiScaleSSIMLoss(BaseLoss):
    """Multi-Scale SSIM Loss
    
    Standard implementation of Multi-Scale Structural Similarity Index Loss.
    This version is optimized for stability and ease of use, making it the 
    recommended choice for most applications.
    
    Args:
        num_scales (int): Number of scales to use (default: 3)
        window_size (int): Size of the Gaussian window (default: 11)
        weight (float): Weight for this loss component
    """
    
    def __init__(self, num_scales: int = 3, window_size: int = 11, 
                 weight: float = 1.0, **kwargs):
        super().__init__(weight)
        self.window_size = window_size
        self.num_scales = num_scales
        # Accept 'scales' parameter for backward compatibility
        if 'scales' in kwargs:
            self.num_scales = kwargs['scales']
        
        # Fixed weights for each scale
        if self.num_scales == 1:
            self.scale_weights = [1.0]
        elif self.num_scales == 2:
            self.scale_weights = [0.6, 0.4]
        elif self.num_scales == 3:
            self.scale_weights = [0.5, 0.3, 0.2]
        else:
            # For more scales, distribute weights evenly
            self.scale_weights = [1.0 / self.num_scales] * self.num_scales
        
        # Constants for SSIM
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        
        # Create Gaussian window once
        self.register_buffer('window', self._create_window(window_size))
    
    def _create_window(self, window_size):
        """Create Gaussian window for SSIM calculation"""
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= window_size // 2
        g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(1) @ g.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0)
        return window
    
    def _ssim(self, x, y, window_size=None):
        """Calculate SSIM between two images"""
        if window_size is None:
            window_size = self.window_size
            
        # Use pre-created window
        window = self.window.to(x.device)
        
        # Calculate means
        mu1 = F.conv2d(x, window, padding=window_size//2)
        mu2 = F.conv2d(y, window, padding=window_size//2)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Calculate variances and covariance
        sigma1_sq = F.conv2d(x * x, window, padding=window_size//2) - mu1_sq
        sigma2_sq = F.conv2d(y * y, window, padding=window_size//2) - mu2_sq
        sigma12 = F.conv2d(x * y, window, padding=window_size//2) - mu1_mu2
        
        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))
        
        # Return mean SSIM
        return ssim_map.mean()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate MS-SSIM loss"""
        # Ensure inputs are 4D (B, C, H, W)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
            target = target.unsqueeze(1)
        
        # Calculate MS-SSIM
        msssim_val = 0.0
        
        for scale in range(self.num_scales):
            # Check if image is large enough for this scale
            if pred.size(-1) < self.window_size or pred.size(-2) < self.window_size:
                # If too small, use remaining weight on current scale
                remaining_weight = sum(self.scale_weights[scale:])
                ssim_val = self._ssim(pred, target)
                msssim_val += remaining_weight * ssim_val
                break
            
            # Calculate SSIM for this scale
            ssim_val = self._ssim(pred, target)
            msssim_val += self.scale_weights[scale] * ssim_val
            
            # Downsample for next scale (except last scale)
            if scale < self.num_scales - 1:
                pred = F.avg_pool2d(pred, kernel_size=2, stride=2)
                target = F.avg_pool2d(target, kernel_size=2, stride=2)
        
        # Return loss (1 - MS-SSIM)
        return 1.0 - msssim_val
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"num_scales={self.num_scales}, "
                f"window_size={self.window_size})")


# For backward compatibility, you can still import the old name
SimpleMultiScaleSSIMLoss = AdvancedMultiScaleSSIMLoss
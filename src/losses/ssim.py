"""
Structural Similarity Index Loss
================================

SSIM loss implementation for perceptual image quality assessment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseLoss


class SSIMLoss(BaseLoss):
    """Structural Similarity Index Loss
    
    This class implements the Structural Similarity Index (SSIM) loss proposed by Wang et al. (2004).
    SSIM evaluates image quality by comprehensively assessing similarity across three dimensions:
    luminance, contrast, and structure. Compared to traditional mean squared error loss,
    SSIM better aligns with human visual perception characteristics.
    
    This implementation follows the modular loss function framework design specifications,
    ensuring seamless integration with other loss functions. Loss values from all batch
    samples are automatically aggregated into a single scalar value.
    
    Args:
        weight (float): Weight coefficient of this loss function in the total loss. Default: 1.0
        window_size (int): Size of the Gaussian window, must be odd. Recommend using 11 or larger
                          for stable results. Default: 11
        sigma (float): Standard deviation of the Gaussian kernel, controls the distribution of
                      window weights. Larger values produce smoother weight distributions. Default: 1.5
    """
    
    def __init__(self, 
                 weight: float = 1.0, 
                 window_size: int = 11, 
                 sigma: float = 1.5):
        super().__init__(weight)
        
        # Parameter validation to ensure reasonable input values
        if window_size % 2 == 0:
            raise ValueError(f"Window size must be odd, got {window_size}")
        if window_size < 3:
            raise ValueError(f"Window size must be at least 3, got {window_size}")
        if sigma <= 0:
            raise ValueError(f"Sigma must be positive, got {sigma}")
        
        self.window_size = window_size
        self.sigma = sigma
        self.channel = None  # Will be dynamically set based on input
        
        # Pre-create single-channel Gaussian window and register as buffer
        # Using register_buffer ensures the window is properly handled during
        # model saving, loading, and device transfers, avoiding manual management
        initial_window = self._create_window(window_size, 1)
        self.register_buffer('window', initial_window, persistent=False)
        
    def _gaussian_1d(self, window_size: int, sigma: float) -> torch.Tensor:
        """Generate 1D Gaussian kernel
        
        Creates a normalized 1D kernel based on the Gaussian distribution formula.
        This kernel will be used to construct the 2D Gaussian window.
        Normalization ensures the weights sum to 1.
        
        Args:
            window_size: Size of the kernel
            sigma: Standard deviation of the Gaussian distribution
            
        Returns:
            torch.Tensor: Normalized 1D Gaussian kernel
        """
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= window_size // 2
        
        gaussian = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return gaussian / gaussian.sum()
    
    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        """Create 2D Gaussian window
        
        Generates a 2D Gaussian window through outer product of 1D Gaussian kernels.
        This method leverages the separability of Gaussian functions to improve
        computational efficiency. The generated window is expanded to the specified
        number of channels as needed.
        
        Args:
            window_size: Size of the window (same for height and width)
            channel: Number of channels needed
            
        Returns:
            torch.Tensor: Gaussian window with shape (channel, 1, window_size, window_size)
        """
        gaussian_1d = self._gaussian_1d(window_size, self.sigma)
        gaussian_2d = gaussian_1d.unsqueeze(1) @ gaussian_1d.unsqueeze(0)
        gaussian_2d = gaussian_2d.unsqueeze(0).unsqueeze(0)
        
        # Expand to specified number of channels, using same weights for each channel
        window = gaussian_2d.expand(channel, 1, window_size, window_size)
        return window.contiguous()
    
    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate structural similarity index
        
        Implements the core computation logic of SSIM. This method calculates local
        statistics of the input image pairs and applies the SSIM formula to evaluate
        similarity. The computation includes estimation of local means, variances,
        and covariances, which correspond to similarities in luminance, contrast,
        and structure respectively.
        
        Args:
            x: First image tensor with shape [N, C, H, W]
            y: Second image tensor with shape [N, C, H, W]
            
        Returns:
            torch.Tensor: SSIM values for each batch sample, shape [N], range [0, 1]
        """
        # SSIM stability constants based on recommendations from the original paper
        # These constants prevent division by zero and control relative importance of components
        C1 = (0.01 ** 2)
        C2 = (0.03 ** 2)
        
        # Calculate local means using Gaussian weighting
        # groups parameter ensures each channel is convolved independently
        mu_x = F.conv2d(x, self.window, padding=self.window_size//2, groups=self.channel)
        mu_y = F.conv2d(y, self.window, padding=self.window_size//2, groups=self.channel)
        
        # Calculate squares and products of means for subsequent calculations
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        # Calculate local variances and covariance
        # Using formula: Var(X) = E[X²] - E[X]²
        sigma_x_sq = F.conv2d(x ** 2, self.window, padding=self.window_size//2, groups=self.channel) - mu_x_sq
        sigma_y_sq = F.conv2d(y ** 2, self.window, padding=self.window_size//2, groups=self.channel) - mu_y_sq
        sigma_xy = F.conv2d(x * y, self.window, padding=self.window_size//2, groups=self.channel) - mu_xy
        
        # Apply SSIM formula
        # SSIM = (2μxμy + C1)(2σxy + C2) / (μx² + μy² + C1)(σx² + σy² + C2)
        luminance_contrast = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
        
        ssim_map = luminance_contrast / denominator
        
        # Average over spatial and channel dimensions for each sample, keeping batch dimension
        # This ensures each sample in the batch gets independent gradient signals
        return ssim_map.mean(dim=[1, 2, 3])
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate SSIM loss
        
        This method is the main interface of the loss function, responsible for
        calculating SSIM loss between predicted and target images. Loss is defined
        as 1 - SSIM, ensuring loss is 0 for perfect match and approaches 1 for
        complete mismatch.
        
        Args:
            pred: Predicted image tensor, shape must be [N, C, H, W]
            target: Target image tensor, shape must be [N, C, H, W]
            
        Returns:
            torch.Tensor: Scalar loss value representing average loss for the batch
            
        Raises:
            ValueError: When input tensors are not 4D or shapes don't match
        """
        # Input validation to ensure tensor format is correct
        if pred.dim() != 4 or target.dim() != 4:
            raise ValueError(f"Expected 4D tensors, got {pred.dim()}D and {target.dim()}D")
        
        if pred.shape != target.shape:
            raise ValueError(f"Input shapes must match, got {pred.shape} and {target.shape}")
        
        # Dynamically update Gaussian window to match input channel count
        channel = pred.size(1)
        if self.channel != channel:
            self.channel = channel
            new_window = self._create_window(self.window_size, channel)
            self.window = new_window.to(device=pred.device, dtype=pred.dtype)
        
        # Ensure window is on same device and data type as input tensors
        if self.window.device != pred.device or self.window.dtype != pred.dtype:
            self.window = self.window.to(device=pred.device, dtype=pred.dtype)
        
        # Calculate SSIM values and convert to loss
        ssim_values = self._ssim(pred, target)
        loss = 1.0 - ssim_values
        
        # Return batch average loss (scalar), ensuring interface consistency with other loss functions
        return loss.mean()
    
    def compute_per_sample_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate individual loss values for each sample
        
        This helper method provides a way to obtain detailed loss information for
        each sample in the batch, suitable for analysis, debugging, or scenarios
        requiring sample-level loss information. This method does not affect the
        main training process.
        
        Args:
            pred: Predicted image tensor with shape [N, C, H, W]
            target: Target image tensor with shape [N, C, H, W]
            
        Returns:
            torch.Tensor: Loss values for each sample, shape [N]
        """
        with torch.no_grad():
            # Ensure device and window settings are correct
            if pred.size(1) != self.channel:
                self.forward(pred, target)  # Trigger window update
            
            ssim_values = self._ssim(pred, target)
            return 1.0 - ssim_values
    
    def __repr__(self) -> str:
        """Provide string representation of the class
        
        Returns a descriptive string containing all key parameters,
        useful for debugging and logging.
        """
        return (f"{self.__class__.__name__}("
                f"weight={self.weight}, "
                f"window_size={self.window_size}, "
                f"sigma={self.sigma})")
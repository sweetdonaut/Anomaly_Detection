import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import cv2
from typing import Dict, Optional, Any

# ==================== Loss Functions Module ====================
# ==================== Base Loss Class ====================
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
    
# ==================== Individual Loss Implementations ====================
class MSELoss(BaseLoss):
    """Mean Squared Error Loss"""
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)

class SSIMLoss(BaseLoss):
    """Structural Similarity Index Loss"""
    def __init__(self, weight: float = 1.0, window_size: int = 11, size_average: bool = True):
        super().__init__(weight)
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)
        
    def _gaussian(self, window_size: int, sigma: float = 1.5) -> torch.Tensor:
        """Create 1D Gaussian kernel"""
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2 / (2 * sigma**2)) 
                             for x in range(window_size)])
        return gauss / gauss.sum()
    
    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        """Create 2D Gaussian window"""
        _1D_window = self._gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate SSIM loss"""
        # Ensure window is on the same device
        if pred.is_cuda:
            self.window = self.window.cuda(pred.get_device())
        self.window = self.window.type_as(pred)
        
        # Update window if channel count changes
        channel = pred.size(1)
        if channel != self.channel:
            self.window = self._create_window(self.window_size, channel)
            self.channel = channel
            if pred.is_cuda:
                self.window = self.window.cuda(pred.get_device())
            self.window = self.window.type_as(pred)
        
        return 1 - self._ssim(pred, target)
    
    def _ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Calculate SSIM"""
        C1 = 0.01**2
        C2 = 0.03**2
        
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, self.window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, self.window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, self.window, padding=self.window_size//2, groups=self.channel) - mu1_mu2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

class SobelGradientLoss(BaseLoss):
    """Sobel gradient loss for edge preservation"""
    def __init__(self, weight: float = 1.0):
        super().__init__(weight)
        # Register Sobel kernels as buffers
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate Sobel gradient loss"""
        # Expand kernels to match input channels
        batch_size, channels = pred.shape[:2]
        sobel_x = self.sobel_x.repeat(channels, 1, 1, 1)
        sobel_y = self.sobel_y.repeat(channels, 1, 1, 1)
        
        # Calculate gradients for prediction
        pred_grad_x = F.conv2d(pred, sobel_x, padding=1, groups=channels)
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1, groups=channels)
        pred_grad = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8)
        
        # Calculate gradients for target
        target_grad_x = F.conv2d(target, sobel_x, padding=1, groups=channels)
        target_grad_y = F.conv2d(target, sobel_y, padding=1, groups=channels)
        target_grad = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)
        
        return F.mse_loss(pred_grad, target_grad)
    
# ==================== Focal Frequency Loss (Official Implementation) ====================
# version adaptation for PyTorch > 1.7.1
IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft

class FocalFrequencyLoss(nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.

    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>

    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        # stack to patch tensor
        y = torch.stack(patch_list, 1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        if IS_HIGH_VERSION:
            freq = torch.fft.fft2(y, norm='ortho')
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(y, 2, onesided=False, normalized=True)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None, **kwargs):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight

# ==================== Modular Loss Manager ====================
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
 
# ==================== Usage Examples ====================
def create_default_loss_manager() -> ModularLossManager:
    """Create a loss manager with default configuration"""
    loss_config = {
        'mse': {
            'class': MSELoss,
            'weight': 0.3
        },
        'ssim': {
            'class': SSIMLoss,
            'weight': 0.3,
            'params': {'window_size': 11}
        },
        'focal_freq': {
            'class': FocalFrequencyLoss,
            'weight': 0.2,
            'params': {'alpha': 1.0, 'patch_factor': 1}
        },
        'sobel': {
            'class': SobelGradientLoss,
            'weight': 0.2
        }
    }
    
    return ModularLossManager(loss_config, normalize_weights=True)

def create_loss_config_from_flags(use_mse=True, mse_weight=0.3,
                                  use_ssim=True, ssim_weight=0.3,
                                  use_focal_freq=True, focal_freq_weight=0.2,
                                  use_sobel=True, sobel_weight=0.2,
                                  **kwargs):
    """Create loss configuration from boolean flags (for backward compatibility)"""
    loss_config = {}
    
    if use_mse:
        loss_config['mse'] = {
            'class': MSELoss,
            'weight': mse_weight
        }
    
    if use_ssim:
        loss_config['ssim'] = {
            'class': SSIMLoss,
            'weight': ssim_weight,
            'params': kwargs.get('ssim_params', {'window_size': 11})
        }
    
    if use_focal_freq:
        loss_config['focal_freq'] = {
            'class': FocalFrequencyLoss,
            'weight': focal_freq_weight,
            'params': kwargs.get('focal_freq_params', {'alpha': 1.0, 'patch_factor': 1})
        }
    
    if use_sobel:
        loss_config['sobel'] = {
            'class': SobelGradientLoss,
            'weight': sobel_weight
        }
    
    return loss_config           
            
# ==================== Synthetic Anomaly Generation ====================
class SyntheticAnomalyGenerator:
    """Generate synthetic anomalies for training - bright/dark spots only"""
    def __init__(self, anomaly_prob=0.3):
        self.anomaly_prob = anomaly_prob
    
    def generate_anomaly(self, image):
        """Generate synthetic bright or dark spot anomaly"""
        if random.random() > self.anomaly_prob:
            return image, torch.zeros_like(image)
        
        # Clone image to avoid modifying original
        anomaly_image = image.clone()
        mask = torch.zeros_like(image)
        
        # Generate bright or dark spot
        anomaly_image, mask = self._generate_spot_anomaly(anomaly_image)
        
        return anomaly_image, mask
    
    def _generate_spot_anomaly(self, image):
        """Generate circular/elliptical bright or dark spots"""
        B, C, H, W = image.shape
        mask = torch.zeros_like(image)
        
        for b in range(B):
            # Spot size around 10x10 pixels with some variation
            base_size = 10
            size_variation = random.uniform(0.7, 1.3)  # 70% to 130% of base size
            spot_h = int(base_size * size_variation * random.uniform(0.8, 1.2))  # Elliptical variation
            spot_w = int(base_size * size_variation * random.uniform(0.8, 1.2))
            
            # Ensure minimum size
            spot_h = max(6, min(spot_h, 15))
            spot_w = max(6, min(spot_w, 15))
            
            # Random position (ensure spot fits within image)
            y = random.randint(spot_h//2, H - spot_h//2 - 1)
            x = random.randint(spot_w//2, W - spot_w//2 - 1)
            
            # Create elliptical mask
            y_grid, x_grid = torch.meshgrid(
                torch.arange(spot_h, dtype=torch.float32) - spot_h/2,
                torch.arange(spot_w, dtype=torch.float32) - spot_w/2,
                indexing='ij'
            )
            
            # Elliptical distance
            ellipse_mask = ((x_grid / (spot_w/2))**2 + (y_grid / (spot_h/2))**2) <= 1
            ellipse_mask = ellipse_mask.float()
            
            # Smooth edges with Gaussian-like falloff
            distance = torch.sqrt((x_grid / (spot_w/2))**2 + (y_grid / (spot_h/2))**2)
            smooth_mask = torch.exp(-2 * torch.clamp(distance - 0.8, min=0))
            smooth_mask = smooth_mask * ellipse_mask
            smooth_mask = smooth_mask / smooth_mask.max() if smooth_mask.max() > 0 else smooth_mask
            
            # Decide if bright or dark spot
            is_bright = random.random() > 0.5
            
            # Calculate spot intensity (adjusted for normalized images)
            if is_bright:
                # Bright spot: increase pixel values
                intensity = random.uniform(0.2, 0.4)  # Reduced intensity for normalized images
            else:
                # Dark spot: decrease pixel values
                intensity = random.uniform(-0.4, -0.2)  # Reduced intensity for normalized images
            
            # Apply spot to image
            y_start = y - spot_h//2
            x_start = x - spot_w//2
            y_end = y_start + spot_h
            x_end = x_start + spot_w
            
            # Apply smooth intensity change
            for c in range(C):
                region = image[b, c, y_start:y_end, x_start:x_end]
                image[b, c, y_start:y_end, x_start:x_end] = torch.clamp(
                    region + intensity * smooth_mask.to(image.device),
                    -1, 1  # Assuming normalized images
                )
            
            # Binary mask for evaluation
            mask[b, :, y_start:y_end, x_start:x_end] = (ellipse_mask > 0).float()
        
        return image, mask

# ==================== Network Architectures ====================
class BaselineAutoencoder(nn.Module):
    """Standard autoencoder without skip connections"""
    def __init__(self, latent_dim=128, input_size=(976, 176)):
        super().__init__()
        self.input_size = input_size
        
        # Encoder with 3x3 kernels and SiLU activation
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),  # 976x176 -> 488x88
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),  # 488x88 -> 244x44
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), # 244x44 -> 122x22
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1), # 122x22 -> 61x11
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 512, 3, 2, 0), # 61x11 -> 30x5 (no decimals)
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True),
        )
        
        # Calculate encoder output size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *input_size)
            encoder_output = self.encoder(dummy_input)
            self.encoder_output_size = encoder_output.shape[2:]
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, latent_dim, 1),
            nn.BatchNorm2d(latent_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(latent_dim, 512, 1),
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True),
        )
        
        # Decoder with matching kernels for proper upsampling
        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(512, 256, 3, 2, 0, output_padding=1),  # 30x5 -> 61x11
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 3, 2, 1, output_padding=1),
            nn.Sigmoid()
        ])
    
    def forward(self, x):
        # Store original size
        original_size = x.shape[2:]
        
        # Encode
        encoded = self.encoder(x)
        # Bottleneck
        bottleneck = self.bottleneck(encoded)
        
        # Decode with proper layer handling
        x = bottleneck
        for i, layer in enumerate(self.decoder):
            x = layer(x)
        
        # Resize to match original input size if needed
        if x.shape[2:] != original_size:
            x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        
        return x
    
    def get_latent_features(self, x):
        """Extract latent space features"""
        encoded = self.encoder(x)
        return self.bottleneck(encoded)

class EnhancedAutoencoder(nn.Module):
    """Autoencoder with U-Net style skip connections"""
    def __init__(self):
        super().__init__()
        # Encoder blocks
        self.enc1 = self._conv_block(1, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)
        self.enc4 = self._conv_block(128, 256)
        self.enc5 = self._conv_block(256, 512)
        
        # Final encoder layer to match BaselineAutoencoder
        self.enc_final = nn.Sequential(
            nn.Conv2d(512, 512, 3, 2, 0),  # 61x11 -> 30x5 (no decimals)
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 512, 1),
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True)
        )
        
        # Decoder blocks with skip connections
        self.dec5 = self._conv_block(512 + 512, 256)  # Skip from enc5
        self.dec4 = self._conv_block(256 + 256, 128)  # Skip from enc4
        self.dec3 = self._conv_block(128 + 128, 64)   # Skip from enc3
        self.dec2 = self._conv_block(64 + 64, 32)     # Skip from enc2
        self.dec1 = self._conv_block(32 + 32, 32)     # Skip from enc1
        
        # Precise upsampling layer for bottleneck
        self.bottleneck_upsample = nn.ConvTranspose2d(512, 512, 3, 2, 0, output_padding=1)  # 30x5 -> 61x11
        
        self.final = nn.Conv2d(32, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoding with feature storage
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        
        # Final encoding step
        e_final = self.enc_final(e5)  # 61x11 -> 30x5
        
        # Bottleneck
        b = self.bottleneck(e_final)
        
        # Decoding with skip connections
        # Use precise transposed convolution for upsampling
        b_up = self.bottleneck_upsample(b)  # 30x5 -> 61x11
        d5 = self.dec5(torch.cat([b_up, e5], dim=1))
        d4 = self.dec4(torch.cat([self.upsample(d5), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upsample(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))
        
        return torch.sigmoid(self.final(d1))
    
    def get_multi_level_features(self, x):
        """Extract features from multiple encoder levels"""
        features = []
        
        e1 = self.enc1(x)
        features.append(F.adaptive_avg_pool2d(e1, 1).squeeze(-1).squeeze(-1))  # Squeeze only spatial dims
        
        e2 = self.enc2(self.pool(e1))
        features.append(F.adaptive_avg_pool2d(e2, 1).squeeze(-1).squeeze(-1))
        
        e3 = self.enc3(self.pool(e2))
        features.append(F.adaptive_avg_pool2d(e3, 1).squeeze(-1).squeeze(-1))
        
        e4 = self.enc4(self.pool(e3))
        features.append(F.adaptive_avg_pool2d(e4, 1).squeeze(-1).squeeze(-1))
        
        e5 = self.enc5(self.pool(e4))
        features.append(F.adaptive_avg_pool2d(e5, 1).squeeze(-1).squeeze(-1))
        
        return torch.cat(features, dim=-1)

# ==================== Dataset with Augmentation ====================
class MVTecDataset(Dataset):
    def __init__(self, root_dir, category, split='train', transform=None, 
                 use_augmentation=False, synthetic_anomaly_generator=None):
        self.root_dir = Path(root_dir) / category / split
        self.transform = transform
        self.use_augmentation = use_augmentation
        self.synthetic_anomaly_generator = synthetic_anomaly_generator
        self.split = split
        self.images = []
        self.labels = []
        
        # Load images
        for folder in self.root_dir.iterdir():
            if folder.is_dir():
                is_anomaly = folder.name != 'good'
                for img_path in folder.glob('*.png'):
                    self.images.append(img_path)
                    self.labels.append(1 if is_anomaly else 0)
        
        # Conservative augmentation
        if use_augmentation and split == 'train':
            self.augmentation = transforms.Compose([
                transforms.RandomAffine(degrees=0, scale=(0.95, 1.05)),  # 0.95-1.05 scale
            ])
        else:
            self.augmentation = None
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load as grayscale
        image = Image.open(img_path).convert('L')
        
        # Apply augmentation if enabled
        if self.augmentation:
            image = self.augmentation(image)
        
        # Convert to tensor
        if self.transform:
            image = self.transform(image)
        
        # For training with synthetic anomalies
        if self.synthetic_anomaly_generator and self.split == 'train':
            # Store clean image
            clean_image = image.clone()
            # Generate anomaly
            anomaly_image, anomaly_mask = self.synthetic_anomaly_generator.generate_anomaly(image)
            # Return clean image, anomaly image, and mask
            return clean_image, anomaly_image, anomaly_mask
        
        return image, label

# ==================== Latent Space Analysis ====================
class LatentSpaceAnalyzer:
    """Analyze anomalies in latent space"""
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Store normal features for comparison
        self.normal_features = None
        self.feature_mean = None
        self.feature_std = None
    
    def fit_normal_features(self, normal_loader):
        """Fit analyzer on normal training data"""
        all_features = []
        
        with torch.no_grad():
            for batch, _ in tqdm(normal_loader, desc='Extracting normal features'):
                batch = batch.to(self.device)
                
                if hasattr(self.model, 'get_multi_level_features'):
                    features = self.model.get_multi_level_features(batch)
                else:
                    features = self.model.get_latent_features(batch)
                
                all_features.append(features.cpu())
        
        self.normal_features = torch.cat(all_features, dim=0)
        self.feature_mean = self.normal_features.mean(dim=0)
        self.feature_std = self.normal_features.std(dim=0)
    
    def compute_anomaly_score(self, image):
        """Compute anomaly score using L2 distance in latent space"""
        with torch.no_grad():
            if hasattr(self.model, 'get_multi_level_features'):
                features = self.model.get_multi_level_features(image)
            else:
                features = self.model.get_latent_features(image)
            
            # Normalize features
            normalized_features = (features - self.feature_mean.to(self.device)) / (self.feature_std.to(self.device) + 1e-6)
            
            # L2 distance to normal distribution
            score = torch.norm(normalized_features, p=2, dim=-1)
            
            return score

# ==================== Anomaly Visualization ====================
class AnomalyVisualizer:
    """Visualize anomaly detection results without ground truth"""
    def __init__(self, save_dir='./results'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def visualize_reconstruction(self, original, reconstruction, anomaly_map, 
                               save_name=None, show=True):
        """Visualize original, reconstruction, and anomaly heatmap"""
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(131)
        plt.imshow(original.squeeze().cpu().numpy(), cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        # Reconstruction
        plt.subplot(132)
        plt.imshow(reconstruction.squeeze().cpu().numpy(), cmap='gray')
        plt.title('Reconstruction')
        plt.axis('off')
        
        # Anomaly heatmap
        plt.subplot(133)
        plt.imshow(anomaly_map, cmap='hot')
        plt.title('Anomaly Heatmap')
        plt.colorbar()
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def save_anomaly_scores(self, scores, image_paths, save_name='anomaly_scores.txt'):
        """Save anomaly scores to text file"""
        with open(os.path.join(self.save_dir, save_name), 'w') as f:
            f.write("Image Path\tAnomaly Score\n")
            for path, score in zip(image_paths, scores):
                f.write(f"{path}\t{score:.6f}\n")

# ==================== Training Function ====================
def train_anomaly_model(model, train_loader, config):
    """Train anomaly detection model with modular loss support"""
    device = config['device']
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['num_epochs'])
    
    # Initialize loss function with ModularLossManager
    criterion = ModularLossManager(config['loss_config'], normalize_weights=True)
    criterion.to(device)
    
    # Track training history
    train_history = {
        'total_loss': [],
        'component_losses': {name: [] for name in criterion.losses.keys()},
        'weights': []
    }
    
    for epoch in range(config['num_epochs']):
        # Training
        model.train()
        train_losses = {key: 0 for key in criterion.losses.keys()}
        train_losses['total'] = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}'):
            if config.get('use_synthetic_anomalies', False):
                # Dataset returns clean_images, anomaly_images, anomaly_masks
                clean_images, anomaly_images, anomaly_masks = batch
                clean_images = clean_images.to(device)
                anomaly_images = anomaly_images.to(device)
                anomaly_masks = anomaly_masks.to(device)
                
                # Use clean images as target and anomaly images as input
                target = clean_images
                input_images = anomaly_images
            else:
                # Normal training without synthetic anomalies
                images, _ = batch
                images = images.to(device)
                target = images
                input_images = images
            
            # Forward pass
            recon = model(input_images)
            loss_dict = criterion(recon, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss_dict['total'].backward()
            optimizer.step()
            
            # Update losses
            for key, value in loss_dict.items():
                if key in train_losses:
                    train_losses[key] += value.item()
        
        # Average losses
        for key in train_losses:
            train_losses[key] /= len(train_loader)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Store history
        train_history['total_loss'].append(train_losses['total'])
        for name in criterion.losses.keys():
            if name in train_losses:
                train_history['component_losses'][name].append(train_losses[name])
        train_history['weights'].append(criterion.get_weights())
        
        # Print progress
        print(f"Epoch {epoch+1}: Train Loss: {train_losses['total']:.4f}")
        loss_components = [f"{name}: {train_losses.get(name, 0):.4f}" for name in criterion.losses.keys()]
        print(f"  Components - {', '.join(loss_components)}")
        
        # Print current weights
        weights = criterion.get_weights()
        weight_str = ', '.join([f"{name}: {weight:.3f}" for name, weight in weights.items()])
        print(f"  Weights - {weight_str}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_losses['total'],
            }, f"{config['save_path']}/checkpoint_epoch_{epoch+1}.pth")
    
    return model, train_history

# ==================== Main Execution ====================
def main():
    # Determine optimal number of workers
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    optimal_workers = min(4, cpu_count - 1)  # Leave one CPU free
    
    # Configuration
    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'batch_size': 16,
        'num_epochs': 100,
        'lr': 1e-3,
        'image_size': (976, 176),  # Updated size
        'architecture': 'enhanced',  # 'baseline' or 'enhanced'
        'use_synthetic_anomalies': True,
        'loss_config': {
            'mse': {
                'class': MSELoss,
                'weight': 0.3
            },
            'ssim': {
                'class': SSIMLoss,
                'weight': 0.3,
                'params': {'window_size': 11}
            },
            'focal_freq': {
                'class': FocalFrequencyLoss,
                'weight': 0.2,
                'params': {'alpha': 1.0, 'patch_factor': 1}
            },
            'sobel': {
                'class': SobelGradientLoss,
                'weight': 0.2
            }
        },
        'save_path': './models',
        'num_workers': optimal_workers  # Dynamic worker count
    }
    
    # Create save directory
    os.makedirs(config['save_path'], exist_ok=True)
    
    # Data transforms - resize to 976x176
    transform = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Initialize model
    if config['architecture'] == 'baseline':
        model = BaselineAutoencoder()
    else:
        model = EnhancedAutoencoder()
    
    print(f"Model architecture: {config['architecture']}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of workers: {config['num_workers']} (detected {cpu_count} CPUs)")
    
    # Training on MVTec categories
    categories = ['grid']  # Can be extended
    
    for category in categories:
        print(f"\nTraining on {category} category...")
        
        # Create training dataset
        train_dataset = MVTecDataset(
            '/Users/laiyongcheng/Desktop/autoencoder/', 
            category, 
            'train', 
            transform,
            use_augmentation=True,
            synthetic_anomaly_generator=SyntheticAnomalyGenerator() if config['use_synthetic_anomalies'] else None
        )
        
        # Create dataloader
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                                shuffle=True, num_workers=config['num_workers'])
        
        # Train model (no validation set needed)
        model, train_history = train_anomaly_model(model, train_loader, config)
        
        # Save final model
        torch.save(model.state_dict(), f"{config['save_path']}/{category}_final_model.pth")
        
        # Test on any available test images (optional)
        print("\nTesting on available images...")
        
        # Check if test directory exists
        test_path = Path('/Users/laiyongcheng/Desktop/autoencoder/') / category / 'test'
        if test_path.exists():
            print(f"Found test directory: {test_path}")
            
            # Create test dataset
            test_dataset = MVTecDataset(
                '/Users/laiyongcheng/Desktop/autoencoder/', 
                category, 
                'test', 
                transform
            )
            
            test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                                   shuffle=False, num_workers=config['num_workers'])
            
            # Setup visualization
            visualizer = AnomalyVisualizer(save_dir=f"{config['save_path']}/visualizations_{category}")
            latent_analyzer = LatentSpaceAnalyzer(model, config['device'])
            
            # Fit latent space on normal training data
            # Reuse train_loader instead of creating a new one
            latent_analyzer.fit_normal_features(train_loader)
            
            # Process test images
            model.eval()
            anomaly_scores = []
            
            # Visualize a few examples
            num_visualizations = min(10, len(test_dataset))
            viz_count = 0
            
            with torch.no_grad():
                for i, (batch, _) in enumerate(test_loader):
                    images = batch.to(config['device'])
                    
                    # Get reconstruction
                    recon = model(images)
                    
                    # Calculate anomaly scores
                    recon_error = torch.mean((images - recon) ** 2, dim=(1, 2, 3))
                    latent_scores = latent_analyzer.compute_anomaly_score(images)
                    batch_anomaly_scores = recon_error + 0.5 * latent_scores
                    
                    # Store scores
                    anomaly_scores.extend(batch_anomaly_scores.cpu().numpy())
                    
                    # Visualize some examples
                    for j in range(images.size(0)):
                        if viz_count < num_visualizations:
                            # Generate anomaly heatmap (raw difference)
                            diff = torch.abs(images[j] - recon[j])
                            heatmap = diff.cpu().numpy()[0]
                            # No gaussian smoothing - show raw difference
                            
                            # Normalize heatmap
                            if heatmap.max() > heatmap.min():
                                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
                            
                            # Visualize
                            visualizer.visualize_reconstruction(
                                images[j], recon[j], heatmap,
                                save_name=f'test_sample_{viz_count}.png',
                                show=False
                            )
                            viz_count += 1
            
            print(f"\nAnomaly detection completed for {category}")
            print(f"Number of test images: {len(anomaly_scores)}")
            if anomaly_scores:
                print(f"Average anomaly score: {np.mean(anomaly_scores):.4f}")
                print(f"Max anomaly score: {np.max(anomaly_scores):.4f}")
                print(f"Min anomaly score: {np.min(anomaly_scores):.4f}")
        else:
            print(f"No test directory found at {test_path}")
            print("Model training completed. Ready for inference on new images.")

if __name__ == '__main__':
    main()
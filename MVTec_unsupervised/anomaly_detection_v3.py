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
from typing import Dict, Optional, Any, List, Tuple

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
    """結構相似性指標損失函數
    
    本類別實現了 Wang et al. (2004) 提出的結構相似性指標 (SSIM) 損失函數。
    SSIM 透過綜合評估亮度、對比度和結構三個維度的相似性來衡量圖像品質，
    相較於傳統的均方誤差損失，SSIM 更符合人類視覺系統的感知特性。
    
    本實現遵循模組化損失函數框架的設計規範，確保能夠與其他損失函數
    無縫組合使用。所有批次樣本的損失值會自動聚合為單一標量值。
    
    參數說明：
        weight (float): 損失函數在總體損失中的權重係數。預設值為 1.0
        window_size (int): 高斯窗口的大小，必須為奇數。建議使用 11 或更大的值
                          以獲得穩定的結果。預設值為 11
        sigma (float): 高斯核的標準差，控制窗口權重的分布。較大的值會產生
                      更平滑的權重分布。預設值為 1.5
    """
    
    def __init__(self, 
                 weight: float = 1.0, 
                 window_size: int = 11, 
                 sigma: float = 1.5):
        super().__init__(weight)
        
        # 參數驗證確保輸入值的合理性
        if window_size % 2 == 0:
            raise ValueError(f"Window size must be odd, got {window_size}")
        if window_size < 3:
            raise ValueError(f"Window size must be at least 3, got {window_size}")
        if sigma <= 0:
            raise ValueError(f"Sigma must be positive, got {sigma}")
        
        self.window_size = window_size
        self.sigma = sigma
        self.channel = None  # 將根據輸入動態設定
        
        # 預先創建單通道高斯窗口並註冊為緩衝區
        # register_buffer 的使用確保了窗口在模型保存、載入和設備轉移時
        # 能夠得到正確的處理，避免手動管理這些操作
        initial_window = self._create_window(window_size, 1)
        self.register_buffer('window', initial_window, persistent=False)
        
    def _gaussian_1d(self, window_size: int, sigma: float) -> torch.Tensor:
        """生成一維高斯核
        
        根據高斯分布公式創建歸一化的一維核心，該核心將用於
        構建二維高斯窗口。歸一化確保權重總和為 1。
        
        參數：
            window_size: 核心的大小
            sigma: 高斯分布的標準差
            
        返回：
            torch.Tensor: 歸一化的一維高斯核
        """
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= window_size // 2
        
        gaussian = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return gaussian / gaussian.sum()
    
    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        """創建二維高斯窗口
        
        透過一維高斯核的外積生成二維高斯窗口。這種方法利用了
        高斯函數的可分離性，提高了計算效率。生成的窗口會根據
        需要擴展到指定的通道數。
        
        參數：
            window_size: 窗口的大小（高度和寬度相同）
            channel: 需要的通道數
            
        返回：
            torch.Tensor: 形狀為 (channel, 1, window_size, window_size) 的高斯窗口
        """
        gaussian_1d = self._gaussian_1d(window_size, self.sigma)
        gaussian_2d = gaussian_1d.unsqueeze(1) @ gaussian_1d.unsqueeze(0)
        gaussian_2d = gaussian_2d.unsqueeze(0).unsqueeze(0)
        
        # 擴展到指定通道數，每個通道使用相同的權重
        window = gaussian_2d.expand(channel, 1, window_size, window_size)
        return window.contiguous()
    
    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """計算結構相似性指標
        
        實現 SSIM 的核心計算邏輯。該方法計算輸入圖像對的局部統計量，
        並應用 SSIM 公式來評估相似性。計算過程包括局部均值、方差和
        協方差的估計，這些統計量分別對應於亮度、對比度和結構的相似性。
        
        參數：
            x: 第一張圖像張量，形狀為 [N, C, H, W]
            y: 第二張圖像張量，形狀為 [N, C, H, W]
            
        返回：
            torch.Tensor: 每個批次樣本的 SSIM 值，形狀為 [N]，值域為 [0, 1]
        """
        # SSIM 穩定性常數，基於原始論文的建議值
        # 這些常數防止分母為零，並控制各成分的相對重要性
        C1 = (0.01 ** 2)
        C2 = (0.03 ** 2)
        
        # 使用高斯加權計算局部均值
        # groups 參數確保每個通道獨立進行卷積運算
        mu_x = F.conv2d(x, self.window, padding=self.window_size//2, groups=self.channel)
        mu_y = F.conv2d(y, self.window, padding=self.window_size//2, groups=self.channel)
        
        # 計算均值的平方和乘積，用於後續計算
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        # 計算局部方差和協方差
        # 使用公式：Var(X) = E[X²] - E[X]²
        sigma_x_sq = F.conv2d(x ** 2, self.window, padding=self.window_size//2, groups=self.channel) - mu_x_sq
        sigma_y_sq = F.conv2d(y ** 2, self.window, padding=self.window_size//2, groups=self.channel) - mu_y_sq
        sigma_xy = F.conv2d(x * y, self.window, padding=self.window_size//2, groups=self.channel) - mu_xy
        
        # 應用 SSIM 公式
        # SSIM = (2μxμy + C1)(2σxy + C2) / (μx² + μy² + C1)(σx² + σy² + C2)
        luminance_contrast = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
        
        ssim_map = luminance_contrast / denominator
        
        # 對每個樣本的空間和通道維度取平均，保持批次維度
        # 這確保了批次中每個樣本都能獲得獨立的梯度信號
        return ssim_map.mean(dim=[1, 2, 3])
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """計算 SSIM 損失
        
        本方法是損失函數的主要介面，負責計算預測圖像與目標圖像之間的
        SSIM 損失。損失值定義為 1 - SSIM，確保完美匹配時損失為 0，
        完全不匹配時損失接近 1。
        
        參數：
            pred: 預測圖像張量，形狀必須為 [N, C, H, W]
            target: 目標圖像張量，形狀必須為 [N, C, H, W]
            
        返回：
            torch.Tensor: 標量損失值，代表整個批次的平均損失
            
        異常：
            ValueError: 當輸入張量的維度不是 4 或形狀不匹配時
        """
        # 輸入驗證確保張量格式正確
        if pred.dim() != 4 or target.dim() != 4:
            raise ValueError(f"Expected 4D tensors, got {pred.dim()}D and {target.dim()}D")
        
        if pred.shape != target.shape:
            raise ValueError(f"Input shapes must match, got {pred.shape} and {target.shape}")
        
        # 動態更新高斯窗口以匹配輸入的通道數
        channel = pred.size(1)
        if self.channel != channel:
            self.channel = channel
            new_window = self._create_window(self.window_size, channel)
            self.window = new_window.to(device=pred.device, dtype=pred.dtype)
        
        # 確保窗口與輸入張量在相同的設備和資料類型
        if self.window.device != pred.device or self.window.dtype != pred.dtype:
            self.window = self.window.to(device=pred.device, dtype=pred.dtype)
        
        # 計算 SSIM 值並轉換為損失
        ssim_values = self._ssim(pred, target)
        loss = 1.0 - ssim_values
        
        # 返回批次平均損失（標量），確保與其他損失函數的介面一致
        return loss.mean()
    
    def compute_per_sample_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """計算每個樣本的個別損失值
        
        此輔助方法提供了獲取批次中每個樣本詳細損失資訊的途徑，
        適用於分析、調試或需要樣本級別損失資訊的場景。該方法
        不會影響主要的訓練流程。
        
        參數：
            pred: 預測圖像張量，形狀為 [N, C, H, W]
            target: 目標圖像張量，形狀為 [N, C, H, W]
            
        返回：
            torch.Tensor: 每個樣本的損失值，形狀為 [N]
        """
        with torch.no_grad():
            # 確保設備和窗口設定正確
            if pred.size(1) != self.channel:
                self.forward(pred, target)  # 觸發窗口更新
            
            ssim_values = self._ssim(pred, target)
            return 1.0 - ssim_values
    
    def __repr__(self) -> str:
        """提供類別的字串表示
        
        返回包含所有關鍵參數的描述性字串，便於調試和日誌記錄。
        """
        return (f"{self.__class__.__name__}("
                f"weight={self.weight}, "
                f"window_size={self.window_size}, "
                f"sigma={self.sigma})")


class MultiScaleSSIMLoss(BaseLoss):
    """Multi-Scale Structural Similarity Index Loss
    
    This loss function computes SSIM at multiple scales to capture both 
    fine-grained details and global structure. It's particularly effective
    for images where both local and global features are important.
    
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
        return (luminance * contrast_structure).mean(dim=[2, 3]), contrast_structure.mean(dim=[2, 3])
    
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
                    # If images are too small, stop and adjust weights
                    remaining_scales = self.scales - scale - 1
                    if remaining_scales > 0:
                        # Redistribute remaining weight to current scale
                        weight_sum = self.scale_weights[scale:].sum()
                        self.scale_weights = self.scale_weights[:scale+1]
                        self.scale_weights[-1] = weight_sum
                        self.scale_weights = self.scale_weights / self.scale_weights.sum()
                    break
        
        # Compute MS-SSIM
        if len(mcs_values) == 0:
            # Only one scale was used
            ms_ssim = final_ssim
        else:
            # Combine contrast-structure values with final SSIM
            mcs_stack = torch.stack(mcs_values)
            
            # Apply weights to contrast-structure values
            # mcs_stack shape: [num_scales, batch_size]
            # weights shape: [num_scales, 1] for broadcasting
            mcs_weighted = torch.prod(
                torch.pow(mcs_stack, self.scale_weights[:len(mcs_values)].unsqueeze(-1)),
                dim=0  # Product over scales, keep batch dimension
            )
            
            # Combine with final SSIM
            ms_ssim = mcs_weighted * (final_ssim ** self.scale_weights[-1])
        
        # Return loss (1 - MS-SSIM)
        # Average over batch dimension to get scalar loss
        return torch.mean(1.0 - ms_ssim)
    
    def __repr__(self) -> str:
        """String representation for debugging"""
        return (f"MultiScaleSSIMLoss(weight={self.weight}, scales={self.scales}, "
                f"scale_weights={self.scale_weights.tolist()}, "
                f"downsample_method={self.downsample_method})")
        
        
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
        channels = pred.shape[1]
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

    def forward(self, pred, target, matrix=None):
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
        # Loss function configuration - adjust weights and parameters freely
        'loss_config': {
            # MSE Loss: Basic pixel-level reconstruction
            'mse': {
                'class': MSELoss,
                'weight': 0.3  # Weight: 0-1, all weights will be auto-normalized
            },
            
            # SSIM Loss: Structural similarity preservation
            'ssim': {
                'class': SSIMLoss,
                'weight': 0.3,
                'params': {
                    'window_size': 11,  # Gaussian window size, must be odd
                    'sigma': 1.5  # Gaussian kernel standard deviation
                }
            },
            
            # Focal Frequency Loss: Dynamic focus on hard-to-reconstruct regions
            'focal_freq': {
                'class': FocalFrequencyLoss,
                'weight': 0.2,
                'params': {
                    'alpha': 1.0,  # Spectrum weight scaling factor
                    'patch_factor': 1,  # Image patch factor
                    'ave_spectrum': False,  # Use batch average spectrum
                    'log_matrix': False,  # Apply log to spectrum weights
                    'batch_matrix': False  # Use batch statistics
                }
            },
            
            # Sobel Gradient Loss: Edge information preservation
            'sobel': {
                'class': SobelGradientLoss,
                'weight': 0.2
            }
            
            # Easy to add/remove loss functions, e.g.:
            # 'perceptual': {
            #     'class': PerceptualLoss,  # Need to define this class first
            #     'weight': 0.1,
            #     'params': {'feature_layers': ['relu1_2', 'relu2_2']}
            # }
        },
        
        # Example configurations for experiments:
        # 1. MSE + SSIM only (fast training)
        # 'loss_config': {
        #     'mse': {'class': MSELoss, 'weight': 0.5},
        #     'ssim': {'class': SSIMLoss, 'weight': 0.5, 'params': {'window_size': 11}}
        # },
        
        # 2. Emphasize frequency domain (good for textured images)
        # 'loss_config': {
        #     'mse': {'class': MSELoss, 'weight': 0.2},
        #     'focal_freq': {'class': FocalFrequencyLoss, 'weight': 0.5, 'params': {'alpha': 2.0}},
        #     'sobel': {'class': SobelGradientLoss, 'weight': 0.3}
        # },
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
        
        # Save training history for analysis
        import json
        history_path = f"{config['save_path']}/{category}_training_history.json"
        with open(history_path, 'w') as f:
            # Convert history to serializable format
            serializable_history = {
                'total_loss': train_history['total_loss'],
                'component_losses': train_history['component_losses'],
                'weights': [{k: float(v) for k, v in w.items()} for w in train_history['weights']]
            }
            json.dump(serializable_history, f, indent=2)
        print(f"Training history saved to {history_path}")
        
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
"""
Difference Denoiser with Dual Output
====================================

A specialized autoencoder that learns to denoise difference images and detect anomalies.
Takes three difference images as input and outputs both an anomaly heatmap and reconstructed differences.
"""

import torch
import torch.nn as nn
from typing import Tuple

# Import C3k2 components
from .c3k2 import Conv, C3k2


class DifferenceDenoiserEncoder(nn.Module):
    """
    Encoder based on C3k2 blocks, modified for 3-channel input.
    """
    
    def __init__(self, latent_dim=128):
        super().__init__()
        
        # Initial convolution for 3-channel input
        self.enc_conv1 = Conv(3, 32, 3, 2, 1)  # /2
        self.enc_c3k2_1 = C3k2(32, 32, n=1, shortcut=True, e=0.25)  # Small n for details
        
        self.enc_conv2 = Conv(32, 64, 3, 2, 1)  # /2
        self.enc_c3k2_2 = C3k2(64, 64, n=1, shortcut=True, e=0.25)
        
        self.enc_conv3 = Conv(64, 128, 3, 2, 1)  # /2
        self.enc_c3k2_3 = C3k2(128, 128, n=2, shortcut=True, e=0.5)
        
        self.enc_conv4 = Conv(128, 256, 3, 2, 1)  # /2
        self.enc_c3k2_4 = C3k2(256, 256, n=3, shortcut=True, e=0.5)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, latent_dim, 1),
            nn.BatchNorm2d(latent_dim),
            nn.SiLU(inplace=True),
        )
        
        # Expansion back to 256 channels for decoder
        self.expansion = nn.Sequential(
            nn.Conv2d(latent_dim, 256, 1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
        )
        
    def forward(self, x):
        # Encoding path
        x = self.enc_conv1(x)
        x = self.enc_c3k2_1(x)
        
        x = self.enc_conv2(x)
        x = self.enc_c3k2_2(x)
        
        x = self.enc_conv3(x)
        x = self.enc_c3k2_3(x)
        
        x = self.enc_conv4(x)
        x = self.enc_c3k2_4(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        x = self.expansion(x)
        
        return x


class DifferenceReconstructionDecoder(nn.Module):
    """
    Decoder for reconstructing the input difference images.
    """
    
    def __init__(self):
        super().__init__()
        
        # Decoder with C3k2 blocks
        self.dec_upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv(256, 128, 3, 1, 1)
        )
        self.dec_c3k2_1 = C3k2(128, 128, n=2, shortcut=True, e=0.5)
        
        self.dec_upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv(128, 64, 3, 1, 1)
        )
        self.dec_c3k2_2 = C3k2(64, 64, n=1, shortcut=True, e=0.25)
        
        self.dec_upconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv(64, 32, 3, 1, 1)
        )
        self.dec_c3k2_3 = C3k2(32, 32, n=1, shortcut=True, e=0.25)
        
        self.dec_upconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv(32, 32, 3, 1, 1)
        )
        
        # Final convolution to reconstruct 3 channels
        self.final = nn.Conv2d(32, 3, 1)
        
    def forward(self, x):
        x = self.dec_upconv1(x)
        x = self.dec_c3k2_1(x)
        
        x = self.dec_upconv2(x)
        x = self.dec_c3k2_2(x)
        
        x = self.dec_upconv3(x)
        x = self.dec_c3k2_3(x)
        
        x = self.dec_upconv4(x)
        
        # No sigmoid - difference images can be negative
        x = self.final(x)
        
        return x


class AnomalyDecoder(nn.Module):
    """
    Lightweight decoder for generating anomaly heatmap.
    """
    
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            # Simple upsampling path without heavy processing
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # x2
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # x2
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # x2
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            
            nn.ConvTranspose2d(32, 16, 4, 2, 1),  # x2
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            
            # Final layer outputs single channel
            nn.Conv2d(16, 1, 1)
        )
        
    def forward(self, x):
        return self.layers(x)


class DifferenceDenoiser(nn.Module):
    """
    Dual-output network for difference image denoising and anomaly detection.
    
    Input: target, reference1, reference2 images
    Process: Computes three difference images internally
    Output: (anomaly_heatmap, reconstructed_differences)
    """
    
    def __init__(self, latent_dim=128):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Shared encoder
        self.encoder = DifferenceDenoiserEncoder(latent_dim)
        
        # Dual decoders
        self.reconstruction_decoder = DifferenceReconstructionDecoder()
        self.anomaly_decoder = AnomalyDecoder()
        
    def compute_differences(self, target: torch.Tensor, ref1: torch.Tensor, ref2: torch.Tensor) -> torch.Tensor:
        """
        Compute the three difference images that serve as input.
        
        Args:
            target: Target image (potentially with blur and defects)
            ref1: First reference image
            ref2: Second reference image
            
        Returns:
            Concatenated difference images [diff1, diff2, ref_diff]
        """
        diff1 = target - ref1  # Difference between target and ref1
        diff2 = target - ref2  # Difference between target and ref2
        ref_diff = ref1 - ref2  # Difference between references (noise level indicator)
        
        # Stack along channel dimension
        return torch.cat([diff1, diff2, ref_diff], dim=1)
    
    def forward(self, target: torch.Tensor, ref1: torch.Tensor, ref2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            target: Target image [B, 1, H, W]
            ref1: First reference image [B, 1, H, W]
            ref2: Second reference image [B, 1, H, W]
            
        Returns:
            anomaly_map: Predicted anomaly heatmap [B, 1, H, W]
            reconstructed_diffs: Reconstructed difference images [B, 3, H, W]
            input_diffs: Original difference images [B, 3, H, W] (for loss computation)
        """
        # Compute input differences
        input_diffs = self.compute_differences(target, ref1, ref2)
        
        # Shared encoding
        features = self.encoder(input_diffs)
        
        # Dual decoding
        reconstructed_diffs = self.reconstruction_decoder(features)
        anomaly_map = self.anomaly_decoder(features)
        
        return anomaly_map, reconstructed_diffs, input_diffs
    
    def get_anomaly_score(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """
        Convert anomaly map to a single anomaly score.
        
        Args:
            anomaly_map: Predicted anomaly heatmap [B, 1, H, W]
            
        Returns:
            Anomaly scores for each image in batch [B]
        """
        # Use maximum absolute value as anomaly score
        return torch.max(torch.abs(anomaly_map.view(anomaly_map.shape[0], -1)), dim=1)[0]
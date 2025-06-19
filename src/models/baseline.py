"""
Baseline Autoencoder
===================

Standard autoencoder architecture without skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineAutoencoder(nn.Module):
    """Standard autoencoder without skip connections"""
    def __init__(self, latent_dim=128, input_size=(1024, 1024)):
        super().__init__()
        self.input_size = input_size
        
        # Encoder with 3x3 kernels and SiLU activation
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),  # /2 downsampling
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),  # /2 downsampling
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), # /2 downsampling
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1), # /2 downsampling
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 512, 3, 2, 1), # /2 downsampling
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
            nn.ConvTranspose2d(512, 256, 3, 2, 1, output_padding=1),  # Matches encoder
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
    
    def get_features(self, x):
        """Extract features from the encoder (before bottleneck)"""
        return self.encoder(x)
    
    def get_latent(self, x):
        """Extract latent representation (after bottleneck)"""
        encoded = self.encoder(x)
        return self.bottleneck(encoded)
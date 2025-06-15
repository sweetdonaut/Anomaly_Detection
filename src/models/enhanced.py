"""
Enhanced Autoencoder
===================

U-Net style autoencoder with skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            nn.Conv2d(512, 512, 3, 2, 1),  # Supports various input sizes
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
        self.bottleneck_upsample = nn.ConvTranspose2d(512, 512, 3, 2, 1, output_padding=1)  # Matches encoder
        
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
        e_final = self.enc_final(e5)  # Additional /2 downsampling
        
        # Bottleneck
        b = self.bottleneck(e_final)
        
        # Decoding with skip connections
        # Helper function to match sizes for skip connections
        def match_size(upsampled, skip):
            if upsampled.shape[2:] != skip.shape[2:]:
                # Adjust size using interpolation
                upsampled = F.interpolate(upsampled, size=skip.shape[2:], 
                                        mode='bilinear', align_corners=False)
            return upsampled
        
        # Use precise transposed convolution for upsampling
        b_up = self.bottleneck_upsample(b)  # x2 upsampling
        b_up = match_size(b_up, e5)
        d5 = self.dec5(torch.cat([b_up, e5], dim=1))
        
        d4_up = match_size(self.upsample(d5), e4)
        d4 = self.dec4(torch.cat([d4_up, e4], dim=1))
        
        d3_up = match_size(self.upsample(d4), e3)
        d3 = self.dec3(torch.cat([d3_up, e3], dim=1))
        
        d2_up = match_size(self.upsample(d3), e2)
        d2 = self.dec2(torch.cat([d2_up, e2], dim=1))
        
        d1_up = match_size(self.upsample(d2), e1)
        d1 = self.dec1(torch.cat([d1_up, e1], dim=1))
        
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
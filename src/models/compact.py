"""
Compact Autoencoder
==================

Autoencoder with 4 downsampling layers instead of 5, preserving more spatial information.
Particularly suitable for elongated images where excessive downsampling can lose important details.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompactAutoencoder(nn.Module):
    """Compact autoencoder with 4 downsampling layers
    
    This architecture reduces the number of downsampling operations from 5 to 4,
    which helps preserve more spatial information. For 176x976 images:
    - Original: 176x976 → 88x488 → 44x244 → 22x122 → 11x61 → 5x30
    - Compact:  176x976 → 88x488 → 44x244 → 22x122 → 11x61
    
    The final feature map (11x61) retains more spatial resolution than (5x30).
    """
    
    def __init__(self, input_size=(256, 256), latent_dim=256):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        
        # Encoder with 4 downsampling layers
        self.encoder = nn.Sequential(
            # Layer 1: 1 → 32 channels, /2
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            
            # Layer 2: 32 → 64 channels, /2
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            
            # Layer 3: 64 → 128 channels, /2
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            
            # Layer 4: 128 → 256 channels, /2
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
        )
        
        # Calculate encoder output size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *input_size)
            encoder_output = self.encoder(dummy_input)
            self.encoder_output_size = encoder_output.shape[2:]
        
        # Bottleneck with residual connection
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, latent_dim, 1),
            nn.BatchNorm2d(latent_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(latent_dim, 256, 1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
        )
        
        # Decoder with 4 upsampling layers
        self.decoder = nn.ModuleList([
            # Layer 1: 256 → 128 channels, x2
            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            
            # Layer 2: 128 → 64 channels, x2
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            
            # Layer 3: 64 → 32 channels, x2
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            
            # Layer 4: 32 → 32 channels, x2 (keep more channels before final)
            nn.ConvTranspose2d(32, 32, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        ])
        
        # Final convolution to reconstruct single channel
        self.final = nn.Conv2d(32, 1, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the autoencoder"""
        # Encode
        encoded = self.encoder(x)
        
        # Bottleneck with residual connection
        bottleneck_output = self.bottleneck(encoded)
        bottleneck_output = bottleneck_output + encoded  # Residual connection
        
        # Decode
        decoded = bottleneck_output
        for i in range(0, len(self.decoder), 3):  # Process in groups of 3 (conv, bn, activation)
            if i + 2 < len(self.decoder):
                decoded = self.decoder[i](decoded)
                decoded = self.decoder[i + 1](decoded)
                decoded = self.decoder[i + 2](decoded)
        
        # Ensure output matches input size
        if decoded.shape[2:] != x.shape[2:]:
            decoded = F.interpolate(decoded, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Final reconstruction
        output = torch.sigmoid(self.final(decoded))
        
        return output
    
    def get_features(self, x):
        """Extract features from the encoder (useful for analysis)"""
        return self.encoder(x)
    
    def get_latent(self, x):
        """Extract latent representation (after bottleneck)"""
        encoded = self.encoder(x)
        return self.bottleneck(encoded)


class CompactUNetAutoencoder(nn.Module):
    """Compact U-Net style autoencoder with skip connections and 4 downsampling layers
    
    Similar to CompactAutoencoder but with skip connections for better detail preservation.
    """
    
    def __init__(self, input_size=(256, 256)):
        super().__init__()
        
        # Encoder blocks
        self.enc1 = self._conv_block(1, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)
        self.enc4 = self._conv_block(128, 256)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True)
        )
        
        # Decoder blocks with skip connections
        self.dec4 = self._conv_block(256 + 256, 128)  # Skip from enc4
        self.dec3 = self._conv_block(128 + 128, 64)   # Skip from enc3
        self.dec2 = self._conv_block(64 + 64, 32)     # Skip from enc2
        self.dec1 = self._conv_block(32 + 32, 32)     # Skip from enc1
        
        # Output layer
        self.final = nn.Conv2d(32, 1, 1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    def _conv_block(self, in_ch, out_ch):
        """Double convolution block"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x):
        """Forward pass with skip connections"""
        # Encoding path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoding path with skip connections
        d4 = self.dec4(torch.cat([self.upsample(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upsample(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))
        
        # Output
        return torch.sigmoid(self.final(d1))
    
    def get_features(self, x):
        """Extract features from the encoder (before bottleneck)"""
        # Encoding path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        # Return the pooled e4 (input to bottleneck)
        return self.pool(e4)
    
    def get_latent(self, x):
        """Extract latent representation (after bottleneck)"""
        # Get encoder features
        features = self.get_features(x)
        # Apply bottleneck
        return self.bottleneck(features)
    
    def get_multi_level_features(self, x):
        """Extract features from multiple encoder levels (useful for analysis)"""
        features = []
        
        e1 = self.enc1(x)
        features.append(F.adaptive_avg_pool2d(e1, 1).squeeze(-1).squeeze(-1))
        
        e2 = self.enc2(self.pool(e1))
        features.append(F.adaptive_avg_pool2d(e2, 1).squeeze(-1).squeeze(-1))
        
        e3 = self.enc3(self.pool(e2))
        features.append(F.adaptive_avg_pool2d(e3, 1).squeeze(-1).squeeze(-1))
        
        e4 = self.enc4(self.pool(e3))
        features.append(F.adaptive_avg_pool2d(e4, 1).squeeze(-1).squeeze(-1))
        
        return torch.cat(features, dim=1)
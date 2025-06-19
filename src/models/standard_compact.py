"""
Standard Compact Autoencoder
============================

A standard autoencoder with 4 downsampling layers WITHOUT residual connections in the bottleneck.
This is a more traditional autoencoder design that enforces a true information bottleneck.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardCompactAutoencoder(nn.Module):
    """Standard compact autoencoder without residual connections
    
    This architecture is similar to CompactAutoencoder but without the residual
    connection in the bottleneck layer, making it a more traditional autoencoder
    that enforces a stricter information bottleneck.
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
        
        # Standard bottleneck WITHOUT residual connection
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, latent_dim, 1),
            nn.BatchNorm2d(latent_dim),
            nn.SiLU(inplace=True),
        )
        
        # Expansion layer to return to 256 channels for decoder
        self.expansion = nn.Sequential(
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
            
            # Layer 4: 32 → 32 channels, x2
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
        
        # Bottleneck (NO residual connection)
        bottleneck_output = self.bottleneck(encoded)
        
        # Expand back to 256 channels
        expanded = self.expansion(bottleneck_output)
        
        # Decode
        decoded = expanded
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
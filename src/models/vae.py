"""
Variational Autoencoder (VAE)
=============================

VAE implementation for anomaly detection with KL divergence regularization.
This helps prevent the model from memorizing fine details including small anomalies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder with 4 downsampling layers
    
    VAE adds stochasticity to the latent space which helps prevent
    overfitting and memorization of small details/anomalies.
    """
    
    def __init__(self, input_size=(256, 256), latent_dim=128):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        
        # Encoder - outputs mean and log variance
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
            self.flatten_size = encoder_output.shape[1] * encoder_output.shape[2] * encoder_output.shape[3]
        
        # Latent space projection layers
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # Decoder projection
        self.fc_decode = nn.Linear(latent_dim, self.flatten_size)
        
        # Decoder
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
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def encode(self, x):
        """Encode input to latent distribution parameters"""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During evaluation, just use the mean
            return mu
    
    def decode(self, z):
        """Decode latent vector to reconstruction"""
        # Project back to feature map size
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, *self.encoder_output_size)
        
        # Decode through transposed convolutions
        for i in range(0, len(self.decoder), 3):  # Process in groups of 3 (conv, bn, activation)
            if i + 2 < len(self.decoder):
                h = self.decoder[i](h)
                h = self.decoder[i + 1](h)
                h = self.decoder[i + 2](h)
        
        return h
    
    def forward(self, x):
        """Forward pass through VAE"""
        # Encode
        mu, logvar = self.encode(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        decoded = self.decode(z)
        
        # Ensure output matches input size
        if decoded.shape[2:] != x.shape[2:]:
            decoded = F.interpolate(decoded, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Final reconstruction
        output = torch.sigmoid(self.final(decoded))
        
        # Return reconstruction and distribution parameters for loss calculation
        return output, mu, logvar
    
    def get_features(self, x):
        """Extract features from the encoder (useful for analysis)"""
        return self.encoder(x)
    
    def get_latent(self, x):
        """Extract latent representation (mean of distribution)"""
        mu, _ = self.encode(x)
        return mu
    
    def sample(self, num_samples, device):
        """Sample new images from the latent space"""
        # Sample from standard normal
        z = torch.randn(num_samples, self.latent_dim).to(device)
        
        # Decode
        decoded = self.decode(z)
        
        # Final reconstruction
        samples = torch.sigmoid(self.final(decoded))
        
        return samples


class ConditionalVAE(VariationalAutoencoder):
    """Conditional VAE that can incorporate additional information
    
    This is useful if you have different types of normal samples
    and want the model to learn conditional distributions.
    """
    
    def __init__(self, input_size=(256, 256), latent_dim=128, num_conditions=0):
        self.num_conditions = num_conditions
        super().__init__(input_size, latent_dim)
        
        if num_conditions > 0:
            # Modify fc layers to include condition information
            self.fc_mu = nn.Linear(self.flatten_size + num_conditions, latent_dim)
            self.fc_logvar = nn.Linear(self.flatten_size + num_conditions, latent_dim)
            self.fc_decode = nn.Linear(latent_dim + num_conditions, self.flatten_size)
    
    def encode(self, x, c=None):
        """Encode input with optional condition"""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten
        
        if c is not None and self.num_conditions > 0:
            h = torch.cat([h, c], dim=1)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def decode(self, z, c=None):
        """Decode latent vector with optional condition"""
        if c is not None and self.num_conditions > 0:
            z = torch.cat([z, c], dim=1)
        
        # Project back to feature map size
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, *self.encoder_output_size)
        
        # Decode through transposed convolutions
        for i in range(0, len(self.decoder), 3):
            if i + 2 < len(self.decoder):
                h = self.decoder[i](h)
                h = self.decoder[i + 1](h)
                h = self.decoder[i + 2](h)
        
        return h
    
    def forward(self, x, c=None):
        """Forward pass through conditional VAE"""
        # Encode
        mu, logvar = self.encode(x, c)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        decoded = self.decode(z, c)
        
        # Ensure output matches input size
        if decoded.shape[2:] != x.shape[2:]:
            decoded = F.interpolate(decoded, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Final reconstruction
        output = torch.sigmoid(self.final(decoded))
        
        return output, mu, logvar
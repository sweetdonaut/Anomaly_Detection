"""
Bottleneck Feature Analysis
==========================

Analyze and visualize features extracted from Autoencoder bottleneck layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from datetime import datetime


class CompactAutoencoder(nn.Module):
    """Compact autoencoder with 4 downsampling layers"""
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
    
    def get_latent_features(self, x):
        """Extract latent representation (after bottleneck)"""
        encoded = self.encoder(x)
        return self.bottleneck(encoded) + encoded
    
    def decode_from_bottleneck(self, bottleneck_features, target_size=None):
        """Decode from bottleneck features to reconstruct image
        
        Args:
            bottleneck_features: Tensor of shape (batch, 256, H, W)
                                where H, W are the spatial dimensions after encoding
            target_size: Optional tuple (H, W) for output size. If None, uses self.input_size
        
        Returns:
            Reconstructed image tensor of shape (batch, 1, H, W)
        """
        # Decode
        decoded = bottleneck_features
        for i in range(0, len(self.decoder), 3):  # Process in groups of 3 (conv, bn, activation)
            if i + 2 < len(self.decoder):
                decoded = self.decoder[i](decoded)
                decoded = self.decoder[i + 1](decoded)
                decoded = self.decoder[i + 2](decoded)
        
        # Final reconstruction
        output = torch.sigmoid(self.final(decoded))
        
        # Resize if needed
        if target_size is not None and output.shape[2:] != target_size:
            output = F.interpolate(output, size=target_size, mode='bilinear', align_corners=False)
        
        return output


def load_model(model_path, device='cuda'):
    """Load trained model"""
    # Create model instance
    # Note: PyTorch convention is (height, width), but here we follow the original code using (176, 976)
    # Actual images are 976(H) × 176(W) vertical strips
    model = CompactAutoencoder(input_size=(176, 976), latent_dim=256)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    return model


def load_and_preprocess_image(image_path):
    """Load and preprocess image"""
    # Load image
    img = Image.open(image_path)
    
    # Handle different image modes
    if img.mode == 'F':
        # 32-bit floating point - already in 0-1 range
        img_array = np.array(img).astype(np.float32)
    else:
        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')
        # 8-bit or 16-bit integer, needs normalization
        img_array = np.array(img).astype(np.float32)
        # Check value range and normalize
        if img_array.max() > 1.0:
            img_array = img_array / 255.0
    
    # Resize to model's expected dimensions
    # Convert back to PIL Image for resizing
    img = Image.fromarray(img_array)
    img = img.resize((176, 976), Image.Resampling.LANCZOS)
    
    # Convert to final numpy array
    img_array = np.array(img).astype(np.float32)
    
    # Add batch and channel dimensions
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    
    return img_tensor, img_array


def visualize_bottleneck_features(model, image_tensor, image_name, save_path, device='cuda'):
    """Visualize bottleneck features"""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        # Get bottleneck features
        bottleneck_features = model.get_latent_features(image_tensor)
        
        # Get reconstructed image
        reconstructed = model(image_tensor)
    
    # Convert tensors to numpy
    bottleneck_np = bottleneck_features.cpu().numpy()[0]  # Shape: (latent_dim, H, W)
    reconstructed_np = reconstructed.cpu().numpy()[0, 0]  # Shape: (H, W)
    original_np = image_tensor.cpu().numpy()[0, 0]  # Shape: (H, W)
    
    # Calculate importance (activity) for each channel
    channel_importance = []
    for i in range(bottleneck_np.shape[0]):
        channel = bottleneck_np[i]
        abs_mean = np.mean(np.abs(channel))
        channel_importance.append((i, abs_mean, channel))
    
    # Sort by importance
    channel_importance.sort(key=lambda x: x[1], reverse=True)
    
    # Create visualization
    plt.figure(figsize=(24, 20))
    
    # 1. Original image
    plt.subplot(4, 5, 1)
    plt.imshow(original_np, cmap='gray')
    plt.title(f'Original Image: {image_name}\nShape: {original_np.shape} (H×W)')
    plt.colorbar()
    plt.axis('off')
    
    # 2. Reconstructed image
    plt.subplot(4, 5, 2)
    plt.imshow(reconstructed_np, cmap='gray')
    plt.title('Reconstructed Image')
    plt.colorbar()
    plt.axis('off')
    
    # 3. Reconstruction error
    plt.subplot(4, 5, 3)
    error = np.abs(original_np - reconstructed_np)
    plt.imshow(error, cmap='hot')
    plt.title(f'Reconstruction Error (MSE: {np.mean(error**2):.6f})')
    plt.colorbar()
    plt.axis('off')
    
    # 4. Bottleneck feature statistics
    plt.subplot(4, 5, 4)
    plt.hist(bottleneck_np.flatten(), bins=50, alpha=0.7)
    plt.title('Bottleneck Features Distribution')
    plt.xlabel('Feature Value')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    # 5. Mean feature map
    plt.subplot(4, 5, 5)
    mean_features = np.mean(bottleneck_np, axis=0)
    plt.imshow(mean_features, cmap='viridis')
    plt.title(f'Mean Feature Map\nShape: {mean_features.shape} (H×W)')
    plt.colorbar()
    plt.axis('off')
    
    # 6. Feature map variance
    plt.subplot(4, 5, 6)
    var_features = np.var(bottleneck_np, axis=0)
    plt.imshow(var_features, cmap='plasma')
    plt.title('Feature Map Variance')
    plt.colorbar()
    plt.axis('off')
    
    # 7-20. Top 14 most important feature channels (sorted by activity)
    for i in range(14):
        plt.subplot(4, 5, 7 + i)
        if i < len(channel_importance):
            channel_idx, abs_mean, channel_data = channel_importance[i]
            plt.imshow(channel_data, cmap='coolwarm')
            plt.title(f'Channel {channel_idx}\n(Importance: {abs_mean:.2f})')
            plt.colorbar()
            plt.axis('off')
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return bottleneck_features


def analyze_feature_importance(model, image_tensor, device='cuda'):
    """Analyze feature importance"""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        # Get bottleneck features
        bottleneck_features = model.get_latent_features(image_tensor)
        
        # Calculate statistics for each feature channel
        features_np = bottleneck_features.cpu().numpy()[0]
        
        # Calculate statistics for each channel
        channel_stats = []
        for i in range(features_np.shape[0]):
            channel = features_np[i]
            stats = {
                'channel': i,
                'mean': np.mean(channel),
                'std': np.std(channel),
                'max': np.max(channel),
                'min': np.min(channel),
                'abs_mean': np.mean(np.abs(channel))
            }
            channel_stats.append(stats)
        
        # Sort by absolute mean (represents activity level)
        channel_stats.sort(key=lambda x: x['abs_mean'], reverse=True)
        
    return channel_stats


def manipulate_bottleneck_features(model, image_tensor, manipulations, save_path, device='cuda'):
    """Manipulate bottleneck features and decode to see effects
    
    Args:
        model: The autoencoder model
        image_tensor: Input image tensor
        manipulations: List of tuples (name, function) where function takes bottleneck features
        save_path: Path to save visualization
        device: Device to run on
    """
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        # Get original bottleneck features
        bottleneck_features = model.get_latent_features(image_tensor)
        
        # Get original reconstruction
        original_recon = model(image_tensor)
        
        # Create figure
        n_manipulations = len(manipulations)
        fig, axes = plt.subplots(2, n_manipulations + 2, figsize=(4 * (n_manipulations + 2), 8))
        
        # Show original image
        axes[0, 0].imshow(image_tensor.cpu().numpy()[0, 0], cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Show original reconstruction
        axes[0, 1].imshow(original_recon.cpu().numpy()[0, 0], cmap='gray')
        axes[0, 1].set_title('Original Reconstruction')
        axes[0, 1].axis('off')
        
        # Show bottleneck mean activation
        axes[1, 0].imshow(bottleneck_features.mean(dim=1).cpu().numpy()[0], cmap='viridis')
        axes[1, 0].set_title('Mean Bottleneck Activation')
        axes[1, 0].axis('off')
        
        # Empty subplot
        axes[1, 1].axis('off')
        
        # Apply manipulations
        for i, (name, manipulation_fn) in enumerate(manipulations):
            # Manipulate bottleneck features
            manipulated_features = manipulation_fn(bottleneck_features.clone())
            
            # Decode from manipulated features
            reconstructed = model.decode_from_bottleneck(manipulated_features, target_size=image_tensor.shape[2:])
            
            # Show manipulated reconstruction
            axes[0, i + 2].imshow(reconstructed.cpu().numpy()[0, 0], cmap='gray')
            axes[0, i + 2].set_title(f'{name}')
            axes[0, i + 2].axis('off')
            
            # Show difference map
            diff = torch.abs(reconstructed - original_recon)
            axes[1, i + 2].imshow(diff.cpu().numpy()[0, 0], cmap='hot')
            axes[1, i + 2].set_title(f'Difference Map')
            axes[1, i + 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Bottleneck manipulation visualization saved to: {save_path}")


def main():
    """Main program"""
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y_%m%d_%H%M%S')
    outputs_dir = os.path.join('outputs', timestamp)
    os.makedirs(outputs_dir, exist_ok=True)
    print(f"Output directory: {outputs_dir}")
    
    # Load model
    print("Loading model...")
    model = load_model('models/compact_model.pth', device)
    
    # Get all test images
    images_dir = 'images'
    test_images = [f for f in os.listdir(images_dir) if f.endswith('.tiff')]
    print(f"Found {len(test_images)} test images in {images_dir}/")
    
    # Analyze each image
    all_stats = []
    for img_file in test_images:
        print(f"\nAnalyzing {img_file}...")
        
        # Load and preprocess image
        img_path = os.path.join(images_dir, img_file)
        img_tensor, _ = load_and_preprocess_image(img_path)
        
        # Visualize bottleneck features
        save_path = os.path.join(outputs_dir, f'bottleneck_analysis_{img_file.replace(".tiff", "")}.png')
        visualize_bottleneck_features(
            model, img_tensor, img_file.replace('.tiff', ''), save_path, device
        )
        print(f"  - Saved visualization to: {save_path}")
        
        # Analyze feature importance
        channel_stats = analyze_feature_importance(model, img_tensor, device)
        all_stats.append({
            'image': img_file,
            'stats': channel_stats
        })
        
        # Print top 10 most active channels
        print(f"  - Top 10 most active channels:")
        for i, stat in enumerate(channel_stats[:10]):
            print(f"    Channel {stat['channel']:3d}: abs_mean={stat['abs_mean']:.4f}, "
                  f"std={stat['std']:.4f}")
        
        # Bottleneck manipulation experiments
        print(f"  - Running bottleneck manipulation experiments...")
        
        # Define manipulation functions
        manipulations = [
            ("Zero All Features", lambda x: torch.zeros_like(x)),
            ("Amplify 2x", lambda x: x * 2.0),
            ("Reduce 0.5x", lambda x: x * 0.5),
            ("Add Noise", lambda x: x + torch.randn_like(x) * 0.1),
            ("Top 50% Channels", lambda x: x * (x.abs().mean(dim=(2,3), keepdim=True) > x.abs().mean(dim=(2,3), keepdim=True).median()).float()),
            ("Threshold (>0.5)", lambda x: x * (x.abs() > 0.5).float()),
        ]
        
        # Save manipulation results
        manip_save_path = os.path.join(outputs_dir, f'bottleneck_manipulation_{img_file.replace(".tiff", "")}.png')
        manipulate_bottleneck_features(model, img_tensor, manipulations, manip_save_path, device)
        print(f"  - Saved manipulation results to: {manip_save_path}")
    
    # Save detailed statistics
    print("\nSaving detailed statistics...")
    stats_path = os.path.join(outputs_dir, 'bottleneck_statistics.txt')
    with open(stats_path, 'w') as f:
        f.write("Bottleneck Feature Analysis Report\n")
        f.write("==================================\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: Compact Autoencoder (with residual connection)\n\n")
        
        for img_stats in all_stats:
            f.write(f"Image: {img_stats['image']}\n")
            f.write("-" * 50 + "\n")
            f.write("Channel | Abs Mean | Mean     | Std      | Min      | Max\n")
            f.write("-" * 60 + "\n")
            
            for stat in img_stats['stats'][:20]:  # Show only top 20
                f.write(f"{stat['channel']:7d} | {stat['abs_mean']:8.4f} | "
                       f"{stat['mean']:8.4f} | {stat['std']:8.4f} | "
                       f"{stat['min']:8.4f} | {stat['max']:8.4f}\n")
            f.write("\n")
    
    print("Analysis complete!")
    print(f"Results saved in {outputs_dir}/")


if __name__ == "__main__":
    main()
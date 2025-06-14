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
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# Single-Channel Autoencoder Architecture (unchanged)
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class GrayscaleAnomalyAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder - reduced channel numbers for grayscale
        self.enc1 = ConvBlock(1, 32)  # Input: 1 channel grayscale
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)
        self.enc5 = ConvBlock(256, 512)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.dec5 = ConvBlock(512, 256)
        self.dec4 = ConvBlock(256, 128)
        self.dec3 = ConvBlock(128, 64)
        self.dec2 = ConvBlock(64, 32)
        self.dec1 = ConvBlock(32, 32)
        
        self.final = nn.Conv2d(32, 1, 1)  # Output: 1 channel grayscale
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
    def forward(self, x):
        # Encoding
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e5))
        
        # Decoding
        d5 = self.dec5(self.upsample(b))
        d4 = self.dec4(self.upsample(d5))
        d3 = self.dec3(self.upsample(d4))
        d2 = self.dec2(self.upsample(d3))
        d1 = self.dec1(self.upsample(d2))
        
        return torch.sigmoid(self.final(d1))

# Dataset for Grayscale Images (unchanged)
class MVTecGrayscaleDataset(Dataset):
    def __init__(self, root_dir, category, split='train', transform=None):
        self.root_dir = Path(root_dir) / category / split
        self.transform = transform
        self.images = []
        
        # Load all images (normal and defective)
        for folder in self.root_dir.iterdir():
            if folder.is_dir():
                self.images.extend(list(folder.glob('*.png')))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        # Load image as grayscale
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        return image

# Enhanced Loss Function with minimal complexity
class EnhancedGrayscaleCombinedLoss(nn.Module):
    def __init__(self, ssim_weight=0.7, use_ms_ssim=True):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.l2_weight = 1.0 - ssim_weight
        self.use_ms_ssim = use_ms_ssim
        self.current_epoch = 0
        
        # For your image dimensions (1008x176)
        self.window_aspect_ratio = 1008 / 176
    
    def forward(self, recon, target, epoch=None):
        if epoch is not None:
            self.current_epoch = epoch
            # Dynamic weight adjustment
            progress = epoch / 100.0
            self.ssim_weight = 0.7 - 0.2 * progress  # Gradually increase L2 weight
            self.l2_weight = 1.0 - self.ssim_weight
        
        l2_loss = F.mse_loss(recon, target)
        
        if self.use_ms_ssim:
            ssim_loss = 1 - self.ms_ssim(recon, target)
        else:
            ssim_loss = 1 - self.ssim(recon, target)
        
        total_loss = self.l2_weight * l2_loss + self.ssim_weight * ssim_loss
        
        return {
            'total': total_loss,
            'l2': l2_loss,
            'ssim': ssim_loss,
            'weights': (self.l2_weight, self.ssim_weight)
        }
    
    def ssim(self, x, y, window_size=11):
        C1, C2 = 0.01**2, 0.03**2
        window = self._gaussian_window(window_size).to(x.device)
        
        mu_x = F.conv2d(x, window, padding=window_size//2)
        mu_y = F.conv2d(y, window, padding=window_size//2)
        
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = F.conv2d(x*x, window, padding=window_size//2) - mu_x_sq
        sigma_y_sq = F.conv2d(y*y, window, padding=window_size//2) - mu_y_sq
        sigma_xy = F.conv2d(x*y, window, padding=window_size//2) - mu_xy
        
        ssim_map = ((2*mu_xy + C1)*(2*sigma_xy + C2)) / \
                   ((mu_x_sq + mu_y_sq + C1)*(sigma_x_sq + sigma_y_sq + C2))
        
        return ssim_map.mean()
    
    def ms_ssim(self, x, y, scales=3):
        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001]).to(x.device)
        weights = weights / weights.sum()
        
        ssim_vals = []
        cs_vals = []
        
        for scale in range(scales):
            if scale > 0:
                x = F.avg_pool2d(x, kernel_size=2, stride=2)
                y = F.avg_pool2d(y, kernel_size=2, stride=2)
            
            window_size = max(5, 11 - 2 * scale)
            ssim_val, cs_val = self._ssim_components(x, y, window_size)
            
            ssim_vals.append(ssim_val)
            cs_vals.append(cs_val)
        
        ms_ssim_val = torch.prod(torch.stack(cs_vals[:-1]) ** weights[:-1]) * \
                      (ssim_vals[-1] ** weights[-1])
        
        return ms_ssim_val
    
    def _ssim_components(self, x, y, window_size):
        C1, C2 = 0.01**2, 0.03**2
        window = self._gaussian_window_asymmetric(window_size).to(x.device)
        padding = self._get_asymmetric_padding(window_size)
        
        mu_x = F.conv2d(x, window, padding=padding)
        mu_y = F.conv2d(y, window, padding=padding)
        
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = F.conv2d(x*x, window, padding=padding) - mu_x_sq
        sigma_y_sq = F.conv2d(y*y, window, padding=padding) - mu_y_sq
        sigma_xy = F.conv2d(x*y, window, padding=padding) - mu_xy
        
        l = (2 * mu_xy + C1) / (mu_x_sq + mu_y_sq + C1)
        cs = (2 * sigma_xy + C2) / (sigma_x_sq + sigma_y_sq + C2)
        
        return (l * cs).mean(), cs.mean()
    
    def _gaussian_window(self, size):
        sigma = 1.5
        gauss = torch.Tensor([np.exp(-(x - size//2)**2 / (2 * sigma**2)) 
                             for x in range(size)])
        gauss = gauss / gauss.sum()
        window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        return window.unsqueeze(0).unsqueeze(0)
    
    def _gaussian_window_asymmetric(self, size):
        # Create asymmetric window for wide images
        size_h = size
        size_w = int(size * min(1.5, self.window_aspect_ratio / 4))
        size_w = max(size_w, size)
        
        sigma_h = 1.5
        sigma_w = sigma_h * (size_w / size_h)
        
        gauss_h = torch.Tensor([np.exp(-(x - size_h//2)**2 / (2 * sigma_h**2)) 
                               for x in range(size_h)])
        gauss_w = torch.Tensor([np.exp(-(x - size_w//2)**2 / (2 * sigma_w**2)) 
                               for x in range(size_w)])
        
        gauss_h = gauss_h / gauss_h.sum()
        gauss_w = gauss_w / gauss_w.sum()
        
        window = gauss_h.unsqueeze(1) @ gauss_w.unsqueeze(0)
        return window.unsqueeze(0).unsqueeze(0)
    
    def _get_asymmetric_padding(self, size):
        size_h = size
        size_w = int(size * min(1.5, self.window_aspect_ratio / 4))
        size_w = max(size_w, size)
        return (size_h // 2, size_w // 2)

# Anomaly Detection and Heatmap Generation (simplified)
class GrayscaleAnomalyDetector:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def generate_heatmap(self, image):
        with torch.no_grad():
            # Get reconstruction
            recon = self.model(image)
            
            # Simple pixel-wise difference
            diff = torch.abs(image - recon)
            
            # Convert to numpy and extract single channel
            heatmap_np = diff.cpu().numpy()[0, 0]
            
            # Apply Gaussian smoothing for better visualization
            heatmap_smooth = gaussian_filter(heatmap_np, sigma=4)
            
            # Normalize to [0, 1]
            if heatmap_smooth.max() > heatmap_smooth.min():
                heatmap_smooth = (heatmap_smooth - heatmap_smooth.min()) / (heatmap_smooth.max() - heatmap_smooth.min())
            
            return heatmap_smooth, recon

# Enhanced Training Function
def train_grayscale_model(model, train_loader, num_epochs=100, lr=1e-3, device='cuda'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    # Use enhanced loss function
    criterion = EnhancedGrayscaleCombinedLoss(ssim_weight=0.7, use_ms_ssim=True)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_l2 = 0
        total_ssim = 0
        
        print(f'Epoch {epoch + 1}/{num_epochs}')
        for batch in tqdm(train_loader):
            batch = batch.to(device)
            
            # Forward pass
            recon = model(batch)
            loss_dict = criterion(recon, batch, epoch=epoch)
            
            # Extract loss components
            loss = loss_dict['total']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track all loss components
            total_loss += loss.item()
            total_l2 += loss_dict['l2'].item()
            total_ssim += loss_dict['ssim'].item()
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        avg_l2 = total_l2 / len(train_loader)
        avg_ssim = total_ssim / len(train_loader)
        
        # Enhanced printing with component details
        print(f'  Average Loss: {avg_loss:.4f} (L2: {avg_l2:.4f}, SSIM: {avg_ssim:.4f})')
        print(f'  Current Weights - L2: {criterion.l2_weight:.3f}, SSIM: {criterion.ssim_weight:.3f}')
    
    return model

# Inference functions (unchanged)
def inference_grayscale(model_path, image_path, category, device='cuda'):
    """
    Load a trained model and perform anomaly detection on a single grayscale image
    """
    # Initialize model and load weights
    model = GrayscaleAnomalyAutoencoder()
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Load and preprocess image as grayscale
    image = Image.open(image_path).convert('L')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Initialize detector and generate heatmap
    detector = GrayscaleAnomalyDetector(model, device)
    heatmap, reconstruction = detector.generate_heatmap(image_tensor)
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(image_tensor[0, 0].cpu(), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(reconstruction[0, 0].cpu(), cmap='gray')
    plt.title('Reconstruction')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Anomaly Heatmap')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    output_path = f'grayscale_inference_{category}_{os.path.basename(image_path)}'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Results saved to: {output_path}")
    print(f"Max anomaly score: {heatmap.max():.4f}")
    print(f"Mean anomaly score: {heatmap.mean():.4f}")
    
    return heatmap, reconstruction


# Main execution
def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16  # Can use larger batch size due to reduced memory usage
    num_epochs = 100
    image_size = 1024
    
    # Data transforms for grayscale
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Single value normalization for grayscale
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Initialize model
    model = GrayscaleAnomalyAutoencoder()
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Training for each category
    categories = ['grid']
    
    for category in categories:
        print(f"\nTraining on {category} category (grayscale)...")
        
        # Create dataset and dataloader
        train_dataset = MVTecGrayscaleDataset('/Users/laiyongcheng/Desktop/autoencoder/', category, 'train', transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        # Train model
        trained_model = train_grayscale_model(model, train_loader, num_epochs, device=device)
        
        # Save model
        torch.save(trained_model.state_dict(), f'{category}_grayscale_autoencoder.pth')
        
        # Test anomaly detection
        detector = GrayscaleAnomalyDetector(trained_model, device)
        
        # Load and test on a sample image
        test_dataset = MVTecGrayscaleDataset('/Users/laiyongcheng/Desktop/autoencoder/', category, 'test', transform)
        test_image = test_dataset[0].unsqueeze(0).to(device)
        
        heatmap, reconstruction = detector.generate_heatmap(test_image)
        
        # Visualization
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(test_image[0, 0].cpu(), cmap='gray')
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(reconstruction[0, 0].cpu(), cmap='gray')
        plt.title('Reconstruction')
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Anomaly Heatmap')
        plt.colorbar()
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{category}_grayscale_anomaly_detection.png')
        plt.close()

if __name__ == '__main__':
    main()
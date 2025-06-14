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
from sklearn.metrics import roc_auc_score, average_precision_score
import random
import cv2

# ==================== Loss Functions Module ====================
class ModularLossFunction(nn.Module):
    """Modular loss function framework supporting multiple components"""
    def __init__(self, 
                 use_mse=True, mse_weight=0.3,
                 use_ssim=True, ssim_weight=0.3,
                 use_focal_freq=True, focal_freq_weight=0.2,
                 use_sobel=True, sobel_weight=0.2):
        super().__init__()
        self.use_mse = use_mse
        self.mse_weight = mse_weight
        self.use_ssim = use_ssim
        self.ssim_weight = ssim_weight
        self.use_focal_freq = use_focal_freq
        self.focal_freq_weight = focal_freq_weight
        self.use_sobel = use_sobel
        self.sobel_weight = sobel_weight
        
        # Normalize weights
        total_weight = 0
        if use_mse: total_weight += mse_weight
        if use_ssim: total_weight += ssim_weight
        if use_focal_freq: total_weight += focal_freq_weight
        if use_sobel: total_weight += sobel_weight
        
        if total_weight > 0:
            if use_mse: self.mse_weight /= total_weight
            if use_ssim: self.ssim_weight /= total_weight
            if use_focal_freq: self.focal_freq_weight /= total_weight
            if use_sobel: self.sobel_weight /= total_weight
    
    def forward(self, recon, target):
        losses = {}
        total_loss = 0
        
        if self.use_mse:
            mse_loss = F.mse_loss(recon, target)
            losses['mse'] = mse_loss
            total_loss += self.mse_weight * mse_loss
        
        if self.use_ssim:
            ssim_loss = 1 - self._ssim(recon, target)
            losses['ssim'] = ssim_loss
            total_loss += self.ssim_weight * ssim_loss
        
        if self.use_focal_freq:
            focal_loss = self._focal_frequency_loss(recon, target)
            losses['focal_freq'] = focal_loss
            total_loss += self.focal_freq_weight * focal_loss
        
        if self.use_sobel:
            sobel_loss = self._sobel_gradient_loss(recon, target)
            losses['sobel'] = sobel_loss
            total_loss += self.sobel_weight * sobel_loss
        
        losses['total'] = total_loss
        return losses
    
    def _ssim(self, x, y, window_size=11):
        """Structural Similarity Index"""
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
    
    def _focal_frequency_loss(self, recon, target, alpha=1.0):
        """Focal Frequency Loss for hard-to-reconstruct frequency components"""
        # Convert to frequency domain
        recon_freq = torch.fft.fft2(recon, norm='ortho')
        target_freq = torch.fft.fft2(target, norm='ortho')
        
        # Calculate frequency-wise loss
        freq_loss = torch.abs(recon_freq - target_freq)
        
        # Dynamic weight based on reconstruction difficulty
        weight = torch.pow(freq_loss, alpha)
        
        # Weighted frequency loss
        weighted_loss = weight * freq_loss
        
        return weighted_loss.mean()
    
    def _sobel_gradient_loss(self, recon, target):
        """Sobel gradient loss for edge preservation"""
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        sobel_x = sobel_x.to(recon.device)
        sobel_y = sobel_y.to(target.device)
        
        # Calculate gradients
        recon_grad_x = F.conv2d(recon, sobel_x, padding=1)
        recon_grad_y = F.conv2d(recon, sobel_y, padding=1)
        target_grad_x = F.conv2d(target, sobel_x, padding=1)
        target_grad_y = F.conv2d(target, sobel_y, padding=1)
        
        # Gradient magnitude
        recon_grad = torch.sqrt(recon_grad_x**2 + recon_grad_y**2)
        target_grad = torch.sqrt(target_grad_x**2 + target_grad_y**2)
        
        return F.mse_loss(recon_grad, target_grad)
    
    def _gaussian_window(self, size):
        sigma = 1.5
        gauss = torch.Tensor([np.exp(-(x - size//2)**2 / (2 * sigma**2)) 
                             for x in range(size)])
        gauss = gauss / gauss.sum()
        window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        return window.unsqueeze(0).unsqueeze(0)

# ==================== Synthetic Anomaly Generation ====================
class SyntheticAnomalyGenerator:
    """Generate synthetic anomalies for training"""
    def __init__(self, anomaly_prob=0.3):
        self.anomaly_prob = anomaly_prob
    
    def generate_anomaly(self, image):
        """Generate synthetic anomaly on normal image"""
        if random.random() > self.anomaly_prob:
            return image, torch.zeros_like(image)
        
        # Clone image to avoid modifying original
        anomaly_image = image.clone()
        mask = torch.zeros_like(image)
        
        # Choose anomaly type
        anomaly_type = random.choice(['mask', 'noise', 'blur', 'scratch'])
        
        if anomaly_type == 'mask':
            anomaly_image, mask = self._random_masking(anomaly_image)
        elif anomaly_type == 'noise':
            anomaly_image, mask = self._controlled_noise(anomaly_image)
        elif anomaly_type == 'blur':
            anomaly_image, mask = self._local_blur(anomaly_image)
        elif anomaly_type == 'scratch':
            anomaly_image, mask = self._synthetic_scratch(anomaly_image)
        
        return anomaly_image, mask
    
    def _random_masking(self, image):
        """Random rectangular masking"""
        B, C, H, W = image.shape
        mask = torch.zeros_like(image)
        
        for b in range(B):
            # Random mask size (10-30% of image)
            mask_h = random.randint(int(H * 0.1), int(H * 0.3))
            mask_w = random.randint(int(W * 0.1), int(W * 0.3))
            
            # Random position
            y = random.randint(0, H - mask_h)
            x = random.randint(0, W - mask_w)
            
            # Apply mask
            image[b, :, y:y+mask_h, x:x+mask_w] = random.random()
            mask[b, :, y:y+mask_h, x:x+mask_w] = 1
        
        return image, mask
    
    def _controlled_noise(self, image):
        """Add controlled Gaussian noise to specific regions"""
        B, C, H, W = image.shape
        mask = torch.zeros_like(image)
        
        for b in range(B):
            # Random region
            region_h = random.randint(int(H * 0.1), int(H * 0.2))
            region_w = random.randint(int(W * 0.1), int(W * 0.2))
            y = random.randint(0, H - region_h)
            x = random.randint(0, W - region_w)
            
            # Add noise
            noise = torch.randn(C, region_h, region_w) * 0.3
            image[b, :, y:y+region_h, x:x+region_w] += noise.to(image.device)
            mask[b, :, y:y+region_h, x:x+region_w] = 1
        
        return image, mask
    
    def _local_blur(self, image):
        """Apply local blurring"""
        B, C, H, W = image.shape
        mask = torch.zeros_like(image)
        
        for b in range(B):
            # Random region
            blur_h = random.randint(int(H * 0.1), int(H * 0.2))
            blur_w = random.randint(int(W * 0.1), int(W * 0.2))
            y = random.randint(0, H - blur_h)
            x = random.randint(0, W - blur_w)
            
            # Apply Gaussian blur
            region = image[b, :, y:y+blur_h, x:x+blur_w]
            blurred = F.avg_pool2d(region.unsqueeze(0), 3, 1, 1).squeeze(0)
            image[b, :, y:y+blur_h, x:x+blur_w] = blurred
            mask[b, :, y:y+blur_h, x:x+blur_w] = 1
        
        return image, mask
    
    def _synthetic_scratch(self, image):
        """Generate synthetic scratch defect"""
        B, C, H, W = image.shape
        mask = torch.zeros_like(image)
        
        for b in range(B):
            # Random line parameters
            start_x = random.randint(0, W-1)
            start_y = random.randint(0, H-1)
            end_x = random.randint(0, W-1)
            end_y = random.randint(0, H-1)
            
            # Draw line on numpy array
            img_np = image[b, 0].cpu().numpy()
            mask_np = np.zeros_like(img_np)
            
            # Draw line
            cv2.line(img_np, (start_x, start_y), (end_x, end_y), 
                    color=random.random(), thickness=random.randint(1, 3))
            cv2.line(mask_np, (start_x, start_y), (end_x, end_y), 
                    color=1, thickness=random.randint(1, 3))
            
            image[b, 0] = torch.from_numpy(img_np).to(image.device)
            mask[b, 0] = torch.from_numpy(mask_np).to(image.device)
        
        return image, mask

# ==================== Network Architectures ====================
class BaselineAutoencoder(nn.Module):
    """Standard autoencoder without skip connections"""
    def __init__(self, latent_dim=128):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # 976x176 -> 488x88
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),  # 488x88 -> 244x44
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), # 244x44 -> 122x22
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), # 122x22 -> 61x11
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1), # 61x11 -> 31x6 (approximately)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, latent_dim, 1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_dim, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # Upsample
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        # Bottleneck
        bottleneck = self.bottleneck(encoded)
        # Decode
        decoded = self.decoder(bottleneck)
        return decoded
    
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
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Decoder blocks with skip connections
        self.dec5 = self._conv_block(512 + 512, 256)  # Skip from enc5
        self.dec4 = self._conv_block(256 + 256, 128)  # Skip from enc4
        self.dec3 = self._conv_block(128 + 128, 64)   # Skip from enc3
        self.dec2 = self._conv_block(64 + 64, 32)     # Skip from enc2
        self.dec1 = self._conv_block(32 + 32, 32)     # Skip from enc1
        
        self.final = nn.Conv2d(32, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoding with feature storage
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e5))
        
        # Decoding with skip connections
        d5 = self.dec5(torch.cat([self.upsample(b), e5], dim=1))
        d4 = self.dec4(torch.cat([self.upsample(d5), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upsample(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))
        
        return torch.sigmoid(self.final(d1))
    
    def get_multi_level_features(self, x):
        """Extract features from multiple encoder levels"""
        features = []
        
        e1 = self.enc1(x)
        features.append(F.adaptive_avg_pool2d(e1, 1).squeeze())
        
        e2 = self.enc2(self.pool(e1))
        features.append(F.adaptive_avg_pool2d(e2, 1).squeeze())
        
        e3 = self.enc3(self.pool(e2))
        features.append(F.adaptive_avg_pool2d(e3, 1).squeeze())
        
        e4 = self.enc4(self.pool(e3))
        features.append(F.adaptive_avg_pool2d(e4, 1).squeeze())
        
        e5 = self.enc5(self.pool(e4))
        features.append(F.adaptive_avg_pool2d(e5, 1).squeeze())
        
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
                transforms.RandomRotation(degrees=5),  # ±5 degrees
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
        
        # Generate synthetic anomaly for training
        if self.synthetic_anomaly_generator and self.split == 'train':
            image, anomaly_mask = self.synthetic_anomaly_generator.generate_anomaly(image)
            return image, anomaly_mask
        
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

# ==================== Evaluation Metrics ====================
class AnomalyEvaluator:
    """Evaluate anomaly detection performance"""
    def __init__(self):
        self.predictions = []
        self.ground_truths = []
        self.anomaly_maps = []
        self.ground_truth_masks = []
    
    def add_batch(self, anomaly_scores, labels, anomaly_maps=None, gt_masks=None):
        """Add batch results"""
        self.predictions.extend(anomaly_scores.cpu().numpy())
        self.ground_truths.extend(labels.cpu().numpy())
        
        if anomaly_maps is not None:
            self.anomaly_maps.extend(anomaly_maps.cpu().numpy())
        if gt_masks is not None:
            self.ground_truth_masks.extend(gt_masks.cpu().numpy())
    
    def compute_metrics(self):
        """Compute AUROC, AP, and PRO metrics"""
        metrics = {}
        
        # Image-level metrics
        predictions = np.array(self.predictions)
        ground_truths = np.array(self.ground_truths)
        
        # AUROC
        metrics['auroc'] = roc_auc_score(ground_truths, predictions)
        
        # Average Precision
        metrics['ap'] = average_precision_score(ground_truths, predictions)
        
        # Pixel-level PRO if available
        if self.anomaly_maps and self.ground_truth_masks:
            metrics['pro'] = self._compute_pro()
        
        return metrics
    
    def _compute_pro(self, integration_limit=0.3):
        """Compute Per-Region Overlap (PRO) metric"""
        # Simplified PRO calculation
        pros = []
        
        for anomaly_map, gt_mask in zip(self.anomaly_maps, self.ground_truth_masks):
            if gt_mask.max() == 0:  # Skip normal images
                continue
            
            # Threshold anomaly map at various levels
            thresholds = np.linspace(0, 1, 100)
            overlaps = []
            
            for thresh in thresholds:
                binary_map = anomaly_map > thresh
                overlap = (binary_map & gt_mask).sum() / gt_mask.sum()
                overlaps.append(overlap)
            
            # Integrate up to limit
            pro = np.trapz(overlaps[:int(integration_limit * 100)], 
                          thresholds[:int(integration_limit * 100)])
            pros.append(pro)
        
        return np.mean(pros) if pros else 0

# ==================== Training Function ====================
def train_anomaly_model(model, train_loader, val_loader, config):
    """Train anomaly detection model"""
    device = config['device']
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['num_epochs'])
    
    # Initialize loss function
    criterion = ModularLossFunction(**config['loss_config'])
    
    # Initialize synthetic anomaly generator if enabled
    if config.get('use_synthetic_anomalies', False):
        anomaly_generator = SyntheticAnomalyGenerator(anomaly_prob=0.3)
    
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        # Training
        model.train()
        train_losses = {'total': 0, 'mse': 0, 'ssim': 0, 'focal_freq': 0, 'sobel': 0}
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}'):
            if config.get('use_synthetic_anomalies', False):
                images, anomaly_masks = batch
            else:
                images, _ = batch
                
            images = images.to(device)
            
            # Generate synthetic anomalies if enabled
            if config.get('use_synthetic_anomalies', False) and anomaly_masks.sum() > 0:
                target = images.clone()  # Original clean image
                anomaly_images = images  # Already contains anomalies
            else:
                target = images
                anomaly_images = images
            
            # Forward pass
            recon = model(anomaly_images)
            loss_dict = criterion(recon, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss_dict['total'].backward()
            optimizer.step()
            
            # Update losses
            for key in train_losses:
                if key in loss_dict:
                    train_losses[key] += loss_dict[key].item()
        
        # Average losses
        for key in train_losses:
            train_losses[key] /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images, _ = batch
                images = images.to(device)
                
                recon = model(images)
                loss_dict = criterion(recon, images)
                val_loss += loss_dict['total'].item()
        
        val_loss /= len(val_loader)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Print progress
        print(f"Epoch {epoch+1}: Train Loss: {train_losses['total']:.4f}, Val Loss: {val_loss:.4f}")
        print(f"  Components - MSE: {train_losses['mse']:.4f}, SSIM: {train_losses['ssim']:.4f}, "
              f"Focal: {train_losses['focal_freq']:.4f}, Sobel: {train_losses['sobel']:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{config['save_path']}/best_model.pth")
    
    return model

# ==================== Main Execution ====================
def main():
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
            'use_mse': True, 'mse_weight': 0.3,
            'use_ssim': True, 'ssim_weight': 0.3,
            'use_focal_freq': True, 'focal_freq_weight': 0.2,
            'use_sobel': True, 'sobel_weight': 0.2
        },
        'save_path': './models'
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
    
    # Training on MVTec categories
    categories = ['grid']  # Can be extended
    
    for category in categories:
        print(f"\nTraining on {category} category...")
        
        # Create datasets
        train_dataset = MVTecDataset(
            '/Users/laiyongcheng/Desktop/autoencoder/', 
            category, 
            'train', 
            transform,
            use_augmentation=True,
            synthetic_anomaly_generator=SyntheticAnomalyGenerator() if config['use_synthetic_anomalies'] else None
        )
        
        val_dataset = MVTecDataset(
            '/Users/laiyongcheng/Desktop/autoencoder/', 
            category, 
            'test', 
            transform
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                                shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                              shuffle=False, num_workers=4)
        
        # Train model
        model = train_anomaly_model(model, train_loader, val_loader, config)
        
        # Save final model
        torch.save(model.state_dict(), f"{config['save_path']}/{category}_final_model.pth")
        
        # Evaluate model
        print("\nEvaluating model...")
        evaluator = AnomalyEvaluator()
        latent_analyzer = LatentSpaceAnalyzer(model, config['device'])
        
        # Fit latent space on normal data
        normal_loader = DataLoader(
            MVTecDataset('/Users/laiyongcheng/Desktop/autoencoder/', category, 'train', transform),
            batch_size=config['batch_size'], shuffle=False
        )
        latent_analyzer.fit_normal_features(normal_loader)
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch
                images = images.to(config['device'])
                
                # Reconstruction-based anomaly score
                recon = model(images)
                recon_error = torch.mean((images - recon) ** 2, dim=(1, 2, 3))
                
                # Latent space anomaly score
                latent_scores = latent_analyzer.compute_anomaly_score(images)
                
                # Combined score
                anomaly_scores = recon_error + 0.5 * latent_scores
                
                evaluator.add_batch(anomaly_scores, labels)
        
        # Compute metrics
        metrics = evaluator.compute_metrics()
        print(f"\nResults for {category}:")
        print(f"AUROC: {metrics['auroc']:.4f}")
        print(f"Average Precision: {metrics['ap']:.4f}")

if __name__ == '__main__':
    main()
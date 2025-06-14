"""
Test Modular Implementation
==========================

Simple test script to verify the modularized project works correctly.
"""

import sys
import os
# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Import modular components
from models import BaselineAutoencoder, EnhancedAutoencoder
from losses import MSELoss, SSIMLoss, ModularLossManager
from datasets import MVTecDataset
from utils import SyntheticAnomalyGenerator, train_model
from visualization import AnomalyVisualizer


def test_losses():
    """Test loss functions"""
    print("Testing loss functions...")
    
    # Create dummy tensors
    pred = torch.randn(4, 1, 64, 64)
    target = torch.randn(4, 1, 64, 64)
    
    # Test individual losses
    mse_loss = MSELoss(weight=1.0)
    mse_val = mse_loss(pred, target)
    print(f"MSE Loss: {mse_val.item():.4f}")
    
    ssim_loss = SSIMLoss(weight=1.0, window_size=11)
    ssim_val = ssim_loss(pred, target)
    print(f"SSIM Loss: {ssim_val.item():.4f}")
    
    # Test loss manager
    loss_config = {
        'mse': {'class': MSELoss, 'weight': 0.5},
        'ssim': {'class': SSIMLoss, 'weight': 0.5, 'params': {'window_size': 11}}
    }
    
    loss_manager = ModularLossManager(loss_config)
    losses = loss_manager(pred, target)
    print(f"Combined Loss: {losses['total'].item():.4f}")
    print("✓ Loss functions working correctly\n")


def test_models():
    """Test model architectures"""
    print("Testing models...")
    
    # Test Baseline Autoencoder
    baseline = BaselineAutoencoder(latent_dim=128, input_size=(256, 256))
    x = torch.randn(2, 1, 256, 256)
    y = baseline(x)
    print(f"Baseline: Input shape {x.shape} -> Output shape {y.shape}")
    
    # Test Enhanced Autoencoder
    enhanced = EnhancedAutoencoder()
    y = enhanced(x)
    print(f"Enhanced: Input shape {x.shape} -> Output shape {y.shape}")
    print("✓ Models working correctly\n")


def test_synthetic_anomaly():
    """Test synthetic anomaly generation"""
    print("Testing synthetic anomaly generator...")
    
    generator = SyntheticAnomalyGenerator(anomaly_prob=1.0)  # Force anomaly
    
    # Test single image
    image = torch.randn(1, 256, 256)
    anomaly_image, mask = generator.generate_anomaly(image)
    print(f"Single image: Input shape {image.shape} -> Anomaly shape {anomaly_image.shape}")
    
    # Test batch
    batch = torch.randn(4, 1, 256, 256)
    anomaly_batch, mask_batch = generator.generate_anomaly(batch)
    print(f"Batch: Input shape {batch.shape} -> Anomaly shape {anomaly_batch.shape}")
    print("✓ Synthetic anomaly generator working correctly\n")


def test_training():
    """Test training with small data"""
    print("Testing training pipeline...")
    
    # Configuration
    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'batch_size': 4,
        'num_epochs': 2,
        'lr': 1e-3,
        'image_size': (256, 256),
        'use_synthetic_anomalies': True,
        'loss_config': {
            'mse': {'class': MSELoss, 'weight': 0.7},
            'ssim': {'class': SSIMLoss, 'weight': 0.3}
        },
        'save_path': './test_models'
    }
    
    # Create dummy data
    transform = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create small model
    model = BaselineAutoencoder(latent_dim=64, input_size=config['image_size'])
    
    # Create synthetic training data
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=20, transform=None, generator=None):
            self.size = size
            self.transform = transform
            self.generator = generator
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            # Generate random grayscale image
            image = torch.randn(1, 256, 256)
            
            if self.generator:
                clean_image = image
                anomaly_image, mask = self.generator.generate_anomaly(image)
                return clean_image, anomaly_image, mask
            
            return image, 0
    
    # Create dataset and loader
    dataset = DummyDataset(
        size=20, 
        transform=transform,
        generator=SyntheticAnomalyGenerator(anomaly_prob=0.5)
    )
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Train
    model, history = train_model(model, loader, config)
    
    print(f"Training completed. Final loss: {history['total_loss'][-1]:.4f}")
    print("✓ Training pipeline working correctly\n")
    
    # Cleanup
    import shutil
    if os.path.exists('./test_models'):
        shutil.rmtree('./test_models')


def main():
    """Run all tests"""
    print("="*50)
    print("Testing Modular Anomaly Detection System")
    print("="*50)
    print()
    
    try:
        test_losses()
        test_models()
        test_synthetic_anomaly()
        test_training()
        
        print("="*50)
        print("✓ All tests passed successfully!")
        print("="*50)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
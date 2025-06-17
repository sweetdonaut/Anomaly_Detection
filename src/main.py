"""
Main Training Script
===================

Main entry point for training anomaly detection models.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import json
import multiprocessing
from pathlib import Path
import matplotlib.pyplot as plt

# Import modular components
from models import BaselineAutoencoder, EnhancedAutoencoder
from losses import MSELoss, SSIMLoss, MultiScaleSSIMLoss, SobelGradientLoss, FocalFrequencyLoss
from datasets import MVTecDataset
from utils import SyntheticAnomalyGenerator, LatentSpaceAnalyzer, train_model
from visualization import AnomalyVisualizer


def plot_loss_curves(train_history, save_dir, experiment_name):
    """Plot and save training loss curves"""
    epochs = range(1, len(train_history['total_loss']) + 1)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: Total loss and component losses
    ax1.plot(epochs, train_history['total_loss'], 'b-', label='Total Loss', linewidth=2)
    
    # Plot component losses
    colors = ['r', 'g', 'm', 'c', 'y', 'k']
    for i, (loss_name, values) in enumerate(train_history['component_losses'].items()):
        color = colors[i % len(colors)]
        ax1.plot(epochs, values, f'{color}--', label=f'{loss_name}', linewidth=1.5)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{experiment_name} - Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss weights over time
    for i, (loss_name, _) in enumerate(train_history['component_losses'].items()):
        weights = [w[loss_name] for w in train_history['weights']]
        color = colors[i % len(colors)]
        ax2.plot(epochs, weights, f'{color}-', label=f'{loss_name} weight', linewidth=1.5)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Weight')
    ax2.set_title('Loss Weights Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_dir, 'loss_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save a simpler plot with just the total loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_history['total_loss'], 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title(f'{experiment_name} - Total Loss')
    plt.grid(True, alpha=0.3)
    
    simple_plot_path = os.path.join(save_dir, 'total_loss_curve.png')
    plt.savefig(simple_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Loss curves saved to {plot_path}")


def main():
    """Main training function"""
    
    # Determine optimal number of workers
    cpu_count = multiprocessing.cpu_count()
    optimal_workers = min(4, cpu_count - 1)  # Leave one CPU free
    
    # Configuration
    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'batch_size': 16,
        'num_epochs': 100,
        'lr': 1e-3,
        'image_size': (1024, 1024),  # Now supports square images
        'architecture': 'enhanced',  # 'baseline' or 'enhanced'
        'use_synthetic_anomalies': True,
        
        # Loss function configuration - adjust weights and parameters freely
        'loss_config': {
            # MSE Loss: Basic pixel-level reconstruction
            'mse': {
                'class': MSELoss,
                'weight': 0.3  # Weight: 0-1, all weights will be auto-normalized
            },
            
            # SSIM Loss: Structural similarity preservation
            'ssim': {
                'class': SSIMLoss,
                'weight': 0.3,
                'params': {
                    'window_size': 11,  # Gaussian window size, must be odd
                    'sigma': 1.5  # Gaussian kernel standard deviation
                }
            },
            
            # Focal Frequency Loss: Dynamic focus on hard-to-reconstruct regions
            'focal_freq': {
                'class': FocalFrequencyLoss,
                'weight': 0.2,
                'params': {
                    'alpha': 1.0,  # Spectrum weight scaling factor
                    'patch_factor': 1,  # Image patch factor
                    'ave_spectrum': False,  # Use batch average spectrum
                    'log_matrix': False,  # Apply log to spectrum weights
                    'batch_matrix': False  # Use batch statistics
                }
            },
            
            # Sobel Gradient Loss: Edge information preservation
            'sobel': {
                'class': SobelGradientLoss,
                'weight': 0.2
            }
            
            # Easy to add/remove loss functions
            # Example: Multi-Scale SSIM
            # 'ms_ssim': {
            #     'class': MultiScaleSSIMLoss,
            #     'weight': 0.3,
            #     'params': {'num_scales': 3, 'window_size': 11}
            # }
        },
        
        'save_path': './models',
        'num_workers': optimal_workers  # Dynamic worker count
    }
    
    # Create save directory
    os.makedirs(config['save_path'], exist_ok=True)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Initialize model
    if config['architecture'] == 'baseline':
        model = BaselineAutoencoder(input_size=config['image_size'])
    else:
        model = EnhancedAutoencoder()
    
    print(f"Model architecture: {config['architecture']}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of workers: {config['num_workers']} (detected {cpu_count} CPUs)")
    
    # Training on MVTec categories
    categories = ['grid']  # Can be extended to other categories
    
    for category in categories:
        print(f"\nTraining on {category} category...")
        
        # Create training dataset
        train_dataset = MVTecDataset(
            '/Users/laiyongcheng/Desktop/autoencoder/', 
            category, 
            'train', 
            transform,
            use_augmentation=True,
            synthetic_anomaly_generator=SyntheticAnomalyGenerator() if config['use_synthetic_anomalies'] else None
        )
        
        # Create dataloader
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True, 
            num_workers=config['num_workers']
        )
        
        # Create category-specific directory
        category_dir = os.path.join(config['save_path'], category)
        os.makedirs(category_dir, exist_ok=True)
        
        # Create history subdirectory
        history_dir = os.path.join(category_dir, 'history')
        os.makedirs(history_dir, exist_ok=True)
        
        # Update config with checkpoint directory
        config_with_checkpoint = config.copy()
        config_with_checkpoint['checkpoint_dir'] = os.path.join(category_dir, 'checkpoints')
        
        # Train model
        model, train_history = train_model(model, train_loader, config_with_checkpoint)
        
        # Save final model
        model_path = os.path.join(category_dir, 'final_model.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        # Save training history as CSV
        import pandas as pd
        
        # Prepare data for CSV
        history_data = {
            'epoch': list(range(1, len(train_history['total_loss']) + 1)),
            'total_loss': train_history['total_loss']
        }
        
        # Add component losses
        for loss_name, values in train_history['component_losses'].items():
            history_data[f'{loss_name}_loss'] = values
        
        # Add weights
        for i, weights_dict in enumerate(train_history['weights']):
            for loss_name, weight in weights_dict.items():
                if f'{loss_name}_weight' not in history_data:
                    history_data[f'{loss_name}_weight'] = []
                history_data[f'{loss_name}_weight'].append(weight)
        
        # Create DataFrame and save as CSV
        df = pd.DataFrame(history_data)
        csv_path = os.path.join(history_dir, 'training_history.csv')
        df.to_csv(csv_path, index=False)
        print(f"Training history saved to {csv_path}")
        
        # Plot and save loss curves
        plot_loss_curves(train_history, history_dir, f"{config['architecture']}_{category}")
        
        # Test on available test images (optional)
        print("\nTesting on available images...")
        
        # Check if test directory exists
        test_path = Path('/Users/laiyongcheng/Desktop/autoencoder/') / category / 'test'
        if test_path.exists():
            print(f"Found test directory: {test_path}")
            
            # Create test dataset
            test_dataset = MVTecDataset(
                '/Users/laiyongcheng/Desktop/autoencoder/', 
                category, 
                'test', 
                transform
            )
            
            test_loader = DataLoader(
                test_dataset, 
                batch_size=config['batch_size'], 
                shuffle=False, 
                num_workers=config['num_workers']
            )
            
            # Setup visualization
            visualizer = AnomalyVisualizer(save_dir=f"{config['save_path']}/visualizations_{category}")
            
            # Simple evaluation
            model.eval()
            with torch.no_grad():
                for i, (images, labels) in enumerate(test_loader):
                    if i >= 5:  # Only visualize first 5 batches
                        break
                    
                    images = images.to(config['device'])
                    reconstructions = model(images)
                    
                    # Calculate reconstruction error
                    error_maps = torch.mean((images - reconstructions) ** 2, dim=1)
                    
                    # Visualize first image in batch
                    visualizer.visualize_reconstruction(
                        images[0], 
                        reconstructions[0], 
                        error_maps[0].cpu().numpy(),
                        save_name=f'test_batch_{i}.png',
                        show=False
                    )
            
            print(f"Visualizations saved to {config['save_path']}/visualizations_{category}")
        else:
            print(f"No test directory found at {test_path}")
        
        print(f"\nTraining completed for {category}!")
    
    print("\nAll training completed!")


if __name__ == "__main__":
    main()
"""
Experiment Training Script
==========================

Train multiple model configurations with different loss functions.
Supports CUDA, MPS (Apple Silicon), and CPU.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import json
import multiprocessing
from pathlib import Path
from datetime import datetime

# Import modular components
from models import BaselineAutoencoder, EnhancedAutoencoder
from losses import MSELoss, SSIMLoss, MultiScaleSSIMLoss, SobelGradientLoss, FocalFrequencyLoss
from datasets import MVTecDataset
from utils import SyntheticAnomalyGenerator, LatentSpaceAnalyzer, train_model
from visualization import AnomalyVisualizer


def get_device():
    """Get the best available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS device (Apple Silicon GPU)")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    return device


def create_experiment_name(architecture, loss_name):
    """Create experiment name from configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{architecture}_{loss_name}_{timestamp}"


def setup_loss_configs():
    """Setup different loss configurations for experiments"""
    loss_configs = {
        'mse': {
            'mse': {
                'class': MSELoss,
                'weight': 1.0
            }
        },
        'focal_freq': {
            'focal_freq': {
                'class': FocalFrequencyLoss,
                'weight': 1.0,
                'params': {
                    'alpha': 1.0,
                    'patch_factor': 1,
                    'ave_spectrum': False,
                    'log_matrix': False,
                    'batch_matrix': False
                }
            }
        }
    }
    return loss_configs


def train_experiment(config, experiment_name, base_output_dir):
    """Train a single experiment configuration"""
    
    # Create experiment directory
    experiment_dir = os.path.join(base_output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(experiment_dir, 'config.json')
    with open(config_path, 'w') as f:
        # Convert config to serializable format
        serializable_config = config.copy()
        serializable_config['device'] = str(config['device'])
        # Create a copy of loss config for serialization
        serializable_loss_config = {}
        for loss_name, loss_cfg in config['loss_config'].items():
            serializable_loss_config[loss_name] = {
                'class': loss_cfg['class'].__name__,
                'weight': loss_cfg['weight']
            }
            if 'params' in loss_cfg:
                serializable_loss_config[loss_name]['params'] = loss_cfg['params']
        serializable_config['loss_config'] = serializable_loss_config
        json.dump(serializable_config, f, indent=2)
    
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
    
    model = model.to(config['device'])
    
    print(f"\n{'='*50}")
    print(f"Experiment: {experiment_name}")
    print(f"Architecture: {config['architecture']}")
    print(f"Loss function: {list(config['loss_config'].keys())}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Output directory: {experiment_dir}")
    print(f"{'='*50}\n")
    
    # Training on MVTec grid category
    category = 'grid'
    
    # Create training dataset
    train_dataset = MVTecDataset(
        '/Users/laiyongcheng/VScode/Anomaly_Detection/MVTec_AD_dataset',
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
    
    # Train model
    model, train_history = train_model(model, train_loader, config)
    
    # Save final model
    model_path = os.path.join(experiment_dir, f'{category}_final_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Save training history
    history_path = os.path.join(experiment_dir, f'{category}_training_history.json')
    with open(history_path, 'w') as f:
        # Convert history to serializable format
        serializable_history = {
            'total_loss': train_history['total_loss'],
            'component_losses': train_history['component_losses'],
            'weights': [{k: float(v) for k, v in w.items()} for w in train_history['weights']]
        }
        json.dump(serializable_history, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    # Test on available test images
    print("\nEvaluating on test set...")
    
    # Check if test directory exists
    test_path = Path('/Users/laiyongcheng/VScode/Anomaly_Detection/MVTec_AD_dataset') / category / 'test'
    if test_path.exists():
        # Create test dataset
        test_dataset = MVTecDataset(
            '/Users/laiyongcheng/VScode/Anomaly_Detection/MVTec_AD_dataset',
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
        vis_dir = os.path.join(experiment_dir, 'visualizations')
        visualizer = AnomalyVisualizer(save_dir=vis_dir)
        
        # Simple evaluation
        model.eval()
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                images = images.to(config['device'])
                reconstructions = model(images)
                
                # Calculate reconstruction error
                error_maps = torch.mean((images - reconstructions) ** 2, dim=1)
                
                # Store scores and labels for evaluation
                batch_scores = error_maps.mean(dim=(1, 2)).cpu().numpy()
                all_scores.extend(batch_scores)
                all_labels.extend(labels.numpy())
                
                # Visualize first image in batch for first 5 batches
                if i < 5:
                    visualizer.visualize_reconstruction(
                        images[0],
                        reconstructions[0],
                        error_maps[0].cpu().numpy(),
                        save_name=f'test_batch_{i}.png',
                        show=False
                    )
        
        # Save evaluation results
        eval_results = {
            'scores': all_scores,
            'labels': all_labels,
            'num_samples': len(all_scores)
        }
        
        eval_path = os.path.join(experiment_dir, 'evaluation_results.json')
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        print(f"Evaluation results saved to {eval_path}")
        print(f"Visualizations saved to {vis_dir}")
    
    print(f"\nExperiment {experiment_name} completed!")
    return experiment_dir


def main():
    """Main training function for multiple experiments"""
    
    # Get optimal device
    device = get_device()
    
    # Determine optimal number of workers
    cpu_count = multiprocessing.cpu_count()
    optimal_workers = min(8, cpu_count - 1)  # Leave one CPU free
    print(f"Using {optimal_workers} workers (detected {cpu_count} CPUs)")
    
    # Base configuration
    base_config = {
        'device': device,
        'batch_size': 64,  # Increased for faster training
        'num_epochs': 50,  # Reduced for quick testing
        'lr': 1e-3,
        'image_size': (1024, 1024),
        'use_synthetic_anomalies': True,
        'num_workers': optimal_workers
    }
    
    # Create base output directory
    base_output_dir = './out'
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Get loss configurations
    loss_configs = setup_loss_configs()
    
    # Define experiments
    experiments = [
        ('baseline', 'mse'),
        ('baseline', 'focal_freq'),
        ('enhanced', 'mse'),
        ('enhanced', 'focal_freq')
    ]
    
    # Run all experiments
    for architecture, loss_name in experiments:
        # Create experiment configuration
        config = base_config.copy()
        config['architecture'] = architecture
        config['loss_config'] = loss_configs[loss_name]
        config['save_path'] = base_output_dir
        
        # Create experiment name
        experiment_name = create_experiment_name(architecture, loss_name)
        
        # Train experiment
        try:
            experiment_dir = train_experiment(config, experiment_name, base_output_dir)
            print(f"\nExperiment {experiment_name} completed successfully!")
            print(f"Results saved to: {experiment_dir}")
        except Exception as e:
            print(f"\nError in experiment {experiment_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*50)
    print("All experiments completed!")
    print(f"Results saved in: {base_output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()
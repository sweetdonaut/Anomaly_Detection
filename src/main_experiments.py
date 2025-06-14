"""
Main Experiments Script (Clean Version)
=======================================

Run multiple experiments with different configurations.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from pathlib import Path
from datetime import datetime

# Import modular components
from models import BaselineAutoencoder, EnhancedAutoencoder
from datasets import MVTecDataset
from losses import ModularLossManager
from utils import (
    # Training utilities
    train_model,
    evaluate_model,
    # Data utilities
    SyntheticAnomalyGenerator,
    # Model utilities
    get_device,
    # Experiment management
    create_experiment_name,
    setup_loss_configs,
    create_experiment_directories,
    create_session_summary,
    # File I/O utilities
    save_experiment_config,
    save_training_summary,
    save_training_history_csv,
    save_evaluation_results,
    # Visualization utilities
    plot_loss_curves,
    plot_comparison_curves
)
from visualization import AnomalyVisualizer


def train_experiment(config, experiment_name, base_output_dir):
    """Train a single experiment configuration"""
    
    # Create directories
    dirs = create_experiment_directories(base_output_dir, experiment_name)
    
    # Save configuration
    save_experiment_config(config, os.path.join(dirs['experiment'], 'training_config.json'))
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.ToTensor()  # This already normalizes to [0, 1]
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
    print(f"Output directory: {dirs['experiment']}")
    print(f"{'='*50}\n")
    
    # Training on MVTec grid category
    category = 'grid'
    
    # Create training dataset
    train_dataset = MVTecDataset(
        '/home/yclai/vscode_project/Anomaly_Detection/MVTec_AD_dataset',
        category,
        'train',
        transform,
        use_augmentation=True,
        synthetic_anomaly_generator=SyntheticAnomalyGenerator() if config['use_synthetic_anomalies'] else None
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    
    # Update config with checkpoint directory
    config_with_checkpoint = config.copy()
    config_with_checkpoint['checkpoint_dir'] = dirs['checkpoints']
    
    # Train model
    model, train_history = train_model(model, train_loader, config_with_checkpoint)
    
    # Save model
    model_path = os.path.join(dirs['weights'], 'final_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Save training history and summary
    save_training_history_csv(train_history, os.path.join(dirs['history'], 'training_history.csv'))
    save_training_summary(train_history, config, experiment_name, 
                         os.path.join(dirs['experiment'], 'training_summary.txt'))
    
    # Plot loss curves
    plot_loss_curves(train_history, dirs['history'], experiment_name)
    
    # Evaluate on test set
    eval_results = evaluate_on_test_set(model, config, category, transform, experiment_name, dirs)
    
    print(f"\nExperiment {experiment_name} completed!")
    return dirs['experiment'], train_history, eval_results


def evaluate_on_test_set(model, config, category, transform, experiment_name, dirs):
    """Evaluate model on test set"""
    print("\nEvaluating on test set...")
    
    # Check if test directory exists
    test_path = Path('/home/yclai/vscode_project/Anomaly_Detection/MVTec_AD_dataset') / category / 'test'
    if not test_path.exists():
        print("Test directory not found, skipping evaluation")
        return None
    
    # Create test dataset
    test_dataset = MVTecDataset(
        '/home/yclai/vscode_project/Anomaly_Detection/MVTec_AD_dataset',
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
    vis_dir = os.path.join(dirs['experiment'], 'visualizations')
    visualizer = AnomalyVisualizer(save_dir=vis_dir)
    
    # Initialize loss manager for evaluation
    loss_manager = ModularLossManager(config['loss_config'], config['device'])
    
    # Evaluate using utils function
    scores, labels = evaluate_model(model, test_loader, loss_manager, config['device'])
    
    # Visualize some results
    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= 5:  # Only visualize first 5 batches
                break
                
            images = images.to(config['device'])
            reconstructions = model(images)
            
            # Calculate error maps
            error_maps = torch.mean((images - reconstructions) ** 2, dim=1)
            
            # Visualize first image in batch
            visualizer.visualize_reconstruction(
                images[0],
                reconstructions[0],
                error_maps[0].cpu().numpy(),
                save_name=f'test_batch_{i}.png',
                show=False
            )
    
    # Save evaluation results
    eval_results = save_evaluation_results(scores, labels, dirs['evaluation'], experiment_name)
    print(f"Visualizations saved to {vis_dir}")
    
    return eval_results


def main():
    """Main training function for multiple experiments"""
    
    # Get device and setup
    device = get_device()
    optimal_workers = min(8, os.cpu_count() - 1)
    print(f"Using {optimal_workers} workers (detected {os.cpu_count()} CPUs)")
    
    # Base configuration
    base_config = {
        'device': device,
        'batch_size': 16,
        'num_epochs': 50,
        'lr': 1e-3,
        'image_size': (512, 512),
        'use_synthetic_anomalies': False,  # Disable synthetic anomalies
        'num_workers': optimal_workers
    }
    
    # Create output directory
    project_root = Path(__file__).parent.parent
    base_output_dir = project_root / 'out'
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create session directory
    session_timestamp = datetime.now().strftime("%Y_%m%d_%H%M")
    session_dir = base_output_dir / session_timestamp
    os.makedirs(session_dir, exist_ok=True)
    print(f"\nStarting experiment session: {session_timestamp}")
    print(f"Results will be saved in: {session_dir}")
    
    # Get loss configurations
    loss_configs = setup_loss_configs()
    
    # Define experiments (only baseline model)
    experiments = [
        ('baseline', 'mse'),
        ('baseline', 'mse_ssim'),
        ('baseline', 'focal_freq'), 
        ('baseline', 'mse_focal_freq'),
        ('baseline', 'mse_ssim_focal_freq')
    ]
    
    # Store results for comparison
    all_histories = {}
    all_results = {}
    
    # Run experiments
    for architecture, loss_name in experiments:
        # Create experiment configuration
        config = base_config.copy()
        config['architecture'] = architecture
        config['loss_config'] = loss_configs[loss_name]
        config['save_path'] = str(session_dir)
        
        # Create experiment name
        experiment_name = create_experiment_name(architecture, loss_name)
        
        # Train experiment
        try:
            experiment_dir, train_history, eval_results = train_experiment(
                config, experiment_name, str(session_dir)
            )
            print(f"\nExperiment {experiment_name} completed successfully!")
            print(f"Results saved to: {experiment_dir}")
            
            # Store results for comparison
            all_histories[experiment_name] = train_history
            if eval_results:
                all_results[experiment_name] = eval_results
                
        except Exception as e:
            print(f"\nError in experiment {experiment_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Create comparison plots
    if len(all_histories) > 1:
        comparison_path = os.path.join(str(session_dir), 'loss_comparison.png')
        plot_comparison_curves(all_histories, comparison_path, 
                             title="Training Loss Comparison")
    
    
    # Create session summary
    create_session_summary(str(session_dir), session_timestamp, base_config, experiments)
    
    print("\n" + "="*50)
    print("All experiments completed!")
    print(f"Results saved in: {session_dir}")
    print("="*50)


if __name__ == "__main__":
    main()
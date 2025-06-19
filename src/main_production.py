"""
Main Production Script
=====================

Training on OpticalDataset for production environment.
"""

import torch
import os
from pathlib import Path
from datetime import datetime

# Import modular components
from models import BaselineAutoencoder, EnhancedAutoencoder, CompactAutoencoder, CompactUNetAutoencoder, StandardCompactAutoencoder, C3k2Autoencoder
from datasets import OpticalDataset
from losses import ModularLossManager
from utils import (
    # Training utilities
    train_model,
    # Data utilities
    SyntheticAnomalyGenerator,
    create_optical_dataloader,
    evaluate_optical_model,
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
import matplotlib.pyplot as plt


def save_training_samples(train_loader, save_dir, num_samples=5):
    """Save sample training images after transforms"""
    import matplotlib.pyplot as plt
    
    saved_count = 0
    
    for batch_idx, batch_data in enumerate(train_loader):
        if saved_count >= num_samples:
            break
            
        # Handle different return formats from OpticalDataset
        if isinstance(batch_data, tuple) and len(batch_data) == 3:
            # Training with synthetic anomalies: (clean_images, anomaly_images, masks)
            clean_images = batch_data[0]
            anomaly_images = batch_data[1]
            masks = batch_data[2]
            has_anomalies = True
        elif isinstance(batch_data, tuple) and len(batch_data) == 2:
            # Normal mode with dummy labels: (images, labels)
            clean_images = batch_data[0]
            anomaly_images = None
            has_anomalies = False
        else:
            # Just images
            clean_images = batch_data
            anomaly_images = None
            has_anomalies = False
            
        # Save individual images from the batch
        for img_idx in range(min(clean_images.shape[0], num_samples - saved_count)):
            if has_anomalies and anomaly_images is not None:
                # Create combined figure with 3 images
                fig, axes = plt.subplots(1, 3, figsize=(6, 8))
                
                # Original image
                clean_img = clean_images[img_idx].cpu()
                if clean_img.shape[0] == 1:
                    clean_img = clean_img.squeeze(0)
                axes[0].imshow(clean_img.numpy(), cmap='gray', vmin=0, vmax=1)
                axes[0].set_title('Original', fontsize=10)
                axes[0].axis('off')
                
                # Anomaly image
                anomaly_img = anomaly_images[img_idx].cpu()
                if anomaly_img.shape[0] == 1:
                    anomaly_img = anomaly_img.squeeze(0)
                axes[1].imshow(anomaly_img.numpy(), cmap='gray', vmin=0, vmax=1)
                axes[1].set_title('With Anomaly', fontsize=10)
                axes[1].axis('off')
                
                # Mask
                mask = masks[img_idx].cpu()
                if mask.shape[0] == 1:
                    mask = mask.squeeze(0)
                axes[2].imshow(mask.numpy(), cmap='hot', vmin=0, vmax=1)
                axes[2].set_title('Anomaly Mask', fontsize=10)
                axes[2].axis('off')
                
                # Add main title
                fig.suptitle(f'Training Sample {saved_count + 1}', fontsize=12)
                
                # Adjust layout
                plt.tight_layout()
                
                # Save combined image
                save_path = os.path.join(save_dir, f'train_sample_{saved_count + 1:02d}.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
                plt.close()
            else:
                # If no anomalies, just save the clean image
                clean_img = clean_images[img_idx].cpu()
                if clean_img.shape[0] == 1:
                    clean_img = clean_img.squeeze(0)
                
                plt.figure(figsize=(4, 10))
                plt.imshow(clean_img.numpy(), cmap='gray', vmin=0, vmax=1)
                plt.title(f'Training Sample {saved_count + 1}')
                plt.axis('off')
                save_path = os.path.join(save_dir, f'train_sample_{saved_count + 1:02d}.png')
                plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
                plt.close()
            
            saved_count += 1
            if saved_count >= num_samples:
                break
    
    print(f"Saved {saved_count} training samples to {save_dir}")
    if has_anomalies:
        print(f"  - Combined images showing: Original | With Anomaly | Mask")


def train_experiment(config, experiment_name, base_output_dir, train_img_saved=False):
    """Train a single experiment configuration"""
    
    # Create directories
    dirs = create_experiment_directories(base_output_dir, experiment_name)
    
    # Save configuration
    save_experiment_config(config, os.path.join(dirs['experiment'], 'training_config.json'))
    
    # Data transforms - OpticalDataset already returns tensors, no transform needed
    transform = None
    
    # Initialize model
    if config['architecture'] == 'baseline':
        model = BaselineAutoencoder(input_size=config['image_size'])
    elif config['architecture'] == 'enhanced':
        model = EnhancedAutoencoder()
    elif config['architecture'] == 'compact':
        model = CompactAutoencoder(input_size=config['image_size'])
    elif config['architecture'] == 'compact_unet':
        model = CompactUNetAutoencoder(input_size=config['image_size'])
    elif config['architecture'] == 'standard_compact':
        model = StandardCompactAutoencoder(input_size=config['image_size'])
    elif config['architecture'] == 'c3k2':
        model = C3k2Autoencoder(input_size=config['image_size'])
    else:
        raise ValueError(f"Unknown architecture: {config['architecture']}")
    
    model = model.to(config['device'])
    
    print(f"\n{'='*50}")
    print(f"Experiment: {experiment_name}")
    print(f"Architecture: {config['architecture']}")
    print(f"Loss function: {list(config['loss_config'].keys())}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Output directory: {dirs['experiment']}")
    print(f"{'='*50}\n")
    
    # Create training dataset
    train_dataset = OpticalDataset(
        '/home/yclai/vscode_project/Anomaly_Detection/OpticalDataset',
        split='train',
        transform=transform,
        use_augmentation=True,
        synthetic_anomaly_generator=SyntheticAnomalyGenerator() if config['use_synthetic_anomalies'] else None
    )
    
    train_loader = create_optical_dataloader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    
    # Save sample training images only for the first experiment
    if not train_img_saved:
        train_img_dir = os.path.join(base_output_dir, 'train_img')
        os.makedirs(train_img_dir, exist_ok=True)
        save_training_samples(train_loader, train_img_dir, num_samples=5)
    
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
    eval_results = evaluate_on_test_set(model, config, transform, experiment_name, dirs)
    
    print(f"\nExperiment {experiment_name} completed!")
    return dirs['experiment'], train_history, eval_results


def evaluate_on_test_set(model, config, transform, experiment_name, dirs):
    """Evaluate model on test set"""
    print("\nEvaluating on test set...")
    
    # Create test dataset
    test_dataset = OpticalDataset(
        '/home/yclai/vscode_project/Anomaly_Detection/OpticalDataset',
        split='test',
        transform=transform
    )
    
    test_loader = create_optical_dataloader(
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
    
    # Evaluate using optical-specific function
    scores, labels = evaluate_optical_model(model, test_loader, loss_manager, config['device'])
    
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
        'num_epochs': 100,  # Production training
        'lr': 1e-3,
        'image_size': (176, 976),
        'use_synthetic_anomalies': True,
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
    
    # Define experiments - all available loss combinations
    # experiments = [
    #     # Basic losses
    #     ('baseline', 'mse'),
    #     ('baseline', 'ssim'),
    #     ('baseline', 'ms_ssim'),
    #     ('baseline', 'sobel'),
        
    #     # Two-component combinations
    #     ('baseline', 'mse_ssim'),
    #     ('baseline', 'mse_ms_ssim'),
    #     ('baseline', 'mse_sobel'),
    #     ('baseline', 'ssim_sobel'),
    #     ('baseline', 'ms_ssim_sobel'),
        
    #     # Three-component combinations
    #     ('baseline', 'mse_ms_ssim_sobel'),
    #     ('baseline', 'mse_ssim_focal_freq'),
        
    #     # With focal frequency
    #     ('baseline', 'focal_freq'),
    #     ('baseline', 'mse_focal_freq'),
        
    #     # Comprehensive (4 components)
    #     ('baseline', 'comprehensive'),
    # ]
    
    # You can uncomment specific experiments to run:
    experiments = [
        ('c3k2', 'mse'),  # Test the C3k2 architecture
    ]
    
    # Store results for comparison
    all_histories = {}
    all_results = {}
    train_img_saved = False  # Track if training images have been saved
    
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
                config, experiment_name, str(session_dir), train_img_saved
            )
            train_img_saved = True  # Mark that training images have been saved
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
"""
Main Production Script - Triplet Version
========================================

Training on OpticalDatasetTriplet with reference-guided loss.
"""

import torch
import os
import numpy as np
from pathlib import Path
from datetime import datetime

# Import modular components
from models import BaselineAutoencoder, EnhancedAutoencoder, CompactAutoencoder, CompactUNetAutoencoder, StandardCompactAutoencoder
from datasets import OpticalDatasetTriplet
from torch.utils.data import DataLoader
from losses.triplet_manager import TripletLossManager
from utils import (
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


def save_triplet_training_samples(train_loader, save_dir, num_samples=3):
    """Save sample triplet training images"""
    import matplotlib.pyplot as plt
    
    saved_count = 0
    
    for batch_idx, batch_data in enumerate(train_loader):
        if saved_count >= num_samples:
            break
        
        targets = batch_data['target']
        ref1s = batch_data['reference1']
        ref2s = batch_data['reference2']
        
        # Save individual triplets from the batch
        for img_idx in range(min(targets.shape[0], num_samples - saved_count)):
            # Create figure with 3 columns
            fig, axes = plt.subplots(1, 3, figsize=(6, 8))
            
            # Target (with blur)
            target = targets[img_idx].cpu()
            if target.shape[0] == 1:
                target = target.squeeze(0)
            axes[0].imshow(target.numpy(), cmap='gray', vmin=0, vmax=1)
            axes[0].set_title('Target (Blurred)', fontsize=10)
            axes[0].axis('off')
            
            # Reference 1
            ref1 = ref1s[img_idx].cpu()
            if ref1.shape[0] == 1:
                ref1 = ref1.squeeze(0)
            axes[1].imshow(ref1.numpy(), cmap='gray', vmin=0, vmax=1)
            axes[1].set_title('Reference 1', fontsize=10)
            axes[1].axis('off')
            
            # Reference 2
            ref2 = ref2s[img_idx].cpu()
            if ref2.shape[0] == 1:
                ref2 = ref2.squeeze(0)
            axes[2].imshow(ref2.numpy(), cmap='gray', vmin=0, vmax=1)
            axes[2].set_title('Reference 2', fontsize=10)
            axes[2].axis('off')
            
            # Add main title
            fig.suptitle(f'Triplet Training Sample {saved_count + 1}', fontsize=12)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save
            save_path = os.path.join(save_dir, f'triplet_sample_{saved_count + 1:02d}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            
            saved_count += 1
            if saved_count >= num_samples:
                break
    
    print(f"Saved {saved_count} triplet training samples to {save_dir}")


def train_triplet_model(model, train_loader, config):
    """
    Train model with triplet data
    Modified version of train_model for triplet format
    """
    from tqdm import tqdm
    
    # Setup training components
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config['num_epochs']
    )
    loss_manager = TripletLossManager(config['loss_config'], config['device'])
    
    # Training history (use 'total_loss' for compatibility with save_training_history_csv)
    history = {
        'total_loss': [],  # Changed from 'train_loss' for compatibility
        'train_loss_components': {k: [] for k in config['loss_config'].keys()},
        'lr': []
    }
    
    # Training loop
    model.train()
    for epoch in range(config['num_epochs']):
        epoch_loss = 0.0
        epoch_loss_components = {k: 0.0 for k in config['loss_config'].keys()}
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}') as pbar:
            for batch_idx, batch_data in enumerate(pbar):
                # Move data to device
                target = batch_data['target'].to(config['device'])
                batch_data['reference1'] = batch_data['reference1'].to(config['device'])
                batch_data['reference2'] = batch_data['reference2'].to(config['device'])
                
                # Forward pass - input is target
                optimizer.zero_grad()
                output = model(target)
                
                # Calculate loss - pass batch_data dict for reference access
                total_loss, loss_components = loss_manager(output, batch_data)
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                # Update metrics
                epoch_loss += total_loss.item()
                for k, v in loss_components.items():
                    if k != 'total' and k in epoch_loss_components:
                        epoch_loss_components[k] += v.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{total_loss.item():.4f}',
                    **{k: f'{v.item():.4f}' for k, v in loss_components.items() if k != 'total'}
                })
        
        # Average losses
        avg_epoch_loss = epoch_loss / len(train_loader)
        history['total_loss'].append(avg_epoch_loss)
        
        for k in epoch_loss_components:
            avg_component_loss = epoch_loss_components[k] / len(train_loader)
            history['train_loss_components'][k].append(avg_component_loss)
        
        # Record learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        
        # Step scheduler
        if scheduler:
            scheduler.step()
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{config['num_epochs']} - "
              f"Loss: {avg_epoch_loss:.4f} - "
              f"LR: {current_lr:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0 and 'checkpoint_dir' in config:
            checkpoint_path = os.path.join(config['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'loss': avg_epoch_loss,
                'history': history
            }, checkpoint_path)
    
    return model, history


def train_experiment(config, experiment_name, base_output_dir, train_img_saved=False):
    """Train a single triplet experiment"""
    
    # Create directories
    dirs = create_experiment_directories(base_output_dir, experiment_name)
    
    # Save configuration
    save_experiment_config(config, os.path.join(dirs['experiment'], 'training_config.json'))
    
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
    else:
        raise ValueError(f"Unknown architecture: {config['architecture']}")
    
    model = model.to(config['device'])
    
    print(f"\n{'='*50}")
    print(f"Triplet Experiment: {experiment_name}")
    print(f"Architecture: {config['architecture']}")
    print(f"Loss function: {list(config['loss_config'].keys())}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Output directory: {dirs['experiment']}")
    print(f"{'='*50}\n")
    
    # Create triplet datasets
    train_dataset = OpticalDatasetTriplet(
        root_dir=config['dataset_path'],
        mode='train',
        image_size=config['image_size']
    )
    
    test_dataset = OpticalDatasetTriplet(
        root_dir=config['dataset_path'],
        mode='test',
        image_size=config['image_size']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Save sample training images only for the first experiment
    if not train_img_saved:
        train_img_dir = os.path.join(base_output_dir, 'triplet_train_img')
        os.makedirs(train_img_dir, exist_ok=True)
        save_triplet_training_samples(train_loader, train_img_dir, num_samples=3)
    
    # Update config with checkpoint directory
    config_with_checkpoint = config.copy()
    config_with_checkpoint['checkpoint_dir'] = dirs['checkpoints']
    
    # Train model
    model, train_history = train_triplet_model(model, train_loader, config_with_checkpoint)
    
    # Save model
    model_path = os.path.join(dirs['weights'], 'final_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Save training history and summary
    # Convert train_history format for compatibility
    compatible_history = {
        'total_loss': train_history['total_loss'],
        'component_losses': train_history['train_loss_components'],
        'lr': train_history['lr'],
        'weights': [{k: v['weight'] for k, v in config['loss_config'].items()} 
                   for _ in range(len(train_history['total_loss']))]  # Fixed weights for all epochs
    }
    save_training_history_csv(compatible_history, os.path.join(dirs['history'], 'training_history.csv'))
    save_training_summary(compatible_history, config, experiment_name, 
                         os.path.join(dirs['experiment'], 'training_summary.txt'))
    
    # Plot loss curves - use compatible_history which has the correct format
    plot_loss_curves(compatible_history, dirs['history'], experiment_name)
    
    # Evaluate on test set
    eval_results = evaluate_on_test_set(model, config, test_loader, experiment_name, dirs)
    
    print(f"\nTriplet experiment {experiment_name} completed!")
    return dirs['experiment'], train_history, eval_results


def evaluate_on_test_set(model, config, test_loader, experiment_name, dirs):
    """Evaluate triplet model on test set with comprehensive visualization"""
    print("\nEvaluating on test set...")
    
    # Setup visualization directories
    vis_dir = os.path.join(dirs['experiment'], 'visualizations')
    vis_full_dir = os.path.join(vis_dir, 'full_images')
    vis_patch_dir = os.path.join(vis_dir, 'patches')
    os.makedirs(vis_full_dir, exist_ok=True)
    os.makedirs(vis_patch_dir, exist_ok=True)
    visualizer = AnomalyVisualizer(save_dir=vis_dir)
    
    # Initialize loss manager for evaluation
    loss_manager = TripletLossManager(config['loss_config'], config['device'])
    
    # Evaluate
    model.eval()
    total_loss = 0.0
    all_scores = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            # Move data to device
            target = batch_data['target'].to(config['device'])
            batch_data['reference1'] = batch_data['reference1'].to(config['device'])
            batch_data['reference2'] = batch_data['reference2'].to(config['device'])
            
            # Forward pass
            output = model(target)
            
            # Calculate loss
            loss, _ = loss_manager(output, batch_data)
            total_loss += loss.item()
            
            # Calculate anomaly scores (MSE between output and references)
            ref1 = batch_data['reference1']
            anomaly_scores = torch.mean((output - ref1) ** 2, dim=(1, 2, 3))
            all_scores.extend(anomaly_scores.cpu().numpy())
            
            # Visualize all images in batch (or first 5)
            num_to_viz = min(output.shape[0], 5)
            for i in range(num_to_viz):
                # Skip test_sample visualization as requested
                
                # Get images
                target_img = target[i].cpu().squeeze()
                output_img = output[i].cpu().squeeze()
                ref1_img = ref1[i].cpu().squeeze()
                ref2_img = batch_data['reference2'][i].cpu().squeeze()
                
                # Calculate differences
                diff_reconstruction = torch.abs(target[i] - output[i]).mean(dim=0).cpu().numpy()
                diff_ref1 = torch.abs(target[i] - ref1[i]).mean(dim=0).cpu().numpy()
                diff_ref2 = torch.abs(target[i] - batch_data['reference2'][i]).mean(dim=0).cpu().numpy()
                
                # Create visualization with manual positioning for exact pixel spacing
                # Image dimensions (176x976) -> aspect ratio
                img_height, img_width = target_img.shape
                aspect_ratio = img_height / img_width  # 976/176 ≈ 5.5
                
                # Settings for each subplot
                subplot_width = 1.5  # inches
                subplot_height = subplot_width * aspect_ratio
                gap_pixels = 10  # pixels between images
                gap_inches = gap_pixels / 150  # convert to inches at 150 dpi
                
                # Calculate total figure size
                total_width = 7 * subplot_width + 6 * gap_inches
                total_height = subplot_height + 0.7  # extra space for larger titles
                
                # Create figure
                fig = plt.figure(figsize=(total_width, total_height))
                
                # Manually create subplots with exact positions
                images = [target_img, ref1_img, ref2_img, output_img, 
                         diff_reconstruction, diff_ref1, diff_ref2]
                titles = ['T', 'ref1', 'ref2', 'r(T)', 'T-r(T)', 'T-ref1', 'T-ref2']
                cmaps = ['gray', 'gray', 'gray', 'gray', 'hot', 'hot', 'hot']
                vmaxs = [1, 1, 1, 1, 0.5, 0.5, 0.5]
                
                axes_list = []
                for idx in range(7):
                    # Calculate position for each subplot
                    left = (idx * (subplot_width + gap_inches)) / total_width
                    bottom = 0.05
                    width = subplot_width / total_width
                    height = subplot_height / total_height
                    
                    ax = fig.add_axes([left, bottom, width, height])
                    ax.imshow(images[idx], cmap=cmaps[idx], vmin=0, vmax=vmaxs[idx])
                    ax.set_title(titles[idx], fontsize=20, pad=5, fontweight='normal', fontfamily='Arial')
                    ax.axis('off')
                    axes_list.append(ax)
                
                # Check if filename contains patch coordinates
                filename = batch_data['filename'][i]
                base_filename = os.path.splitext(os.path.basename(filename))[0]
                
                # If file has patch coordinates, draw rectangle on the full images
                if '#' in filename:
                    import re
                    match = re.search(r'#(\d+)_(\d+)', filename)
                    if match:
                        center_x = int(match.group(1))
                        center_y = int(match.group(2))
                        patch_size = 50
                        half_size = patch_size // 2
                        
                        # Draw rectangles on all 7 images with cyan color
                        from matplotlib.patches import Rectangle
                        for ax_idx in range(7):
                            rect = Rectangle((center_x - half_size, center_y - half_size), 
                                           patch_size, patch_size, 
                                           linewidth=2, edgecolor='cyan', facecolor='none')
                            axes_list[ax_idx].add_patch(rect)
                
                # Save with original filename
                save_path_new = os.path.join(vis_full_dir, f'{base_filename}.png')
                plt.savefig(save_path_new, dpi=150, bbox_inches='tight', pad_inches=0.2)
                plt.close()
                
                # Process patch if coordinates exist
                if '#' in filename:
                    # Extract patch coordinates from filename
                    import re
                    match = re.search(r'#(\d+)_(\d+)', filename)
                    if match:
                        center_x = int(match.group(1))
                        center_y = int(match.group(2))
                        patch_size = 50
                        half_size = patch_size // 2
                        
                        # Extract patches from all images
                        # Calculate boundaries
                        y_start = max(0, center_y - half_size)
                        y_end = min(target_img.shape[0], center_y + half_size)
                        x_start = max(0, center_x - half_size)
                        x_end = min(target_img.shape[1], center_x + half_size)
                        
                        # Extract patches
                        target_patch = target_img[y_start:y_end, x_start:x_end]
                        output_patch = output_img[y_start:y_end, x_start:x_end]
                        ref1_patch = ref1_img[y_start:y_end, x_start:x_end]
                        ref2_patch = ref2_img[y_start:y_end, x_start:x_end]
                        diff_recon_patch = diff_reconstruction[y_start:y_end, x_start:x_end]
                        diff_ref1_patch = diff_ref1[y_start:y_end, x_start:x_end]
                        diff_ref2_patch = diff_ref2[y_start:y_end, x_start:x_end]
                        
                        # Create patch visualization with same layout (original proportions)
                        fig_patch, axes_patch = plt.subplots(1, 7, figsize=(14, 3))
                        
                        # 1. T patch
                        axes_patch[0].imshow(target_patch, cmap='gray', vmin=0, vmax=1)
                        axes_patch[0].set_title(f'T patch ({center_x},{center_y})', fontsize=9, pad=3)
                        axes_patch[0].axis('off')
                        
                        # 2. ref1 patch
                        axes_patch[1].imshow(ref1_patch, cmap='gray', vmin=0, vmax=1)
                        axes_patch[1].set_title('ref1 patch', fontsize=9, pad=3)
                        axes_patch[1].axis('off')
                        
                        # 3. ref2 patch
                        axes_patch[2].imshow(ref2_patch, cmap='gray', vmin=0, vmax=1)
                        axes_patch[2].set_title('ref2 patch', fontsize=9, pad=3)
                        axes_patch[2].axis('off')
                        
                        # 4. r(T) patch
                        axes_patch[3].imshow(output_patch, cmap='gray', vmin=0, vmax=1)
                        axes_patch[3].set_title('r(T) patch', fontsize=9, pad=3)
                        axes_patch[3].axis('off')
                        
                        # 5. T-r(T) patch
                        axes_patch[4].imshow(diff_recon_patch, cmap='hot', vmin=0, vmax=0.5)
                        axes_patch[4].set_title('T-r(T) patch', fontsize=9, pad=3)
                        axes_patch[4].axis('off')
                        
                        # 6. T-ref1 patch
                        axes_patch[5].imshow(diff_ref1_patch, cmap='hot', vmin=0, vmax=0.5)
                        axes_patch[5].set_title('T-ref1 patch', fontsize=9, pad=3)
                        axes_patch[5].axis('off')
                        
                        # 7. T-ref2 patch
                        axes_patch[6].imshow(diff_ref2_patch, cmap='hot', vmin=0, vmax=0.5)
                        axes_patch[6].set_title('T-ref2 patch', fontsize=9, pad=3)
                        axes_patch[6].axis('off')
                        
                        # Adjust spacing - use original spacing
                        plt.subplots_adjust(wspace=0.02, hspace=0.02)
                        save_path_patch = os.path.join(vis_patch_dir, f'{base_filename}_patch.png')
                        plt.savefig(save_path_patch, dpi=150, bbox_inches='tight', pad_inches=0.1)
                        plt.close()
    
    avg_test_loss = total_loss / len(test_loader)
    print(f"Average test loss: {avg_test_loss:.4f}")
    
    # Skip distribution plot as requested by user
    
    # Save evaluation results
    eval_results = {
        'test_loss': avg_test_loss,
        'anomaly_scores': all_scores
    }
    
    # Save to file
    import json
    eval_path = os.path.join(dirs['evaluation'], 'evaluation_results.json')
    with open(eval_path, 'w') as f:
        json.dump({
            'test_loss': avg_test_loss,
            'num_samples': len(all_scores),
            'mean_score': float(np.mean(all_scores)) if all_scores else 0,
            'std_score': float(np.std(all_scores)) if all_scores else 0,
            'min_score': float(np.min(all_scores)) if all_scores else 0,
            'max_score': float(np.max(all_scores)) if all_scores else 0
        }, f, indent=2)
    
    print(f"Visualizations saved to {vis_dir}")
    print(f"- Full images in 'full_images/' with 10-pixel gaps between images")
    print(f"- Patches in 'patches/' if filename contains #x_y coordinates")
    print(f"- Format: T, ref1, ref2, r(T), T-r(T), T-ref1, T-ref2")
    
    return eval_results


def main():
    """Main training function for triplet experiments"""
    
    # Get device and setup
    device = get_device()
    optimal_workers = min(8, os.cpu_count() - 1)
    print(f"Using {optimal_workers} workers (detected {os.cpu_count()} CPUs)")
    
    # Base configuration for triplet training
    base_config = {
        'device': device,
        'batch_size': 8,  # Smaller batch size for triplet data
        'num_epochs': 2,  # Very short for quick testing
        'lr': 1e-3,
        'image_size': (976, 176),  # Note: H x W for triplet dataset
        'dataset_path': '../triplet_dataset',
        'num_workers': optimal_workers
    }
    
    # Create output directory
    project_root = Path(__file__).parent.parent
    base_output_dir = project_root / 'out_triplet'
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create session directory
    session_timestamp = datetime.now().strftime("%Y_%m%d_%H%M")
    session_dir = base_output_dir / session_timestamp
    os.makedirs(session_dir, exist_ok=True)
    print(f"\nStarting triplet experiment session: {session_timestamp}")
    print(f"Results will be saved in: {session_dir}")
    
    # Get loss configurations
    loss_configs = setup_loss_configs()
    
    # Define triplet experiments - 只訓練一個模型配一個 loss
    experiments = [
        ('baseline', 'trip_mse_ssim'),  # 只使用這個組合
    ]
    
    # Store results for comparison
    all_histories = {}
    all_results = {}
    train_img_saved = False
    
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
            train_img_saved = True
            print(f"\nTriplet experiment {experiment_name} completed successfully!")
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
                             title="Triplet Training Loss Comparison")
    
    # Create session summary
    create_session_summary(str(session_dir), session_timestamp, base_config, experiments)
    
    print("\n" + "="*50)
    print("All triplet experiments completed!")
    print(f"Results saved in: {session_dir}")
    print("="*50)


if __name__ == "__main__":
    main()
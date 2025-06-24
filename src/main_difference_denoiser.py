"""
Main Training Script for Difference Denoiser
============================================

Specialized training script for the dual-output difference denoiser model.
Based on main_production_triplet.py but adapted for the specific needs of this model.
"""

import torch
import os
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import model and loss
from models.difference_denoiser import DifferenceDenoiser
from losses.difference_denoiser_loss import DifferenceDenoiserTripletLoss

# Import dataset
from datasets import OpticalDatasetTriplet
from torch.utils.data import DataLoader

# Import utilities
from utils import (
    get_device,
    create_experiment_directories,
    save_experiment_config,
    save_training_history_csv,
    save_training_summary,
    plot_loss_curves
)

# Import visualization
from visualization import AnomalyVisualizer


def save_difference_training_samples(train_loader, save_dir, num_samples=3):
    """Save sample difference images and model inputs"""
    import matplotlib.pyplot as plt
    
    saved_count = 0
    
    for batch_idx, batch_data in enumerate(train_loader):
        if saved_count >= num_samples:
            break
        
        targets = batch_data['target']
        ref1s = batch_data['reference1']
        ref2s = batch_data['reference2']
        
        # Compute differences
        diff1 = targets - ref1s
        diff2 = targets - ref2s
        ref_diff = ref1s - ref2s
        
        # Save samples
        for img_idx in range(min(targets.shape[0], num_samples - saved_count)):
            # Get filename without extension and path
            filename = batch_data['filename'][img_idx]
            basename = os.path.splitext(os.path.basename(filename))[0]
            
            # Create figure with 6 subplots (3 original + 3 differences)
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            
            # Top row: Original images
            images = [targets[img_idx], ref1s[img_idx], ref2s[img_idx]]
            titles = ['Target', 'Reference 1', 'Reference 2']
            
            for i, (img, title) in enumerate(zip(images, titles)):
                img_cpu = img.cpu().squeeze(0)
                axes[0, i].imshow(img_cpu.numpy(), cmap='gray', vmin=0, vmax=1)
                axes[0, i].set_title(title)
                axes[0, i].axis('off')
            
            # Bottom row: Difference images
            diffs = [diff1[img_idx], diff2[img_idx], ref_diff[img_idx]]
            diff_titles = ['Target - Ref1', 'Target - Ref2', 'Ref1 - Ref2']
            
            for i, (diff, title) in enumerate(zip(diffs, diff_titles)):
                diff_cpu = diff.cpu().squeeze(0)
                # Use diverging colormap for differences
                im = axes[1, i].imshow(diff_cpu.numpy(), cmap='RdBu_r', vmin=-0.5, vmax=0.5)
                axes[1, i].set_title(title)
                axes[1, i].axis('off')
                # Add colorbar
                plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
            
            fig.suptitle(f'Training Sample: {basename}', fontsize=14)
            plt.tight_layout()
            
            save_path = os.path.join(save_dir, f'{basename}_diff.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            saved_count += 1
            if saved_count >= num_samples:
                break
    
    print(f"Saved {saved_count} difference training samples to {save_dir}")


def train_difference_denoiser(model, train_loader, config):
    """
    Train the difference denoiser model
    """
    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config['num_epochs']
    )
    
    # Setup loss function
    loss_fn = DifferenceDenoiserTripletLoss(
        weight=1.0,
        recon_weight=config.get('recon_weight', 1.0),
        anomaly_weight=config.get('anomaly_weight', 0.5),
        smooth_weight=config.get('smooth_weight', 0.01)
    )
    
    # Training history
    history = {
        'total_loss': [],
        'component_losses': {
            'reconstruction': [],
            'anomaly': [],
            'smoothness': []
        },
        'weights': [],
        'lr': []
    }
    
    # Training loop
    model.train()
    for epoch in range(config['num_epochs']):
        epoch_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'anomaly': 0.0,
            'smoothness': 0.0
        }
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}') as pbar:
            for batch_idx, batch_data in enumerate(pbar):
                # Move data to device
                target = batch_data['target'].to(config['device'])
                ref1 = batch_data['reference1'].to(config['device'])
                ref2 = batch_data['reference2'].to(config['device'])
                
                # Forward pass
                optimizer.zero_grad()
                anomaly_map, reconstructed_diffs, input_diffs = model(target, ref1, ref2)
                
                # Compute loss
                loss, loss_dict = loss_fn(
                    (anomaly_map, reconstructed_diffs, input_diffs),
                    batch_data
                )
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update metrics
                for key in epoch_losses:
                    if key in loss_dict:
                        epoch_losses[key] += loss_dict[key].item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'recon': f'{loss_dict["reconstruction"].item():.4f}',
                    'anomaly': f'{loss_dict["anomaly"].item():.4f}'
                })
        
        # Average losses
        num_batches = len(train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        # Update history
        history['total_loss'].append(epoch_losses['total'])
        history['component_losses']['reconstruction'].append(epoch_losses['reconstruction'])
        history['component_losses']['anomaly'].append(epoch_losses['anomaly'])
        history['component_losses']['smoothness'].append(epoch_losses['smoothness'])
        
        # Fixed weights for this version
        history['weights'].append({
            'reconstruction': config.get('recon_weight', 1.0),
            'anomaly': config.get('anomaly_weight', 0.5),
            'smoothness': config.get('smooth_weight', 0.01)
        })
        
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Update scheduler
        scheduler.step()
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{config['num_epochs']} Summary:")
        print(f"  Total Loss: {epoch_losses['total']:.6f}")
        print(f"  Reconstruction: {epoch_losses['reconstruction']:.6f}")
        print(f"  Anomaly: {epoch_losses['anomaly']:.6f}")
        print(f"  Smoothness: {epoch_losses['smoothness']:.6f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0 and 'checkpoint_dir' in config:
            checkpoint_path = os.path.join(
                config['checkpoint_dir'],
                f'checkpoint_epoch_{epoch+1}.pth'
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history
            }, checkpoint_path)
            print(f"  Checkpoint saved to {checkpoint_path}")
    
    return model, history


def evaluate_difference_denoiser(model, test_loader, device, save_dir):
    """
    Evaluate the difference denoiser and save visualizations
    """
    model.eval()
    all_scores = []
    
    # Create subdirectories for full images and patches
    full_dir = os.path.join(save_dir, 'full')
    patch_dir = os.path.join(save_dir, 'patches')
    os.makedirs(full_dir, exist_ok=True)
    os.makedirs(patch_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            if batch_idx >= 10:  # Visualize first 10 batches
                break
            
            # Move data to device
            target = batch_data['target'].to(device)
            ref1 = batch_data['reference1'].to(device)
            ref2 = batch_data['reference2'].to(device)
            
            # Forward pass
            anomaly_map, reconstructed_diffs, input_diffs = model(target, ref1, ref2)
            
            # Compute anomaly scores
            scores = model.get_anomaly_score(anomaly_map)
            all_scores.extend(scores.cpu().numpy())
            
            # Visualize results for each image in batch
            for i in range(len(target)):
                if batch_idx * test_loader.batch_size + i >= 10:  # Limit total visualizations
                    break
                    
                # Get filename
                filename = batch_data['filename'][i]
                basename = os.path.splitext(os.path.basename(filename))[0]
                
                visualize_difference_results(
                    target[i],
                    ref1[i],
                    ref2[i],
                    anomaly_map[i],
                    reconstructed_diffs[i],
                    input_diffs[i],
                    filename=filename,
                    basename=basename,
                    full_dir=full_dir,
                    patch_dir=patch_dir
                )
    
    return np.array(all_scores)


def visualize_difference_results(target, ref1, ref2, anomaly_map, reconstructed_diffs, input_diffs, 
                               filename, basename, full_dir, patch_dir):
    """
    Create comprehensive visualization of difference denoiser results
    """
    import matplotlib.pyplot as plt
    import re
    from matplotlib.patches import Rectangle
    
    # Convert tensors to numpy for visualization
    target_np = target.cpu().squeeze().numpy()
    ref1_np = ref1.cpu().squeeze().numpy()
    ref2_np = ref2.cpu().squeeze().numpy()
    anomaly_map_np = anomaly_map.cpu().squeeze().numpy()
    input_diffs_np = input_diffs.cpu().numpy()
    reconstructed_diffs_np = reconstructed_diffs.cpu().numpy()
    
    # Check if filename contains patch coordinates
    has_patch = '#' in filename
    center_x, center_y = None, None
    if has_patch:
        match = re.search(r'#(\d+)_(\d+)', filename)
        if match:
            center_x = int(match.group(1))
            center_y = int(match.group(2))
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Original images
    ax1 = plt.subplot(3, 4, 1)
    ax1.imshow(target_np, cmap='gray')
    ax1.set_title('Target')
    ax1.axis('off')
    
    ax2 = plt.subplot(3, 4, 2)
    ax2.imshow(ref1_np, cmap='gray')
    ax2.set_title('Reference 1')
    ax2.axis('off')
    
    ax3 = plt.subplot(3, 4, 3)
    ax3.imshow(ref2_np, cmap='gray')
    ax3.set_title('Reference 2')
    ax3.axis('off')
    
    # Anomaly map
    ax4 = plt.subplot(3, 4, 4)
    im = ax4.imshow(anomaly_map_np, cmap='hot')
    ax4.set_title('Anomaly Map')
    ax4.axis('off')
    plt.colorbar(im, ax=ax4, fraction=0.046)
    
    # Input differences
    diff_vmax = 0.3
    ax5 = plt.subplot(3, 4, 5)
    ax5.imshow(input_diffs_np[0], cmap='RdBu_r', vmin=-diff_vmax, vmax=diff_vmax)
    ax5.set_title('Input: T-R1')
    ax5.axis('off')
    
    ax6 = plt.subplot(3, 4, 6)
    ax6.imshow(input_diffs_np[1], cmap='RdBu_r', vmin=-diff_vmax, vmax=diff_vmax)
    ax6.set_title('Input: T-R2')
    ax6.axis('off')
    
    ax7 = plt.subplot(3, 4, 7)
    ax7.imshow(input_diffs_np[2], cmap='RdBu_r', vmin=-diff_vmax, vmax=diff_vmax)
    ax7.set_title('Input: R1-R2')
    ax7.axis('off')
    
    # Reconstructed differences
    ax9 = plt.subplot(3, 4, 9)
    ax9.imshow(reconstructed_diffs_np[0], cmap='RdBu_r', vmin=-diff_vmax, vmax=diff_vmax)
    ax9.set_title('Recon: T-R1')
    ax9.axis('off')
    
    ax10 = plt.subplot(3, 4, 10)
    ax10.imshow(reconstructed_diffs_np[1], cmap='RdBu_r', vmin=-diff_vmax, vmax=diff_vmax)
    ax10.set_title('Recon: T-R2')
    ax10.axis('off')
    
    ax11 = plt.subplot(3, 4, 11)
    ax11.imshow(reconstructed_diffs_np[2], cmap='RdBu_r', vmin=-diff_vmax, vmax=diff_vmax)
    ax11.set_title('Recon: R1-R2')
    ax11.axis('off')
    
    # Reconstruction error
    ax12 = plt.subplot(3, 4, 12)
    recon_error = np.mean(np.abs(input_diffs_np - reconstructed_diffs_np), axis=0)
    ax12.imshow(recon_error, cmap='hot')
    ax12.set_title('Recon Error')
    ax12.axis('off')
    
    # Draw patch rectangles if coordinates exist
    if has_patch and center_x is not None and center_y is not None:
        patch_size = 50
        half_size = patch_size // 2
        
        # Draw rectangles on relevant subplots
        axes_to_mark = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax9, ax10, ax11, ax12]
        for ax in axes_to_mark:
            rect = Rectangle((center_x - half_size, center_y - half_size), 
                           patch_size, patch_size, 
                           linewidth=2, edgecolor='cyan', facecolor='none')
            ax.add_patch(rect)
    
    plt.tight_layout()
    save_path = os.path.join(full_dir, f'{basename}_result.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Process patch if coordinates exist
    if has_patch and center_x is not None and center_y is not None:
        # Calculate boundaries with bounds checking
        img_h, img_w = target_np.shape
        patch_size = 50
        half_size = patch_size // 2
        
        # Ensure center coordinates are within valid bounds
        center_x = min(max(half_size, center_x), img_w - half_size)
        center_y = min(max(half_size, center_y), img_h - half_size)
        
        y_start = center_y - half_size
        y_end = center_y + half_size
        x_start = center_x - half_size
        x_end = center_x + half_size
        
        # Extract patches
        target_patch = target_np[y_start:y_end, x_start:x_end]
        ref1_patch = ref1_np[y_start:y_end, x_start:x_end]
        ref2_patch = ref2_np[y_start:y_end, x_start:x_end]
        anomaly_patch = anomaly_map_np[y_start:y_end, x_start:x_end]
        
        # Extract difference patches
        input_diff1_patch = input_diffs_np[0, y_start:y_end, x_start:x_end]
        input_diff2_patch = input_diffs_np[1, y_start:y_end, x_start:x_end]
        input_ref_diff_patch = input_diffs_np[2, y_start:y_end, x_start:x_end]
        
        recon_diff1_patch = reconstructed_diffs_np[0, y_start:y_end, x_start:x_end]
        recon_diff2_patch = reconstructed_diffs_np[1, y_start:y_end, x_start:x_end]
        recon_ref_diff_patch = reconstructed_diffs_np[2, y_start:y_end, x_start:x_end]
        
        # Create patch visualization
        fig_patch, axes_patch = plt.subplots(2, 4, figsize=(12, 6))
        
        # Top row: Original images and anomaly map
        axes_patch[0, 0].imshow(target_patch, cmap='gray', vmin=0, vmax=1)
        axes_patch[0, 0].set_title(f'Target patch ({center_x},{center_y})', fontsize=9)
        axes_patch[0, 0].axis('off')
        
        axes_patch[0, 1].imshow(ref1_patch, cmap='gray', vmin=0, vmax=1)
        axes_patch[0, 1].set_title('Ref1 patch', fontsize=9)
        axes_patch[0, 1].axis('off')
        
        axes_patch[0, 2].imshow(ref2_patch, cmap='gray', vmin=0, vmax=1)
        axes_patch[0, 2].set_title('Ref2 patch', fontsize=9)
        axes_patch[0, 2].axis('off')
        
        axes_patch[0, 3].imshow(anomaly_patch, cmap='hot')
        axes_patch[0, 3].set_title('Anomaly patch', fontsize=9)
        axes_patch[0, 3].axis('off')
        
        # Bottom row: Reconstructed differences
        axes_patch[1, 0].imshow(recon_diff1_patch, cmap='RdBu_r', vmin=-diff_vmax, vmax=diff_vmax)
        axes_patch[1, 0].set_title('Recon T-R1 patch', fontsize=9)
        axes_patch[1, 0].axis('off')
        
        axes_patch[1, 1].imshow(recon_diff2_patch, cmap='RdBu_r', vmin=-diff_vmax, vmax=diff_vmax)
        axes_patch[1, 1].set_title('Recon T-R2 patch', fontsize=9)
        axes_patch[1, 1].axis('off')
        
        axes_patch[1, 2].imshow(recon_ref_diff_patch, cmap='RdBu_r', vmin=-diff_vmax, vmax=diff_vmax)
        axes_patch[1, 2].set_title('Recon R1-R2 patch', fontsize=9)
        axes_patch[1, 2].axis('off')
        
        # Reconstruction error patch
        recon_error_patch = recon_error[y_start:y_end, x_start:x_end]
        axes_patch[1, 3].imshow(recon_error_patch, cmap='hot')
        axes_patch[1, 3].set_title('Error patch', fontsize=9)
        axes_patch[1, 3].axis('off')
        
        plt.tight_layout()
        save_path_patch = os.path.join(patch_dir, f'{basename}_patch.png')
        plt.savefig(save_path_patch, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()


def main():
    """Main training function for difference denoiser"""
    
    # Get device and setup
    device = get_device()
    optimal_workers = min(8, os.cpu_count() - 1)
    print(f"Using device: {device}")
    print(f"Using {optimal_workers} workers")
    
    # Configuration
    config = {
        'device': device,
        'batch_size': 2,  # Small batch size for limited training data
        'num_epochs': 5,  # Quick test with 5 epochs
        'lr': 1e-3,
        'image_size': (176, 976),
        'num_workers': optimal_workers,
        'architecture': 'difference_denoiser',
        'latent_dim': 128,
        'recon_weight': 1.0,
        'anomaly_weight': 0.5,
        'smooth_weight': 0.01,
        'loss_config': {
            'difference_denoiser': {
                'class': DifferenceDenoiserTripletLoss,
                'weight': 1.0,
                'params': {
                    'recon_weight': 1.0,
                    'anomaly_weight': 0.5,
                    'smooth_weight': 0.01
                }
            }
        }
    }
    
    # Create output directory
    project_root = Path(__file__).parent.parent
    base_output_dir = project_root / 'out'
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create session directory
    session_timestamp = datetime.now().strftime("%Y_%m%d_%H%M")
    session_dir = base_output_dir / session_timestamp
    os.makedirs(session_dir, exist_ok=True)
    
    # Create experiment name
    experiment_name = 'difference_denoiser'
    
    # Create directories
    dirs = create_experiment_directories(str(session_dir), experiment_name)
    
    # Update config with checkpoint directory
    config['checkpoint_dir'] = dirs['checkpoints']
    
    # Save configuration
    save_experiment_config(config, os.path.join(dirs['experiment'], 'training_config.json'))
    
    print(f"\nStarting Difference Denoiser training")
    print(f"Results will be saved in: {dirs['experiment']}")
    print(f"{'='*60}\n")
    
    # Create model
    model = DifferenceDenoiser(latent_dim=config['latent_dim'])
    model = model.to(device)
    
    print(f"Model architecture: DifferenceDenoiser")
    print(f"Latent dimension: {config['latent_dim']}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}\n")
    
    # Create dataset
    train_dataset = OpticalDatasetTriplet(
        '/home/yclai/vscode_project/Anomaly_Detection/triplet_dataset',
        mode='train',
        transform=None
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    
    # Save training samples
    train_samples_dir = os.path.join(dirs['experiment'], 'train_samples')
    os.makedirs(train_samples_dir, exist_ok=True)
    save_difference_training_samples(train_loader, train_samples_dir, num_samples=5)
    
    # Train model
    model, train_history = train_difference_denoiser(model, train_loader, config)
    
    # Save model
    model_path = os.path.join(dirs['weights'], 'final_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save training history
    save_training_history_csv(train_history, os.path.join(dirs['history'], 'training_history.csv'))
    save_training_summary(train_history, config, experiment_name, 
                         os.path.join(dirs['experiment'], 'training_summary.txt'))
    
    # Plot loss curves
    plot_loss_curves(train_history, dirs['history'], experiment_name)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_dataset = OpticalDatasetTriplet(
        '/home/yclai/vscode_project/Anomaly_Detection/triplet_dataset',
        mode='test',
        transform=None
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    # Evaluate and visualize
    vis_dir = os.path.join(dirs['experiment'], 'visualizations')
    anomaly_scores = evaluate_difference_denoiser(model, test_loader, device, vis_dir)
    
    # Save evaluation results
    eval_results = {
        'anomaly_scores': anomaly_scores.tolist(),
        'mean_score': float(np.mean(anomaly_scores)),
        'std_score': float(np.std(anomaly_scores)),
        'max_score': float(np.max(anomaly_scores)),
        'min_score': float(np.min(anomaly_scores))
    }
    
    import json
    with open(os.path.join(dirs['evaluation'], 'evaluation_results.json'), 'w') as f:
        json.dump(eval_results, f, indent=4)
    
    print(f"\nTraining completed!")
    print(f"Results saved in: {dirs['experiment']}")
    print(f"\nEvaluation summary:")
    print(f"  Mean anomaly score: {eval_results['mean_score']:.6f}")
    print(f"  Std anomaly score: {eval_results['std_score']:.6f}")
    print(f"  Score range: [{eval_results['min_score']:.6f}, {eval_results['max_score']:.6f}]")


if __name__ == "__main__":
    main()
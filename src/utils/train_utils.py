"""
Training Utilities
==================

Functions for model training and evaluation.
"""

import torch
from tqdm import tqdm
import os

# Handle both relative and absolute imports
try:
    from ..losses import ModularLossManager
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from losses import ModularLossManager


def train_model(model, train_loader, config):
    """Train anomaly detection model with modular loss support"""
    device = config['device']
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['num_epochs'])
    
    # Initialize loss function with ModularLossManager
    criterion = ModularLossManager(config['loss_config'], normalize_weights=True)
    criterion.to(device)
    
    # Track training history
    train_history = {
        'total_loss': [],
        'component_losses': {name: [] for name in criterion.losses.keys()},
        'weights': []
    }
    
    for epoch in range(config['num_epochs']):
        # Training
        model.train()
        train_losses = {key: 0 for key in criterion.losses.keys()}
        train_losses['total'] = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}'):
            if config.get('use_synthetic_anomalies', False):
                # Dataset returns clean_images, anomaly_images, anomaly_masks
                clean_images, anomaly_images, anomaly_masks = batch
                clean_images = clean_images.to(device)
                anomaly_images = anomaly_images.to(device)
                anomaly_masks = anomaly_masks.to(device)
                
                # Use clean images as target and anomaly images as input
                target = clean_images
                input_images = anomaly_images
            else:
                # Normal training without synthetic anomalies
                images, _ = batch
                images = images.to(device)
                target = images
                input_images = images
            
            # Forward pass
            output = model(input_images)
            
            # Handle VAE case which returns (recon, mu, logvar)
            if isinstance(output, tuple) and len(output) == 3:
                recon, mu, logvar = output
                # VAE needs special loss handling
                if 'vae' in config.get('loss_config', {}):
                    # Use VAE loss
                    vae_loss = criterion.losses['vae']
                    total_loss, recon_loss, kl_loss = vae_loss(recon, target, mu, logvar)
                    loss_dict = {
                        'total': total_loss,
                        'recon': recon_loss,
                        'kl': kl_loss
                    }
                else:
                    # Use standard losses on reconstruction only
                    loss_dict = criterion(recon, target)
            else:
                # Standard autoencoder
                recon = output
                loss_dict = criterion(recon, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss_dict['total'].backward()
            optimizer.step()
            
            # Update losses
            for key, value in loss_dict.items():
                if key in train_losses:
                    train_losses[key] += value.item()
        
        # Average losses
        for key in train_losses:
            train_losses[key] /= len(train_loader)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Store history
        train_history['total_loss'].append(train_losses['total'])
        for name in criterion.losses.keys():
            if name in train_losses:
                train_history['component_losses'][name].append(train_losses[name])
        train_history['weights'].append(criterion.get_weights())
        
        # Print progress
        print(f"Epoch {epoch+1}: Train Loss: {train_losses['total']:.4f}")
        loss_components = [f"{name}: {train_losses.get(name, 0):.4f}" for name in criterion.losses.keys()]
        print(f"  Components - {', '.join(loss_components)}")
        
        # Print current weights
        weights = criterion.get_weights()
        weight_str = ', '.join([f"{name}: {weight:.3f}" for name, weight in weights.items()])
        print(f"  Weights - {weight_str}")
        
        # Save checkpoint every 10 epochs (if checkpoint_dir is provided)
        if (epoch + 1) % 10 == 0 and 'checkpoint_dir' in config:
            checkpoint_dir = config['checkpoint_dir']
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_losses['total'],
                'component_losses': train_losses,
                'weights': criterion.get_weights()
            }, checkpoint_path)
            print(f"  Checkpoint saved to {checkpoint_path}")
    
    return model, train_history


def evaluate_model(model, test_loader, loss_manager=None, device='cuda'):
    """Evaluate model on test data
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        loss_manager: Optional loss manager for custom loss calculation
        device: Device to run evaluation on
    
    Returns:
        tuple: (scores, labels) where scores are anomaly scores and labels are ground truth
    """
    model.to(device)
    model.eval()
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            images, labels = batch
            images = images.to(device)
            
            # Forward pass
            output = model(images)
            
            # Handle VAE case
            if isinstance(output, tuple) and len(output) == 3:
                recon, mu, logvar = output
            else:
                recon = output
            
            # Calculate reconstruction error as anomaly score
            if loss_manager is not None:
                # For evaluation, we just use MSE as anomaly score
                # since loss_manager returns aggregated loss
                mse = torch.mean((recon - images) ** 2, dim=[1, 2, 3])
                batch_scores = mse.cpu().numpy()
            else:
                # Default to MSE
                mse = torch.mean((recon - images) ** 2, dim=[1, 2, 3])
                batch_scores = mse.cpu().numpy()
            
            all_scores.extend(batch_scores)
            all_labels.extend(labels.numpy())
    
    return all_scores, all_labels
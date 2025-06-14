"""
Training Functions
==================

Training and evaluation utilities.
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
            recon = model(input_images)
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
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            os.makedirs(config['save_path'], exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_losses['total'],
            }, f"{config['save_path']}/checkpoint_epoch_{epoch+1}.pth")
    
    return model, train_history


def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model on test data"""
    model.to(device)
    model.eval()
    
    total_loss = 0
    reconstruction_errors = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            images, labels = batch
            images = images.to(device)
            
            # Forward pass
            recon = model(images)
            
            # Calculate reconstruction error
            mse = torch.mean((recon - images) ** 2, dim=[1, 2, 3])
            reconstruction_errors.extend(mse.cpu().numpy())
            total_loss += mse.mean().item()
    
    avg_loss = total_loss / len(test_loader)
    
    return {
        'avg_loss': avg_loss,
        'reconstruction_errors': reconstruction_errors
    }
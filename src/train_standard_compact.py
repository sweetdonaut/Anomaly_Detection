"""
Training script for StandardCompactAutoencoder
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.standard_compact import StandardCompactAutoencoder
from data.datasets import OpticalDataset
from losses.mse import MSELoss
from utils.training import train_model
from utils.monitoring import setup_monitoring
from losses.manager import LossManager

def main():
    # Configuration
    config = {
        'architecture': 'standard_compact',
        'batch_size': 16,
        'learning_rate': 1e-3,
        'num_epochs': 100,
        'image_size': (176, 976),
        'data_path': '/home/yclai/vscode_project/Anomaly_Detection/OpticalDataset',
        'output_dir': '/home/yclai/vscode_project/Anomaly_Detection/Out',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'loss_config': {
            'mse': {
                'class': MSELoss,
                'weight': 1.0
            }
        }
    }
    
    # Create model
    model = StandardCompactAutoencoder(input_size=config['image_size'])
    model = model.to(config['device'])
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create dataset and dataloader
    dataset = OpticalDataset(
        root_dir=config['data_path'],
        split='train',
        image_size=config['image_size']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Setup loss
    loss_manager = LossManager(config['loss_config'])
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Setup scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Setup monitoring
    writer, log_dir = setup_monitoring(config)
    
    # Train model
    trained_model = train_model(
        model=model,
        train_loader=dataloader,
        val_loader=None,
        loss_manager=loss_manager,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config['num_epochs'],
        device=config['device'],
        writer=writer,
        save_dir=config['output_dir'],
        model_name='standard_compact_mse'
    )
    
    # Save final model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(config['output_dir'], f'standard_compact_mse_{timestamp}.pth')
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': config,
        'architecture': 'standard_compact'
    }, save_path)
    
    print(f"\nTraining completed! Model saved to: {save_path}")
    
    if writer:
        writer.close()

if __name__ == "__main__":
    main()
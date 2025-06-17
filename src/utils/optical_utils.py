"""
Optical Dataset Utilities
=========================

Utilities for working with OpticalDataset in production.
"""

import torch
from torch.utils.data import DataLoader
from typing import Optional, Tuple


def create_optical_dataloader(dataset, batch_size: int, shuffle: bool = True, 
                            num_workers: int = 0, **kwargs) -> DataLoader:
    """
    Create a DataLoader for OpticalDataset with proper collate function.
    
    Args:
        dataset: OpticalDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of workers
        **kwargs: Additional DataLoader arguments
    
    Returns:
        DataLoader that handles OpticalDataset output properly
    """
    
    def optical_collate_fn(batch):
        """Custom collate function that adds dummy labels when needed"""
        if isinstance(batch[0], tuple):
            # Training mode with synthetic anomalies
            clean_images = []
            anomaly_images = []
            anomaly_masks = []
            
            for item in batch:
                clean, anomaly, mask = item
                clean_images.append(clean)
                anomaly_images.append(anomaly)
                anomaly_masks.append(mask)
            
            return (torch.stack(clean_images), 
                    torch.stack(anomaly_images), 
                    torch.stack(anomaly_masks))
        else:
            # Normal mode - just images
            images = torch.stack(batch)
            # Add dummy labels (all zeros) for compatibility
            labels = torch.zeros(len(batch), dtype=torch.long)
            return images, labels
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=optical_collate_fn,
        **kwargs
    )


def evaluate_optical_model(model, test_loader, loss_manager, device):
    """
    Evaluate model on OpticalDataset test data.
    
    This is a wrapper around evaluate_model that handles OpticalDataset's
    single-output format.
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        loss_manager: Loss manager for computing reconstruction error
        device: Device to run on
    
    Returns:
        scores: Anomaly scores
        labels: Labels (all zeros for OpticalDataset)
    """
    from .train_utils import evaluate_model
    
    # The custom collate function in create_optical_dataloader adds dummy labels
    return evaluate_model(model, test_loader, loss_manager, device)
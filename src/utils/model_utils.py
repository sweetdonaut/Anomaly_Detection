"""
Model Utilities
===============

Functions for model-related operations and analysis.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict


def get_device():
    """Get the best available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device


class LatentSpaceAnalyzer:
    """Analyze latent space representations for anomaly detection"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.normal_features = None
    
    def extract_features(self, dataloader):
        """Extract latent features from dataloader"""
        self.model.eval()
        features = []
        
        with torch.no_grad():
            for batch, _ in dataloader:
                batch = batch.to(self.device)
                
                # Get features based on model type
                if hasattr(self.model, 'get_latent'):
                    # Standard method - use latent representation
                    feat = self.model.get_latent(batch)
                    # Flatten if needed
                    if len(feat.shape) > 2:
                        feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
                elif hasattr(self.model, 'get_multi_level_features'):
                    # U-Net models with multi-level features
                    feat = self.model.get_multi_level_features(batch)
                else:
                    raise ValueError("Model doesn't have feature extraction method")
                
                features.append(feat.cpu())
        
        return torch.cat(features, dim=0)
    
    def fit_normal_features(self, normal_dataloader):
        """Fit the analyzer with normal data features"""
        self.normal_features = self.extract_features(normal_dataloader)
        self.mean_normal = self.normal_features.mean(dim=0)
        self.std_normal = self.normal_features.std(dim=0) + 1e-6
    
    def compute_anomaly_score(self, test_batch):
        """Compute anomaly scores for test batch"""
        self.model.eval()
        
        with torch.no_grad():
            test_batch = test_batch.to(self.device)
            
            # Get features
            if hasattr(self.model, 'get_latent'):
                # Standard method - use latent representation
                test_features = self.model.get_latent(test_batch)
                # Flatten if needed
                if len(test_features.shape) > 2:
                    test_features = F.adaptive_avg_pool2d(test_features, 1).squeeze(-1).squeeze(-1)
            elif hasattr(self.model, 'get_multi_level_features'):
                # U-Net models with multi-level features
                test_features = self.model.get_multi_level_features(test_batch)
            else:
                raise ValueError("Model doesn't have feature extraction method")
            
            # Normalize features
            test_features = (test_features - self.mean_normal.to(self.device)) / self.std_normal.to(self.device)
            
            # Compute L2 distance to normal distribution
            scores = torch.norm(test_features, p=2, dim=1)
            
        return scores.cpu().numpy()
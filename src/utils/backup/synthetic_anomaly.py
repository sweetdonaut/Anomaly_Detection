"""
Synthetic Anomaly Generator
==========================

Generate synthetic anomalies for training.
"""

import torch
import random


class SyntheticAnomalyGenerator:
    """Generate synthetic anomalies for training - bright/dark spots only"""
    def __init__(self, anomaly_prob=0.3):
        self.anomaly_prob = anomaly_prob
    
    def generate_anomaly(self, image):
        """Generate synthetic bright or dark spot anomaly"""
        # Handle both single image and batch
        if image.dim() == 3:  # Single image (C, H, W)
            image = image.unsqueeze(0)  # Add batch dimension
            single_image = True
        else:
            single_image = False
            
        if random.random() > self.anomaly_prob:
            if single_image:
                return image.squeeze(0), torch.zeros_like(image.squeeze(0))
            return image, torch.zeros_like(image)
        
        # Clone image to avoid modifying original
        anomaly_image = image.clone()
        mask = torch.zeros_like(image)
        
        # Generate bright or dark spot
        anomaly_image, mask = self._generate_spot_anomaly(anomaly_image)
        
        # Remove batch dimension if it was a single image
        if single_image:
            anomaly_image = anomaly_image.squeeze(0)
            mask = mask.squeeze(0)
            
        return anomaly_image, mask
    
    def _generate_spot_anomaly(self, image):
        """Generate circular/elliptical bright or dark spots"""
        B, C, H, W = image.shape
        mask = torch.zeros_like(image)
        
        for b in range(B):
            # Spot size around 10x10 pixels with some variation
            base_size = 10
            size_variation = random.uniform(0.7, 1.3)  # 70% to 130% of base size
            spot_h = int(base_size * size_variation * random.uniform(0.8, 1.2))  # Elliptical variation
            spot_w = int(base_size * size_variation * random.uniform(0.8, 1.2))
            
            # Ensure minimum size
            spot_h = max(6, min(spot_h, 15))
            spot_w = max(6, min(spot_w, 15))
            
            # Random position (ensure spot fits within image)
            y = random.randint(spot_h//2, H - spot_h//2 - 1)
            x = random.randint(spot_w//2, W - spot_w//2 - 1)
            
            # Create elliptical mask
            y_grid, x_grid = torch.meshgrid(
                torch.arange(spot_h, dtype=torch.float32) - spot_h/2,
                torch.arange(spot_w, dtype=torch.float32) - spot_w/2,
                indexing='ij'
            )
            
            # Elliptical distance
            ellipse_mask = ((x_grid / (spot_w/2))**2 + (y_grid / (spot_h/2))**2) <= 1
            ellipse_mask = ellipse_mask.float()
            
            # Smooth edges with Gaussian-like falloff
            distance = torch.sqrt((x_grid / (spot_w/2))**2 + (y_grid / (spot_h/2))**2)
            smooth_mask = torch.exp(-2 * torch.clamp(distance - 0.8, min=0))
            smooth_mask = smooth_mask * ellipse_mask
            smooth_mask = smooth_mask / smooth_mask.max() if smooth_mask.max() > 0 else smooth_mask
            
            # Decide if bright or dark spot
            is_bright = random.random() > 0.5
            
            # Calculate spot intensity (adjusted for normalized images)
            if is_bright:
                # Bright spot: increase pixel values
                intensity = random.uniform(0.2, 0.4)  # Reduced intensity for normalized images
            else:
                # Dark spot: decrease pixel values
                intensity = random.uniform(-0.4, -0.2)  # Reduced intensity for normalized images
            
            # Apply spot to image
            y_start = y - spot_h//2
            x_start = x - spot_w//2
            y_end = y_start + spot_h
            x_end = x_start + spot_w
            
            # Apply smooth intensity change
            for c in range(C):
                region = image[b, c, y_start:y_end, x_start:x_end]
                image[b, c, y_start:y_end, x_start:x_end] = torch.clamp(
                    region + intensity * smooth_mask.to(image.device),
                    -1, 1  # Assuming normalized images
                )
            
            # Binary mask for evaluation
            mask[b, :, y_start:y_end, x_start:x_end] = (ellipse_mask > 0).float()
        
        return image, mask
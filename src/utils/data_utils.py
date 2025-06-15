"""
Data Utilities
==============

Functions for data processing and augmentation.
"""

import torch
import numpy as np
import cv2


class SyntheticAnomalyGenerator:
    """Generate synthetic anomalies for training"""
    
    def __init__(self, anomaly_prob=0.3):
        """
        Args:
            anomaly_prob: Probability of generating anomaly for each image
        """
        self.anomaly_prob = anomaly_prob
    
    def generate_anomaly(self, image):
        """
        Generate synthetic anomaly on image
        
        Args:
            image: Tensor of shape (C, H, W) or (B, C, H, W)
        
        Returns:
            anomaly_image: Image with synthetic anomaly
            anomaly_mask: Binary mask indicating anomaly location
        """
        if len(image.shape) == 3:
            # Single image
            return self._generate_single_anomaly(image)
        else:
            # Batch of images
            anomaly_images = []
            anomaly_masks = []
            
            for i in range(image.shape[0]):
                if np.random.random() < self.anomaly_prob:
                    anomaly_img, mask = self._generate_single_anomaly(image[i])
                else:
                    anomaly_img = image[i].clone()
                    mask = torch.zeros(image[i].shape[1:])
                
                anomaly_images.append(anomaly_img)
                anomaly_masks.append(mask)
            
            return torch.stack(anomaly_images), torch.stack(anomaly_masks)
    
    def _generate_single_anomaly(self, image):
        """Generate anomaly for a single image"""
        C, H, W = image.shape
        anomaly_image = image.clone()
        
        # Create anomaly mask
        mask = torch.zeros((H, W))
        
        # Random anomaly size (as percentage of image size) - adjusted for narrow images
        # For 176x976 images, we need smaller anomalies relative to width
        size_factor = np.random.uniform(0.15, 0.25)  # 15-25% of min dimension
        base_size = int(min(H, W) * size_factor)
        
        # For extremely elongated images, adjust aspect ratio accordingly
        image_aspect_ratio = H / W
        if image_aspect_ratio > 3:  # Very tall image
            # Make anomalies more circular to avoid spanning entire width
            aspect_ratio = np.random.uniform(0.8, 1.2)
        else:
            # Normal aspect ratio adjustment
            aspect_ratio = np.random.uniform(0.7, 1.4)
            
        anomaly_h = int(base_size * aspect_ratio)
        anomaly_w = int(base_size / aspect_ratio)
        
        # Ensure minimum size but cap maximum width for narrow images
        anomaly_h = max(15, anomaly_h)
        anomaly_w = max(15, min(anomaly_w, int(W * 0.4)))  # Max 40% of width
        
        # Random position
        pos_h = np.random.randint(anomaly_h//2, H - anomaly_h//2)
        pos_w = np.random.randint(anomaly_w//2, W - anomaly_w//2)
        
        # Create elliptical mask
        y, x = np.ogrid[-pos_h:H-pos_h, -pos_w:W-pos_w]
        mask_np = ((x*1.0/anomaly_w)**2 + (y*1.0/anomaly_h)**2 <= 1).astype(float)
        
        # Apply Gaussian blur for smooth edges
        mask_np = cv2.GaussianBlur(mask_np, (21, 21), 5)
        mask = torch.from_numpy(mask_np).float()
        
        # Generate anomaly pattern with adaptive contrast
        # Calculate the local statistics in the anomaly region
        mask_region = mask > 0.5  # Use center of anomaly for statistics
        if mask_region.any():
            # Get the min and max values in the anomaly region
            region_min = image[:, mask_region].min().item()
            region_max = image[:, mask_region].max().item()
            region_mean = image[:, mask_region].mean().item()
            
            # Adaptive anomaly value based on local brightness
            # Ensure minimum contrast for visibility
            MIN_CONTRAST = 0.2  # Minimum contrast difference for visibility
            
            if np.random.random() < 0.5:
                # Bright spot - ensure we don't exceed 0.95
                max_safe_value = 0.95
                # Calculate how much we can safely increase
                max_increase = max_safe_value - region_max
                # Desired increase based on local mean
                desired_increase = np.random.uniform(0.3, 0.5)
                # Apply increase while ensuring minimum contrast and safety limits
                actual_increase = max(MIN_CONTRAST, min(desired_increase, max_increase))
                anomaly_value = torch.ones_like(image) * actual_increase
            else:
                # Dark spot - ensure we don't go below 0.05
                min_safe_value = 0.05
                # Calculate how much we can safely decrease
                max_decrease = region_min - min_safe_value
                # Desired decrease based on local mean
                desired_decrease = np.random.uniform(0.3, 0.5)
                # Apply decrease while ensuring minimum contrast and safety limits
                actual_decrease = max(MIN_CONTRAST, min(desired_decrease, max_decrease))
                anomaly_value = torch.ones_like(image) * -actual_decrease
        else:
            # Fallback to conservative values if mask region is empty
            if np.random.random() < 0.5:
                anomaly_value = torch.ones_like(image) * 0.3
            else:
                anomaly_value = torch.ones_like(image) * -0.3
        
        # Apply anomaly
        for c in range(C):
            anomaly_image[c] = image[c] + mask * anomaly_value[c]
        
        # Clip values to valid range
        anomaly_image = torch.clamp(anomaly_image, -1, 1)
        
        # Binary mask
        mask = (mask > 0.1).float()
        
        return anomaly_image, mask
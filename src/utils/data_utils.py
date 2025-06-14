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
        
        # Random anomaly size (as percentage of image size)
        size_factor = np.random.uniform(0.02, 0.08)  # 2-8% of image
        anomaly_h = int(H * size_factor)
        anomaly_w = int(W * size_factor)
        
        # Make it elliptical
        aspect_ratio = np.random.uniform(0.7, 1.3)
        anomaly_h = int(anomaly_h * aspect_ratio)
        anomaly_w = int(anomaly_w / aspect_ratio)
        
        # Ensure minimum size
        anomaly_h = max(10, anomaly_h)
        anomaly_w = max(10, anomaly_w)
        
        # Random position
        pos_h = np.random.randint(anomaly_h//2, H - anomaly_h//2)
        pos_w = np.random.randint(anomaly_w//2, W - anomaly_w//2)
        
        # Create elliptical mask
        y, x = np.ogrid[-pos_h:H-pos_h, -pos_w:W-pos_w]
        mask_np = ((x*1.0/anomaly_w)**2 + (y*1.0/anomaly_h)**2 <= 1).astype(float)
        
        # Apply Gaussian blur for smooth edges
        mask_np = cv2.GaussianBlur(mask_np, (21, 21), 5)
        mask = torch.from_numpy(mask_np).float()
        
        # Generate anomaly pattern (bright or dark spot)
        if np.random.random() < 0.5:
            # Bright spot
            anomaly_value = torch.ones_like(image) * np.random.uniform(0.2, 0.4)
        else:
            # Dark spot
            anomaly_value = torch.ones_like(image) * np.random.uniform(-0.4, -0.2)
        
        # Apply anomaly
        for c in range(C):
            anomaly_image[c] = image[c] + mask * anomaly_value[c]
        
        # Clip values to valid range
        anomaly_image = torch.clamp(anomaly_image, -1, 1)
        
        # Binary mask
        mask = (mask > 0.1).float()
        
        return anomaly_image, mask
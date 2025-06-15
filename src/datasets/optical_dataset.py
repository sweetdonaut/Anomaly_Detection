"""
Optical Dataset
===============

Dataset class for loading optical inspection images in TIFF format.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import tifffile
import numpy as np
from typing import Optional, Callable, Tuple, List


class OpticalDataset(Dataset):
    """
    Dataset for optical inspection images
    
    Images are stored as float32 TIFF files with shape (976, 176)
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        defect_types: Optional[List[str]] = None,
        use_augmentation: bool = False,
        synthetic_anomaly_generator: Optional[object] = None
    ):
        """
        Args:
            root_dir: Path to OpticalDataset directory
            split: 'train' or 'test'
            transform: Optional transform to apply to images
            target_transform: Optional transform to apply to labels
            defect_types: List of defect types to include (None = all)
            use_augmentation: Whether to use data augmentation
            synthetic_anomaly_generator: Generator for synthetic anomalies
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.use_augmentation = use_augmentation
        self.synthetic_anomaly_generator = synthetic_anomaly_generator
        
        # Load image paths and labels
        self.image_paths = []
        self.labels = []
        self.defect_names = []
        
        split_dir = self.root_dir / split
        
        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} does not exist")
        
        # Get all subdirectories (defect types)
        subdirs = [d for d in split_dir.iterdir() if d.is_dir()]
        
        for subdir in subdirs:
            defect_name = subdir.name
            
            # Filter by defect types if specified
            if defect_types is not None and defect_name not in defect_types:
                continue
            
            # Get all TIFF files in this directory
            tiff_files = list(subdir.glob('*.tiff'))
            
            for tiff_file in tiff_files:
                self.image_paths.append(str(tiff_file))
                # Label: 0 for good, 1 for defective
                self.labels.append(0 if defect_name == 'good' else 1)
                self.defect_names.append(defect_name)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {split_dir}")
        
        print(f"Loaded {len(self.image_paths)} images from {split} split")
        
        # Print distribution
        unique_defects = set(self.defect_names)
        for defect in sorted(unique_defects):
            count = self.defect_names.count(defect)
            print(f"  {defect}: {count} images")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int):
        """
        Returns:
            For training with synthetic anomalies: (clean_image, anomaly_image, anomaly_mask) as tensors
            For normal use: image as tensor
        """
        # Load TIFF image
        image = tifffile.imread(self.image_paths[idx])
        
        # Ensure image is float32
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # Add channel dimension if needed
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)  # (H, W) -> (1, H, W)
        
        # Convert to tensor
        image = torch.from_numpy(image)
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        # Apply synthetic anomalies if in training mode
        if self.split == 'train' and self.synthetic_anomaly_generator is not None:
            # Keep original clean image
            clean_image = image.clone()
            
            # Generate anomaly
            anomaly_image, anomaly_mask = self.synthetic_anomaly_generator.generate_anomaly(image)
            
            # Return clean, anomaly, and mask for training
            return clean_image, anomaly_image, anomaly_mask
        
        # Return just the image for normal use
        return image
    
    def get_image_info(self, idx: int) -> dict:
        """Get information about a specific image"""
        return {
            'path': self.image_paths[idx],
            'label': self.labels[idx],
            'defect_type': self.defect_names[idx],
            'is_anomaly': self.labels[idx] == 1
        }


class OpticalDatasetWithMask(OpticalDataset):
    """Extended version that also loads ground truth masks if available"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Try to find corresponding mask files
        self.mask_paths = []
        for img_path in self.image_paths:
            # Construct mask path (assuming masks are in a parallel structure)
            img_path = Path(img_path)
            mask_dir = img_path.parent.parent.parent / 'ground_truth' / self.split / img_path.parent.name
            mask_path = mask_dir / img_path.name
            
            if mask_path.exists():
                self.mask_paths.append(str(mask_path))
            else:
                self.mask_paths.append(None)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
        """
        Returns:
            image: Tensor of shape (C, H, W)
            label: 0 for good, 1 for defective  
            mask: Ground truth mask (if available) or None
        """
        image, label = super().__getitem__(idx)
        
        # Load mask if available
        mask = None
        if self.mask_paths[idx] is not None:
            mask = tifffile.imread(self.mask_paths[idx])
            mask = torch.from_numpy(mask).float()
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
        
        return image, label, mask
"""
Test MVTec Dataset Loading
==========================

Quick script to test dataset loading and visualize images.
"""

import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from datasets import MVTecDataset

def test_dataset_loading():
    """Test dataset loading with different configurations"""
    
    # Test 1: Without any transform
    print("Test 1: Loading without transform")
    dataset_no_transform = MVTecDataset(
        '/home/yclai/vscode_project/Anomaly_Detection/MVTec_AD_dataset',
        'grid',
        'train',
        transform=None,
        use_augmentation=False
    )
    
    try:
        img1, label1 = dataset_no_transform[0]
        print(f"  - Image type: {type(img1)}")
        print(f"  - Label: {label1}")
    except Exception as e:
        print(f"  - Error: {e}")
    breakpoint()
    # Test 2: With basic transform (only ToTensor)
    print("\nTest 2: Loading with ToTensor")
    transform_basic = transforms.Compose([
        transforms.ToTensor()
    ])
    
    dataset_basic = MVTecDataset(
        '/home/yclai/vscode_project/Anomaly_Detection/MVTec_AD_dataset',
        'grid',
        'train',
        transform=transform_basic,
        use_augmentation=False
    )
    
    try:
        img2, label2 = dataset_basic[0]
        print(f"  - Image type: {type(img2)}")
        print(f"  - Image shape: {img2.shape}")
        print(f"  - Image range: [{img2.min():.3f}, {img2.max():.3f}]")
        print(f"  - Label: {label2}")
    except Exception as e:
        print(f"  - Error: {e}")
    
    # Test 3: With full transform (Resize + ToTensor)
    print("\nTest 3: Loading with Resize + ToTensor")
    transform_full = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    
    dataset_full = MVTecDataset(
        '/home/yclai/vscode_project/Anomaly_Detection/MVTec_AD_dataset',
        'grid',
        'train',
        transform=transform_full,
        use_augmentation=False
    )
    
    try:
        img3, label3 = dataset_full[0]
        print(f"  - Image type: {type(img3)}")
        print(f"  - Image shape: {img3.shape}")
        print(f"  - Image range: [{img3.min():.3f}, {img3.max():.3f}]")
        print(f"  - Label: {label3}")
        
        # Visualize the image
        plt.figure(figsize=(8, 8))
        plt.imshow(img3.squeeze().numpy(), cmap='gray')
        plt.title(f'Sample Image (Label: {label3})')
        plt.colorbar()
        plt.savefig('/home/yclai/vscode_project/Anomaly_Detection/src/test_image.png')
        print("\n  - Saved visualization to test_image.png")
        
    except Exception as e:
        print(f"  - Error: {e}")

if __name__ == "__main__":
    test_dataset_loading()
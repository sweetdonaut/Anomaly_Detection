"""
Visualize Data Pipeline
=======================

Visualize images at different stages of the data pipeline.
"""

import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from datasets import MVTecDataset

def visualize_pipeline():
    """Visualize the complete data pipeline"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('MVTec Dataset Loading Pipeline', fontsize=16)
    
    # 1. Original PIL Image
    img_path = '/home/yclai/vscode_project/Anomaly_Detection/MVTec_AD_dataset/grid/train/good/000.png'
    original_img = Image.open(img_path).convert('L')
    
    axes[0, 0].imshow(original_img, cmap='gray')
    axes[0, 0].set_title(f'1. Original PIL Image\nSize: {original_img.size}')
    axes[0, 0].axis('off')
    
    # 2. After ToTensor (no resize)
    to_tensor = transforms.ToTensor()
    tensor_no_resize = to_tensor(original_img)
    
    axes[0, 1].imshow(tensor_no_resize.squeeze().numpy(), cmap='gray')
    axes[0, 1].set_title(f'2. After ToTensor\nShape: {tensor_no_resize.shape}\nRange: [{tensor_no_resize.min():.3f}, {tensor_no_resize.max():.3f}]')
    axes[0, 1].axis('off')
    
    # 3. After Resize + ToTensor
    resize_tensor = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    tensor_resized = resize_tensor(original_img)
    
    axes[0, 2].imshow(tensor_resized.squeeze().numpy(), cmap='gray')
    axes[0, 2].set_title(f'3. After Resize + ToTensor\nShape: {tensor_resized.shape}\nRange: [{tensor_resized.min():.3f}, {tensor_resized.max():.3f}]')
    axes[0, 2].axis('off')
    
    # 4. With old Normalize
    normalize_old = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    tensor_normalized = normalize_old(original_img)
    
    # Show normalized image (clipped to [0, 1] for display)
    display_normalized = (tensor_normalized.squeeze().numpy() + 1) / 2  # Convert from [-1, 1] to [0, 1]
    axes[1, 0].imshow(display_normalized, cmap='gray')
    axes[1, 0].set_title(f'4. With Normalize(0.5, 0.5)\nRange: [{tensor_normalized.min():.3f}, {tensor_normalized.max():.3f}]')
    axes[1, 0].axis('off')
    
    # 5. Load from dataset
    dataset = MVTecDataset(
        '/home/yclai/vscode_project/Anomaly_Detection/MVTec_AD_dataset',
        'grid',
        'train',
        transform=transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ]),
        use_augmentation=False
    )
    
    img_from_dataset, label = dataset[0]
    axes[1, 1].imshow(img_from_dataset.squeeze().numpy(), cmap='gray')
    axes[1, 1].set_title(f'5. From MVTecDataset\nShape: {img_from_dataset.shape}\nLabel: {label}')
    axes[1, 1].axis('off')
    
    # 6. Histogram of pixel values
    axes[1, 2].hist(img_from_dataset.squeeze().numpy().flatten(), bins=50, alpha=0.7)
    axes[1, 2].set_title('6. Pixel Value Distribution\n(Current Pipeline)')
    axes[1, 2].set_xlabel('Pixel Value')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/yclai/vscode_project/Anomaly_Detection/src/data_pipeline_visualization.png', dpi=150)
    print("Visualization saved to data_pipeline_visualization.png")
    
    # Print summary
    print("\nData Pipeline Summary:")
    print("=" * 50)
    print(f"Original image size: {original_img.size}")
    print(f"After resize: 512 x 512")
    print(f"Data type: torch.float32")
    print(f"Value range: [0, 1] (using ToTensor only)")
    print(f"Previously: [-1, 1] (with Normalize(0.5, 0.5))")

if __name__ == "__main__":
    visualize_pipeline()
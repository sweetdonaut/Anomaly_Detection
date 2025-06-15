"""
Anomaly Visualizer
==================

Visualize anomaly detection results.
"""

import os
import matplotlib.pyplot as plt


class AnomalyVisualizer:
    """Visualize anomaly detection results without ground truth"""
    def __init__(self, save_dir='./results'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def visualize_reconstruction(self, original, reconstruction, anomaly_map, 
                               save_name=None, show=True):
        """Visualize original, reconstruction, and anomaly heatmap"""
        # Create figure with reduced width and proper aspect ratio for vertical images
        fig, axes = plt.subplots(1, 3, figsize=(6, 8))
        
        # Original image
        axes[0].imshow(original.squeeze().cpu().numpy(), cmap='gray')
        axes[0].set_title('Original', fontsize=10)
        axes[0].axis('off')
        
        # Reconstruction
        axes[1].imshow(reconstruction.squeeze().cpu().numpy(), cmap='gray')
        axes[1].set_title('Reconstruction', fontsize=10)
        axes[1].axis('off')
        
        # Anomaly heatmap
        im = axes[2].imshow(anomaly_map, cmap='hot')
        axes[2].set_title('Anomaly Map', fontsize=10)
        axes[2].axis('off')
        
        # Add colorbar with adjusted size
        cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        
        # Reduce spacing between subplots
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), 
                       dpi=150, bbox_inches='tight', pad_inches=0.1)
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def save_anomaly_scores(self, scores, image_paths, save_name='anomaly_scores.txt'):
        """Save anomaly scores to text file"""
        with open(os.path.join(self.save_dir, save_name), 'w') as f:
            f.write("Image Path\tAnomaly Score\n")
            for path, score in zip(image_paths, scores):
                f.write(f"{path}\t{score:.6f}\n")
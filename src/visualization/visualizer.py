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
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(131)
        plt.imshow(original.squeeze().cpu().numpy(), cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        # Reconstruction
        plt.subplot(132)
        plt.imshow(reconstruction.squeeze().cpu().numpy(), cmap='gray')
        plt.title('Reconstruction')
        plt.axis('off')
        
        # Anomaly heatmap
        plt.subplot(133)
        plt.imshow(anomaly_map, cmap='hot')
        plt.title('Anomaly Heatmap')
        plt.colorbar()
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=150, bbox_inches='tight')
        
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
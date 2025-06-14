# Unsupervised Anomaly Detection Project Specification

## Project Overview

This project aims to implement an unsupervised anomaly detection system based on reconstruction models for industrial defect detection. The approach focuses on training autoencoders using only normal images and detecting anomalies through reconstruction errors.

## Technical Architecture

### Core Components

**Reconstruction Model**
- Primary approach: Autoencoder-based anomaly detection
- Training exclusively on normal images to learn normal patterns
- Anomaly detection through reconstruction error analysis

**Loss Function Design**
The system implements a modular loss function framework combining multiple components:
- Mean Squared Error (MSE): Pixel-level reconstruction accuracy
- Structural Similarity Index (SSIM): Structural similarity preservation
- Focal Frequency Loss: Dynamic focus on hard-to-reconstruct frequency components
- Sobel Gradient Loss: Edge information preservation
- Multi-scale consideration for comprehensive feature capture

**Latent Space Analysis**
- Direct utilization of the trained encoder as feature extractor
- Multi-level feature extraction from encoder intermediate layers
- L2 distance calculation in latent space for high-level semantic differences
- No dependency on pre-trained networks for domain-specific feature learning

### Addressing Over-reconstruction Issues

**Synthetic Anomaly Introduction**
- Implementation of artificial defects during training phase
- Methods include random masking and controlled noise injection
- Forces the model to learn repair capabilities rather than pure reconstruction
- Maintains focus on foreground objects for realistic defect simulation

**Data Augmentation Strategy**
- Initially minimal augmentation to preserve anomaly sensitivity
- Conservative augmentation parameters if needed (rotation: ±5°, scale: 0.95-1.05)
- Primary focus on synthetic anomaly generation rather than traditional augmentation

### Network Architecture Variants

**Baseline Architecture**
- Standard autoencoder without skip connections
- Forces information compression through bottleneck
- Suitable for pure reconstruction-based anomaly detection

**Enhanced Architecture**
- Autoencoder with U-Net style skip connections
- Leverages skip connections for precise defect localization when using synthetic anomalies
- Better preservation of fine-grained details and edges

### Implementation Requirements

**Input Specifications**
- Image dimensions: 976 × 176 pixels
- Grayscale images (single channel)
- MVTec AD dataset for evaluation

**Training Configuration**
- Modular loss function system for flexible experimentation
- Configurable loss component weights
- Support for individual loss testing and combinations

**Evaluation Metrics**
- Area Under ROC Curve (AUROC) for detection performance
- Average Precision (AP) for localization accuracy
- Per-Region Overlap (PRO) for pixel-level evaluation
- Visual assessment of reconstruction quality and anomaly heatmaps

### Experimental Design

**Phase 1: Baseline Establishment**
- Train standard autoencoder with various loss combinations
- Evaluate reconstruction quality and anomaly detection performance
- Identify over-reconstruction issues

**Phase 2: Synthetic Anomaly Integration**
- Implement realistic synthetic defect generation
- Compare performance with and without synthetic anomalies
- Optimize synthetic anomaly parameters

**Phase 3: Architecture Comparison**
- Evaluate standard autoencoder versus skip-connection variants
- Ensure fair comparison with similar parameter counts
- Analyze performance on different anomaly scales

**Phase 4: Loss Function Optimization**
- Test individual loss components
- Explore weighted combinations
- Determine optimal configuration for specific defect types

### Key Design Principles

**Modularity**: All components should be independently configurable for experimental flexibility

**Domain Specificity**: Feature learning should be tailored to the specific inspection domain rather than relying on generic pre-trained features

**Realistic Synthesis**: Synthetic anomalies should closely mimic real defects in appearance and location

**Balanced Approach**: Maintain sensitivity to real anomalies while avoiding false positives from normal variations
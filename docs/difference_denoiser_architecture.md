# Difference Denoiser Architecture Documentation

## Overview

The Difference Denoiser is a specialized dual-output autoencoder designed for anomaly detection in optical inspection systems. It leverages the unique property of having multiple reference images to learn and filter out normal variations (noise, alignment errors, systematic patterns) while highlighting true defects.

## Problem Statement

### Image Composition
In optical inspection, each image contains:
- **Pattern**: The expected design/structure
- **Pattern_noise**: Systematic noise (e.g., sensor patterns, lighting variations)
- **Random_noise**: Shot noise, thermal noise, etc.
- **Defect**: Anomalies we want to detect (only in defective samples)

### Traditional Approach Limitations
Simple subtraction (Target - Reference) produces:
```
T - R = (Pattern_diff) + (Pattern_noise_diff) + (Random_noise_diff) + Defect
```
This includes many unwanted variations that can mask small defects.

### Our Goal
Learn to reconstruct only the "normal" differences (Pattern_diff, Pattern_noise_diff, Random_noise patterns) so that abnormal differences (Defects) stand out.

## Architecture Design

### Input Processing
The model takes three images and computes three difference images:
```python
diff1 = Target - Reference1     # Contains possible defect + normal variations
diff2 = Target - Reference2     # Contains possible defect + normal variations'  
ref_diff = Reference1 - Reference2  # Contains only normal variations (no defect)
```

These three channels are concatenated as input: `[diff1, diff2, ref_diff]`

### Network Structure

```
DifferenceDenoiser
├── Shared Encoder (DifferenceDenoiserEncoder)
│   ├── Initial Conv: 3ch → 32ch, stride=2
│   ├── C3k2 Block 1: n=1, e=0.25 (preserve details)
│   ├── Conv + C3k2 Block 2: 32ch → 64ch, n=1, e=0.25
│   ├── Conv + C3k2 Block 3: 64ch → 128ch, n=2, e=0.5
│   ├── Conv + C3k2 Block 4: 128ch → 256ch, n=3, e=0.5
│   └── Bottleneck: 256ch → latent_dim → 256ch
│
├── Reconstruction Decoder (DifferenceReconstructionDecoder)
│   ├── Upsample + Conv + C3k2: 256ch → 128ch
│   ├── Upsample + Conv + C3k2: 128ch → 64ch
│   ├── Upsample + Conv + C3k2: 64ch → 32ch
│   ├── Upsample + Conv: 32ch → 32ch
│   └── Final Conv: 32ch → 3ch (reconstructed diffs)
│
└── Anomaly Decoder (AnomalyDecoder)
    ├── ConvTranspose2d: 256ch → 128ch
    ├── ConvTranspose2d: 128ch → 64ch
    ├── ConvTranspose2d: 64ch → 32ch
    ├── ConvTranspose2d: 32ch → 16ch
    └── Conv2d: 16ch → 1ch (anomaly heatmap)
```

### Key Design Choices

1. **Shared Encoder**: Forces both tasks to use the same feature representation, preventing the anomaly decoder from simply outputting zeros.

2. **Asymmetric C3k2 Parameters**: 
   - Shallow layers (1-2): `n=1, e=0.25` - Minimal processing to preserve small defect details
   - Deep layers (3-4): `n=2-3, e=0.5` - More processing to learn global patterns

3. **Moderate Bottleneck**: `latent_dim=128` provides sufficient compression without losing important information.

4. **Lightweight Anomaly Decoder**: Simple upsampling path to avoid over-processing the anomaly signal.

## Loss Function Design

### Components

1. **Weighted Reconstruction Loss** (Primary driver):
   ```python
   recon_loss = 0.4 * MSE(recon_diff1, diff1) + 
                0.4 * MSE(recon_diff2, diff2) + 
                0.2 * MSE(recon_ref_diff, ref_diff)
   ```
   - Higher weight on diff1/diff2 (contain potential anomalies)
   - Lower weight on ref_diff (auxiliary information)

2. **Anomaly Regularization**:
   ```python
   anomaly_loss = mean(anomaly_map^2)
   ```
   - Encourages near-zero output for normal samples

3. **Smoothness Regularization**:
   ```python
   smooth_loss = TV(anomaly_map)
   ```
   - Prevents noisy/fragmented anomaly maps

### Total Loss
```python
total_loss = 1.0 * recon_loss + 0.5 * anomaly_loss + 0.01 * smooth_loss
```

## Training Strategy

### Data Preparation
- **Training set**: Only normal samples (no defects)
- **Input**: Triplet images (Target, Reference1, Reference2)
- **Processing**: Compute difference images on-the-fly

### Training Dynamics
1. **Encoder** learns to extract features that can reconstruct normal difference patterns
2. **Reconstruction Decoder** learns to reconstruct Pattern_diff, Pattern_noise_diff, and Random_noise statistics
3. **Anomaly Decoder** is forced to output near-zero (no anomalies in training data)

### Why This Works
- The network only sees normal variations during training
- It learns to reconstruct these normal patterns
- During inference, defects create patterns the network has never seen
- These unknown patterns cannot be properly reconstructed
- The reconstruction error and anomaly decoder output highlight these regions

## Inference Pipeline

```python
# 1. Compute differences
diff1 = target - ref1
diff2 = target - ref2  
ref_diff = ref1 - ref2

# 2. Forward pass
anomaly_map, reconstructed_diffs, input_diffs = model(target, ref1, ref2)

# 3. Anomaly detection
# Option A: Use anomaly map directly
anomaly_score = max(abs(anomaly_map))

# Option B: Combine with reconstruction error
recon_error = mean(abs(input_diffs[:2] - reconstructed_diffs[:2]))
combined_score = anomaly_map + alpha * recon_error

# 4. Threshold decision
is_defective = anomaly_score > threshold
```

## Advantages

1. **Leverages Multiple References**: Uses ref_diff as noise level indicator
2. **Unsupervised**: Only requires normal samples for training
3. **Interpretable**: Can visualize what the network considers "normal" vs "abnormal"
4. **Robust**: Dual-task design prevents trivial solutions

## Implementation Details

### File Structure
```
src/
├── models/
│   └── difference_denoiser.py      # Model architecture
├── losses/
│   └── difference_denoiser_loss.py # Loss functions
└── main_difference_denoiser.py     # Training script
```

### Key Parameters
- `latent_dim`: 128 (bottleneck dimension)
- `recon_weight`: 1.0
- `anomaly_weight`: 0.5  
- `smooth_weight`: 0.01
- `batch_size`: 16
- `learning_rate`: 1e-3
- `num_epochs`: 100

### Usage
```bash
cd src
python main_difference_denoiser.py
```

## Experimental Considerations

### For 3×3 to 5×5 Small Defects
The architecture is specifically tuned for detecting very small anomalies:
- Shallow layers use minimal processing (n=1)
- Weighted loss emphasizes diff1/diff2 where defects appear
- 176×976 images downsample to 11×61 at bottleneck (3×3 defect ≈ 0.2×0.2 pixels)

### Potential Improvements
1. **Multi-scale Supervision**: Add auxiliary losses at intermediate layers
2. **Attention Mechanisms**: Help focus on small anomalous regions
3. **Ensemble Methods**: Train multiple models with different random seeds
4. **Post-processing**: Apply morphological operations to clean up anomaly maps

## Theoretical Foundation

### Why Reference Difference Helps
The ref_diff channel provides crucial information:
- In regions with high ref_diff, large diff1/diff2 values are expected (normal noise)
- In regions with low ref_diff but high diff1/diff2, anomalies are likely
- The network learns this correlation implicitly

### Information Bottleneck Principle
The moderate bottleneck (latent_dim=128) forces the network to:
- Compress information efficiently
- Learn only the most important patterns
- Discard random variations that cannot be compressed

### Dual-Task Synergy
The reconstruction task ensures the encoder learns meaningful features, while the anomaly task provides the actual detection capability. Neither task alone would be sufficient.

## Troubleshooting

### Network Outputs All Zeros
- Check if reconstruction loss is working properly
- Ensure normal samples have sufficient variation
- Verify data normalization is correct

### High False Positive Rate
- Increase anomaly_weight in loss
- Add more smoothness regularization
- Consider post-processing with connected components

### Missing Small Defects
- Reduce n in shallow C3k2 blocks
- Increase weight on diff1/diff2 in reconstruction loss
- Try smaller latent_dim to force more selective encoding

## Future Directions

1. **Adaptive Thresholding**: Learn threshold from validation data
2. **Multi-Resolution Processing**: Process at multiple scales in parallel
3. **Temporal Consistency**: If sequential images available, enforce temporal smoothness
4. **Active Learning**: Use model uncertainty to select samples for labeling
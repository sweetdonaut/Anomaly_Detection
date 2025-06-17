# Bottleneck Feature Analysis

This folder contains standalone scripts for analyzing Autoencoder bottleneck layer features.

## File Structure

- `analyze_bottleneck.py`: Analyzes Compact model (with residual connection)
- `analyze_bottleneck_standard.py`: Analyzes Standard Compact model (without residual connection)
- `models/`: Model files directory
  - `compact_model.pth`: Trained Compact Autoencoder model (100 epochs, with residual)
  - `standard_compact_model.pth`: Trained Standard Compact model (100 epochs, no residual)
  - `enhanced_model.pth`: Enhanced model using YOLOv11 C3k2 modules (under development)
- `images/`: Test images directory
  - `000.tiff`, `004.tiff`: Normal samples
  - `defect_type1_001.tiff`, `defect_type2_002.tiff`, `defect_type3_003.tiff`: Various defect types
- `outputs/`: Analysis results output directory

## Requirements

```bash
pip install torch torchvision numpy matplotlib pillow
```

## Usage

Analyze Compact model (with residual connection):
```bash
python analyze_bottleneck.py
```

Analyze Standard Compact model (without residual connection):
```bash
python analyze_bottleneck_standard.py
```

Compare both models:
```bash
python analyze_bottleneck_standard.py --model both
```

## Output Description

The script generates the following for each test image:

1. **Visualization** (`bottleneck_analysis_XXX.png`):
   - Original and reconstructed images
   - Reconstruction error heatmap
   - Bottleneck features distribution histogram
   - Mean feature map and variance map
   - Top 14 most important feature channels (sorted by activity)

2. **Statistics Report** (`bottleneck_statistics.txt`):
   - Detailed statistics for each feature channel
   - Channels sorted by activation magnitude

## CompactAutoencoder Architecture Details

### Model Overview
- **Input Size**: 1 × 976 × 176 (C×H×W)
- **Total Parameters**: 918,497
- **Compression Ratio**: ~256:1

### Encoder Layer Outputs
```
Layer 1: Conv2d(1→32, k=3, s=2) → [1, 32, 488, 88]
Layer 2: Conv2d(32→64, k=3, s=2) → [1, 64, 244, 44]
Layer 3: Conv2d(64→128, k=3, s=2) → [1, 128, 122, 22]
Layer 4: Conv2d(128→256, k=3, s=2) → [1, 256, 61, 11]
```

### Bottleneck Layer
- **Input/Output**: [1, 256, 61, 11] (size unchanged)
- **Internal Structure**:
  1. Conv2d(256→256, k=1) + BN + SiLU
  2. Conv2d(256→256, k=1) + BN + SiLU
  3. Residual connection (output = bottleneck output + original input)

### Decoder Layer Outputs
```
Layer 1: ConvTranspose2d(256→128, k=3, s=2) → [1, 128, 122, 22]
Layer 2: ConvTranspose2d(128→64, k=3, s=2) → [1, 64, 244, 44]
Layer 3: ConvTranspose2d(64→32, k=3, s=2) → [1, 32, 488, 88]
Layer 4: ConvTranspose2d(32→32, k=3, s=2) → [1, 32, 976, 176]
Final Layer: Conv2d(32→1, k=1) + Sigmoid → [1, 1, 976, 176]
```

### Parameter Distribution
- **Encoder Parameters**: 388,800
- **Bottleneck Parameters**: 132,608
- **Decoder Parameters**: 397,056
- **Final Layer Parameters**: 33

### Bottleneck Feature Analysis
- **Feature Map Size**: 256 × 61 × 11
- **Total Features**: 171,776
- **Spatial Dimension**: Compressed from 171,776 pixels to 671 feature points

## Standard Compact Autoencoder Architecture Details

### Model Overview
- **Input Size**: 1 × 976 × 176 (C×H×W)
- **Total Parameters**: 918,497
- **Compression Ratio**: ~256:1
- **Key Difference**: No residual connection, forcing information through bottleneck

### Architecture Comparison (vs Compact with Residual)
| Feature | Compact (with residual) | Standard Compact (no residual) |
|---------|------------------------|------------------------------|
| Bottleneck Design | Has residual connection | No residual connection |
| Information Flow | Allows encoded info to pass directly | Forces through bottleneck |
| Training Stability | Higher | Lower |
| Feature Compression | More relaxed | Stricter |
| Final Training Loss | ~0.0003 | ~0.0004 |

### Training Results
- **Training Duration**: 100 epochs
- **Initial Loss**: 0.0143
- **Final Loss**: 0.0004
- **Optimizer**: Adam (lr=1e-3)

## Enhanced Autoencoder Architecture Details (Under Development)

### Model Overview
- **Based on**: YOLOv11 C3k2 modules
- **Input Size**: 1 × 976 × 176 (C×H×W)
- **Architecture Features**: Uses C3k2 modules instead of traditional convolution layers for stronger feature extraction
- **Detailed Architecture**: (To be added after implementation)

## Notes

- The script defaults to CUDA if available, automatically switches to CPU otherwise
- All necessary code is included in the analysis scripts, no external dependencies
- Enhanced model features are under development
# 模型特徵提取方法說明

## 概述

所有模型都實作了統一的特徵提取介面，方便進行特徵分析和異常檢測。

## 統一方法定義

### 1. `get_features(x)`
- **功能**：提取編碼器的最終輸出（瓶頸層之前的特徵）
- **返回**：編碼器最後一層的輸出張量
- **用途**：獲取未經壓縮的高維特徵表示

### 2. `get_latent(x)`
- **功能**：提取潛在空間表示（瓶頸層之後的特徵）
- **返回**：經過瓶頸層壓縮後的潛在特徵
- **用途**：獲取壓縮後的低維特徵表示，常用於異常檢測

### 3. `get_multi_level_features(x)` (僅 U-Net 架構)
- **功能**：提取多個編碼器層級的特徵
- **返回**：串接的多層級特徵向量（已展平）
- **用途**：獲取多尺度特徵，提供更豐富的特徵表示

## 模型特徵維度對照表

| 模型 | 輸入尺寸 | get_features 輸出 | get_latent 輸出 | get_multi_level_features 輸出 |
|------|----------|-------------------|-----------------|-------------------------------|
| BaselineAutoencoder | [B, 1, 256, 256] | [B, 512, 8, 8] | [B, 512, 8, 8] | N/A |
| EnhancedAutoencoder | [B, 1, 256, 256] | [B, 512, 8, 8] | [B, 512, 8, 8] | [B, 992] |
| CompactAutoencoder | [B, 1, 256, 256] | [B, 256, 16, 16] | [B, 256, 16, 16] | N/A |
| CompactUNetAutoencoder | [B, 1, 256, 256] | [B, 256, 16, 16] | [B, 256, 16, 16] | [B, 480] |
| C3k2Autoencoder | [B, 1, 256, 256] | [B, 256, 16, 16] | [B, 256, 16, 16] | N/A |
| StandardCompactAutoencoder | [B, 1, 256, 256] | [B, 256, 16, 16] | [B, 256, 16, 16] | N/A |

註：B = batch size

## 使用範例

```python
# 基本使用
model = CompactAutoencoder(input_size=(256, 256))
model.eval()

# 準備輸入
x = torch.randn(4, 1, 256, 256)  # batch_size=4

# 獲取編碼器特徵
encoder_features = model.get_features(x)  # Shape: [4, 256, 16, 16]

# 獲取潛在特徵
latent_features = model.get_latent(x)    # Shape: [4, 256, 16, 16]

# 對於 U-Net 架構
unet_model = CompactUNetAutoencoder(input_size=(256, 256))
multi_features = unet_model.get_multi_level_features(x)  # Shape: [4, 480]
```

## LatentSpaceAnalyzer 整合

`LatentSpaceAnalyzer` 已更新為使用統一的方法：

1. 優先使用 `get_latent()` 方法
2. 對於 U-Net 架構，支援 `get_multi_level_features()`
3. 自動處理特徵展平（如需要）

```python
from utils.model_utils import LatentSpaceAnalyzer

analyzer = LatentSpaceAnalyzer(model, device='cuda')
analyzer.fit_normal_features(normal_dataloader)
anomaly_scores = analyzer.compute_anomaly_score(test_batch)
```

## 重要變更記錄

1. **移除的方法**：
   - `get_latent_features()` → 改為 `get_latent()`
   - `get_bottleneck_output()` → 已移除（與 `get_latent()` 重複）

2. **修正的實作**：
   - CompactAutoencoder 的 `get_latent()` 不再包含殘差連接
   - 所有模型的 `get_latent()` 現在都返回純粹的瓶頸層輸出

3. **新增的方法**：
   - BaselineAutoencoder 新增 `get_features()`
   - EnhancedAutoencoder 新增 `get_features()` 和 `get_latent()`
   - CompactUNetAutoencoder 新增 `get_features()` 和 `get_latent()`
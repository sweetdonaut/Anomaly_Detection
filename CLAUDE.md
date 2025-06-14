# CLAUDE.md

此檔案為 Claude Code (claude.ai/code) 在此代碼庫中工作時的指導文件。

## 語言偏好
請使用繁體中文回應所有對話和說明。

## 專案概述

這是一個基於重建模型的無監督異常檢測系統，專為工業缺陷檢測設計。系統僅使用正常影像訓練自編碼器，透過分析重建誤差來檢測異常。專案實作了模組化架構，支援多種損失函數、合成異常生成和完整的評估指標。

## 架構設計

### 核心組件

1. **模組化損失函數框架 (ModularLossManager)**
   - **MSE Loss**：像素級重建精度
   - **SSIM Loss**：結構相似性保留，包含詳細實作與參數驗證
   - **Multi-Scale SSIM Loss**：多尺度結構相似性，同時捕捉局部細節與全局結構
   - **Focal Frequency Loss**：動態聚焦於難以重建的頻率成分
   - **Sobel Gradient Loss**：邊緣資訊保留
   - 支援靈活的權重配置和組合測試
   - 所有損失函數繼承自 `BaseLoss`，統一權重管理

2. **雙重網路架構**
   - **BaselineAutoencoder**：無跳躍連接的標準自編碼器，強制資訊壓縮
   - **EnhancedAutoencoder**：具有 U-Net 風格跳躍連接，精確缺陷定位
   - 兩種架構都針對單通道 976×176 影像優化

3. **合成異常生成器 (SyntheticAnomalyGenerator)**
   - 生成亮點/暗點異常，外觀真實
   - 橢圓形狀，邊緣平滑過渡
   - 可配置大小和強度變化
   - 支援批次處理

4. **潛在空間分析器 (LatentSpaceAnalyzer)**
   - 多層特徵提取（來自編碼器中間層）
   - L2 距離計算高層語義差異
   - 無需預訓練網路的領域特定特徵學習

5. **異常視覺化器 (AnomalyVisualizer)**
   - 視覺化原始影像、重建影像和異常熱圖
   - 異常分數儲存功能
   - 批次處理支援

## 主要特色

- **影像尺寸**：976×176 像素（灰階單通道）
- **模組化設計**：便於實驗不同配置
- **訓練模式**：支援有/無合成異常的訓練
- **保守資料增強**：縮放係數 0.95-1.05
- **綜合異常評分**：結合重建誤差和潛在空間分析
- **單通道優化**：所有組件皆已驗證支援單通道影像處理
- **全英文程式碼**：註解和文檔使用英文，適合生產環境

## 常用指令

### 訓練最新版模型 (v3)
```bash
python MVTec_unsupervised/anomaly_detection_v3.py
```

### 訓練前一版模型 (v2)
```bash
python MVTec_unsupervised/anomaly_detection_v2.py
```

## 配置設定

主要配置參數（在 `main()` 函數中）：
```python
config = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'batch_size': 16,
    'num_epochs': 100,
    'lr': 1e-3,
    'image_size': (976, 176),
    'architecture': 'enhanced',  # 'baseline' 或 'enhanced'
    'use_synthetic_anomalies': True,
    'loss_config': {
        # MSE Loss: 基本像素級重建
        'mse': {
            'class': MSELoss,
            'weight': 0.3
        },
        # SSIM Loss: 結構相似性保留
        'ssim': {
            'class': SSIMLoss,
            'weight': 0.3,
            'params': {'window_size': 11, 'sigma': 1.5}
        },
        # Focal Frequency Loss: 動態聚焦難重建區域
        'focal_freq': {
            'class': FocalFrequencyLoss,
            'weight': 0.2,
            'params': {'alpha': 1.0, 'patch_factor': 1}
        },
        # Sobel Gradient Loss: 邊緣保留
        'sobel': {
            'class': SobelGradientLoss,
            'weight': 0.2
        }
    },
    'save_path': './models'
}
```

### 範例配置

1. **僅使用 MSE + SSIM（快速訓練）**
```python
'loss_config': {
    'mse': {'class': MSELoss, 'weight': 0.5},
    'ssim': {'class': SSIMLoss, 'weight': 0.5, 'params': {'window_size': 11}}
}
```

2. **強調頻率域（適合紋理影像）**
```python
'loss_config': {
    'mse': {'class': MSELoss, 'weight': 0.2},
    'focal_freq': {'class': FocalFrequencyLoss, 'weight': 0.5, 'params': {'alpha': 2.0}},
    'sobel': {'class': SobelGradientLoss, 'weight': 0.3}
}
```

3. **多尺度 SSIM 重點**
```python
'loss_config': {
    'mse': {'class': MSELoss, 'weight': 0.3},
    'ms_ssim': {'class': MultiScaleSSIMLoss, 'weight': 0.5, 'params': {'scales': 3}},
    'sobel': {'class': SobelGradientLoss, 'weight': 0.2}
}
```

## 實驗設計流程

1. **階段一：基準建立**
   - 使用標準自編碼器測試各種損失組合
   - 評估重建品質和異常檢測效能

2. **階段二：合成異常整合**
   - 實作真實缺陷生成
   - 比較有/無合成異常的效能

3. **階段三：架構比較**
   - 評估標準與跳躍連接變體
   - 分析不同異常尺度的表現

4. **階段四：損失函數優化**
   - 測試個別損失組件
   - 探索加權組合

## 最新更新 (v3)

1. **增強的 SSIM Loss 實作**
   - 詳細的文檔和參數驗證
   - 動態通道支援與正確的設備處理
   - 單樣本損失計算功能

2. **新增 Multi-Scale SSIM Loss**
   - 同時捕捉細節與全局結構
   - 可配置尺度與預設權重
   - 自動調整小影像處理

3. **統一的損失函數架構**
   - 所有損失函數現在都繼承自 `BaseLoss`
   - 跨損失函數的一致權重管理
   - 改進與 `ModularLossManager` 的相容性

4. **完整英文介面**
   - 所有程式碼註解和文檔使用英文
   - 為純英文環境準備的生產就緒程式碼
   - 保持清晰度和技術準確性

## 相依套件

- PyTorch（建議使用 CUDA 支援）
- torchvision
- numpy
- PIL (Pillow)
- matplotlib
- scipy
- tqdm
- pathlib
- scikit-learn（用於評估指標）
- opencv-python（用於合成異常生成）
- typing（用於型別提示）
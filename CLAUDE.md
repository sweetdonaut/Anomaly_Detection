# CLAUDE.md

此檔案為 Claude Code (claude.ai/code) 在此代碼庫中工作時的指導文件。

## 語言偏好
請使用繁體中文回應所有對話和說明。

## 專案概述

這是一個基於重建模型的無監督異常檢測系統，專為工業缺陷檢測設計。系統僅使用正常影像訓練自編碼器，透過分析重建誤差來檢測異常。專案實作了模組化架構，支援多種損失函數、合成異常生成和完整的評估指標。

## 架構設計

### 核心組件

1. **模組化損失函數框架 (ModularLossFunction)**
   - MSE：像素級重建精度
   - SSIM：結構相似性保留
   - Focal Frequency Loss：動態聚焦於難以重建的頻率成分
   - Sobel Gradient Loss：邊緣資訊保留
   - 支援靈活的權重配置和組合測試

2. **雙重網路架構**
   - **BaselineAutoencoder**：無跳躍連接的標準自編碼器，強制資訊壓縮
   - **EnhancedAutoencoder**：具有 U-Net 風格跳躍連接，精確缺陷定位

3. **合成異常生成器 (SyntheticAnomalyGenerator)**
   - 隨機遮罩：矩形區域遮蔽
   - 控制噪音：高斯噪音注入
   - 局部模糊：區域性模糊處理
   - 合成刮痕：線條缺陷模擬

4. **潛在空間分析器 (LatentSpaceAnalyzer)**
   - 多層特徵提取（來自編碼器中間層）
   - L2 距離計算高層語義差異
   - 無需預訓練網路的領域特定特徵學習

5. **評估系統 (AnomalyEvaluator)**
   - AUROC：檢測效能評估
   - Average Precision (AP)：定位精度
   - Per-Region Overlap (PRO)：像素級評估

## 主要特色

- 影像尺寸：976×176 像素（灰階單通道）
- 模組化設計便於實驗不同配置
- 支援有/無合成異常的訓練模式
- 保守的資料增強策略（旋轉±5°，縮放0.95-1.05）
- 結合重建誤差和潛在空間分析的綜合異常評分

## 常用指令

### 訓練新版模型
```python
python MVTec_unsupervised/anomaly_detection_v2.py
```

### 舊版模型訓練（僅供參考）
```python
python MVTec_unsupervised/AE_single_channel.py
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
        'use_mse': True, 'mse_weight': 0.3,
        'use_ssim': True, 'ssim_weight': 0.3,
        'use_focal_freq': True, 'focal_freq_weight': 0.2,
        'use_sobel': True, 'sobel_weight': 0.2
    },
    'save_path': './models'
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
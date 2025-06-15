# CLAUDE.md

此檔案為 Claude Code (claude.ai/code) 在此代碼庫中工作時的指導文件。

## 語言偏好
請使用繁體中文回應所有對話和說明。

## 專案快速概述

**無監督異常檢測系統** - 使用自編碼器進行工業缺陷檢測
- 核心原理：僅用正常影像訓練，透過重建誤差檢測異常
- 模組化架構，易於擴展和實驗

## 關鍵檔案位置

```
src/
├── main.py                    # 主訓練入口
├── main_experiments.py        # 實驗管理（MSE vs Focal Frequency）
├── models/
│   ├── baseline.py           # 標準自編碼器（無跳躍連接）
│   └── enhanced.py           # U-Net 風格自編碼器（有跳躍連接）
├── losses/
│   ├── manager.py            # 損失函數管理器
│   └── [各種損失函數實作]
└── utils/
    └── training.py           # 訓練循環實作
```

## 重要技術特點

1. **支援單通道灰階影像**（已驗證所有模組支援）
2. **可變影像尺寸**（預設 1024×1024，可調整）
3. **模組化損失函數系統**（可自由組合不同損失）
4. **合成異常生成**（訓練時可選）

## 已知問題與注意事項

1. **LatentSpaceAnalyzer 未整合**：在 main.py 中創建但未使用
2. **EnhancedAutoencoder 缺少 input_size 參數**：初始化時無法指定尺寸
3. **評估功能簡單**：evaluate_model 只計算 MSE，未計算 AUROC

## 快速操作指令

```bash
# 訓練單一模型
cd src && python main.py

# 執行實驗比較（4 種組合）
cd src && python main_experiments.py

# 執行測試
cd src/test && python test_modular.py
```

## 常見任務快速參考

### 修改損失函數組合
編輯 main.py 中的 `loss_config`：
```python
'loss_config': {
    'mse': {'class': MSELoss, 'weight': 0.5},
    'ssim': {'class': SSIMLoss, 'weight': 0.5}
}
```

### 切換模型架構
修改 config 中的 `'architecture'`：
- `'baseline'`：標準自編碼器
- `'enhanced'`：U-Net 風格

### 調整影像尺寸
修改 config 中的 `'image_size'`：
```python
'image_size': (256, 256)  # 或任意正方形尺寸
```

## MVTec AD 資料集路徑
- 主要路徑：`/home/yclai/vscode_project/Anomaly_Detection/MVTec_AD_dataset`
- 測試類別：`grid`（網格）

## 相依套件
- PyTorch、torchvision、numpy、PIL、matplotlib、scipy、tqdm、scikit-learn、opencv-python

## 待辦實驗項目（針對 176×976 非對稱影像）

### 背景
目前使用標準自編碼器架構在 176×976 的極端長條形影像上表現不佳，模型容易過度擬合（完全 mapping），無法有效檢測異常。需要針對非對稱影像設計特殊架構。

### 實驗方案（按優先順序）

1. **非對稱池化策略**
   - 實作不同方向的池化層（如 (2,1) 和 (1,2)）
   - 前期多做垂直池化，後期平衡池化
   - 目標：保持特徵圖的相對對稱性

2. **非對稱卷積核**
   - 使用 (5,1)、(1,5)、(3,3) 等不同形狀的卷積核
   - 分別捕捉垂直、水平和局部特徵
   - 可並行處理後融合

3. **多尺度特徵融合**
   - 並行處理不同感受野的特徵
   - 結合長條形和方形的特徵提取
   - 在編碼器中期進行特徵融合

4. **分割處理策略**
   - 將 176×976 切成多個 176×176 的方塊
   - 使用標準正方形架構處理每個方塊
   - 在 latent space 合併所有方塊的特徵
   - 優點：可重用現有架構

5. **注意力機制**
   - 加入空間注意力模組（如 CBAM）
   - 讓模型自動學習關注重要區域
   - 特別適合處理長條形影像的不均勻特徵分布

### 實驗記錄
- 2025/06/15：標準 enhanced autoencoder 在 176×976 影像上出現完全 mapping 問題
- 待測試：上述五個方案
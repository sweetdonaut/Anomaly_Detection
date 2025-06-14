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
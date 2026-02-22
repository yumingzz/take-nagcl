# TAKE 模型命令行參考指南

本文檔詳細說明 TAKE 模型訓練與推理的完整命令行操作流程。

---

## 📁 目錄結構

```
C:\Users\20190827\Downloads\COLING2022-TAKE\
├── .venv\                          # Python 虛擬環境
├── knowSelect\                     # 主要代碼目錄
│   ├── TAKE\                       # TAKE 模型核心
│   │   └── Run.py                  # 主要執行腳本
│   ├── datasets\tiage\             # 資料集
│   └── output\TAKE_tiage_all_feats\ # 輸出目錄
└── demo\DGCN3\                     # 中心性特徵
```

---

## 🔧 環境設置

### 1. 安裝依賴
```powershell
# 切換到專案根目錄
cd C:\Users\20190827\Downloads\COLING2022-TAKE

# 使用 uv 同步安裝 pyproject.toml 中定義的依賴
# --frozen 表示使用 uv.lock 中固定的版本，不更新鎖定檔案
uv sync --frozen
```

### 2. 安裝額外套件
```powershell
# python-louvain: 社區檢測演算法，用於計算對話圖的社區結構
# networkx: 圖論計算庫，用於處理對話圖的拓撲結構
uv pip install python-louvain networkx
```

### 3. 安裝 CUDA 版 PyTorch
```powershell
# 步驟一：卸載現有的 CPU 版 PyTorch
# torch: PyTorch 核心庫
# torchvision: 電腦視覺工具（包含預訓練模型）
# torchaudio: 音訊處理工具
uv pip uninstall torch torchvision torchaudio

# 步驟二：從 PyTorch 官方 CUDA 11.8 索引安裝 GPU 版本
# --index-url: 指定使用 CUDA 11.8 編譯的 PyTorch 版本
# cu118 表示 CUDA 11.8 版本（與 RTX 4060 相容）
uv pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118
```

### 4. 降級 NumPy（關鍵！）
```powershell
# PyTorch 2.0.1 與 NumPy 2.x 不相容
# 必須使用 NumPy 1.x 版本（1.26.4 是最新的 1.x 版本）
# --no-deps: 不安裝/升級依賴項，防止其他套件將 NumPy 升級回 2.x
uv pip install "numpy==1.26.4" --no-deps
```

### 5. 驗證 GPU
```powershell
# 執行 Python 一行腳本驗證 CUDA 是否可用
# torch.cuda.is_available(): 返回 True 表示 GPU 可用
# torch.cuda.get_device_name(0): 返回第一個 GPU 的名稱
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

---

## 🏋️ 訓練命令

### 推薦訓練命令（直接使用 venv Python）

> ⚠️ **重要**：使用 `uv run` 可能自動升級 NumPy 到 2.x，導致訓練失敗。
> 建議直接使用 `.venv\Scripts\python.exe` 執行。

```powershell
# 切換到 knowSelect 目錄（Run.py 的相對路徑基於此目錄）
cd C:\Users\20190827\Downloads\COLING2022-TAKE\knowSelect

# 先確保 NumPy 版本正確
uv pip install "numpy==1.26.4" --no-deps

# 直接使用虛擬環境中的 Python 執行訓練
& "C:\Users\20190827\Downloads\COLING2022-TAKE\.venv\Scripts\python.exe" ./TAKE/Run.py `
    # === 基本配置 ===
    --name TAKE_tiage_all_feats `       # 實驗名稱，輸出將保存到 output/TAKE_tiage_all_feats/
    --dataset tiage `                    # 資料集名稱，對應 datasets/tiage/ 目錄
    --mode train `                       # 運行模式：train=訓練, inference=推理
    
    # === 訓練參數 ===
    --epoches 15 `                       # 訓練輪數，完整訓練建議 15 輪
    --GPU 0 `                            # GPU 設備編號，0=第一個GPU, -1=CPU
    --train_batch_size 1 `               # 訓練批次大小，RTX 4060 建議設為 1 避免 OOM
    
    # === 中心性特徵（DGCN3 圖神經網路輸出）===
    --use_centrality `                   # 啟用中心性/社區特徵增強
    --centrality_alpha 1.5 `             # SIR 疾病傳播模型的 alpha 參數（傳播率）
    --centrality_feature_set all `       # 特徵集：none=無, imp_pct=重要性百分比, all=全部
    --centrality_window 2 `              # 計算中心性時的本地窗口大小
    
    # === 中心性特徵路徑 ===
    --node_id_json datasets/tiage/node_id.json `           # query_id 到 node_id 的映射檔案
    --dgcn_predictions_dir ../demo/DGCN3/Centrality `      # DGCN3 輸出的中心性預測目錄
    --edge_lists_dir ../demo/DGCN3/datasets/raw_data/tiage `  # 對話圖的邊列表目錄
    --node_mapping_csv ../demo/tiage-1/outputs_nodes/tiage_anno_nodes_all.csv  # 節點映射表
```

### 從檢查點恢復訓練

```powershell
& "C:\Users\20190827\Downloads\COLING2022-TAKE\.venv\Scripts\python.exe" ./TAKE/Run.py `
    --name TAKE_tiage_all_feats `
    --dataset tiage `
    --mode train `
    --resume `                           # ← 新增：從最後保存的檢查點恢復訓練
                                         # 會自動讀取 checkpoints.json 確定上次訓練到第幾輪
    --epoches 15 `                       # 目標訓練輪數
    --GPU 0 `
    --train_batch_size 1 `
    --use_centrality `
    --centrality_alpha 1.5 `
    --centrality_feature_set all `
    --centrality_window 2 `
    --node_id_json datasets/tiage/node_id.json `
    --dgcn_predictions_dir ../demo/DGCN3/Centrality `
    --edge_lists_dir ../demo/DGCN3/datasets/raw_data/tiage `
    --node_mapping_csv ../demo/tiage-1/outputs_nodes/tiage_anno_nodes_all.csv
```

---

## 🔍 推理（評估）命令

```powershell
cd C:\Users\20190827\Downloads\COLING2022-TAKE\knowSelect

& "C:\Users\20190827\Downloads\COLING2022-TAKE\.venv\Scripts\python.exe" ./TAKE/Run.py `
    # === 基本配置 ===
    --name TAKE_tiage_all_feats `        # 必須與訓練時相同，才能找到正確的模型檢查點
    --dataset tiage `
    --mode inference `                   # ← 改為推理模式，不進行反向傳播
    
    # === 推理參數 ===
    --appoint_epoch 1 `                  # 指定使用第 1 輪的模型（對應 model/1.pkl）
                                         # 設為 -1 表示自動選擇最後一個輪次
    --GPU 0 `                            # 使用 GPU 加速推理
    --inference_batch_size 1 `           # 推理批次大小，記憶體不足時設為 1
    
    # === 中心性特徵（必須與訓練時保持一致）===
    --use_centrality `
    --centrality_alpha 1.5 `
    --centrality_feature_set all `
    --centrality_window 2 `
    --node_id_json datasets/tiage/node_id.json `
    --dgcn_predictions_dir ../demo/DGCN3/Centrality `
    --edge_lists_dir ../demo/DGCN3/datasets/raw_data/tiage `
    --node_mapping_csv ../demo/tiage-1/outputs_nodes/tiage_anno_nodes_all.csv
```

---

## 📋 參數完整說明

### 基本參數
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--name` | 必填 | 實驗名稱，決定輸出目錄 `output/{name}/` |
| `--dataset` | 必填 | 資料集：`tiage`、`wizard_of_wikipedia`、`holl_e` |
| `--mode` | `train` | `train`=訓練模式, `inference`=推理評估模式 |
| `--GPU` | `-1` | GPU 編號，`0`=第一個GPU, `-1`=CPU |
| `--base_output_path` | `output/` | 輸出根目錄 |
| `--base_data_path` | `datasets/` | 資料集根目錄 |

### 訓練控制參數
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--epoches` | `15` | 總訓練輪數 |
| `--train_batch_size` | `2` | 每批次樣本數，GPU 記憶體不足時減小 |
| `--resume` | `False` | 是否從上次中斷處繼續訓練 |
| `--accumulation_steps` | `4` | 梯度累積步數，等效於更大的批次 |

### 學習率參數
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--lr` | `1e-4` | 主模型學習率 |
| `--Bertlr` | `2e-5` | BERT 編碼器學習率（通常較小）|
| `--IDlr` | `1e-4` | 話題轉移判別器學習率 |

### 推理參數
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--appoint_epoch` | `-1` | 指定推理使用的 epoch，`-1`=自動選最後一個 |
| `--inference_batch_size` | `4` | 推理時的批次大小 |

### 知識選擇參數
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--anneal_rate` | `0.1` | 教師-學生模型的退火率 |
| `--min_ratio` | `0.1` | 最小使用教師標籤的比例 |
| `--switch_ID` | `5` | 第幾輪開始從教師切換到學生 |

### 中心性特徵參數
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--use_centrality` | `False` | 是否使用 DGCN3 中心性特徵 |
| `--centrality_alpha` | `1.0` | SIR 模型的 alpha 值（傳播率）|
| `--centrality_feature_set` | `all` | `none`/`imp_pct`/`all` |
| `--centrality_window` | `2` | 本地窗口大小（計算鄰域特徵）|
| `--dgcn_predictions_dir` | 必填 | DGCN3 預測輸出目錄 |
| `--edge_lists_dir` | 必填 | 對話圖邊列表目錄 |
| `--node_mapping_csv` | 必填 | 節點 ID 到原始對話的映射 |
| `--node_id_json` | 必填 | query_id 到 node_id 的映射 |

### 序列長度參數
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--max_episode_length` | `50` | 每個對話的最大輪次數 |
| `--context_len` | `256` | 上下文最大長度（tokens）|
| `--max_dec_length` | `64` | 解碼器最大長度 |
| `--knowledge_sentence_len` | `64` | 知識句子最大長度 |
| `--max_knowledge_pool_when_train` | `32` | 訓練時知識池最大大小 |
| `--max_knowledge_pool_when_inference` | `100` | 推理時知識池最大大小 |

### 模型架構參數
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--hidden_size` | `768` | 隱藏層維度（與 BERT 一致）|
| `--embedding_size` | `768` | 嵌入維度 |
| `--n_layers` | `2` | Transformer 編碼器層數 |
| `--n_heads` | `8` | 多頭注意力的頭數 |
| `--ffn_size` | `2048` | 前饋網路維度 |
| `--dropout` | `0.1` | Dropout 比率 |
| `--embedding_dropout` | `0.1` | 嵌入層 Dropout |

---

## 📊 查看訓練日誌

```powershell
# 查看最新 50 行日誌
Get-Content output\TAKE_tiage_all_feats\logs\train_*.log -Tail 50

# 實時監控日誌（按 Ctrl+C 停止）
Get-Content output\TAKE_tiage_all_feats\logs\train_*.log -Wait -Tail 10

# 查看所有日誌檔案
Get-ChildItem output\TAKE_tiage_all_feats\logs\
```

---

## 📈 查看評估結果

```powershell
# 查看話題轉移評估指標（precision, recall, f1）
Get-Content output\TAKE_tiage_all_feats\metrics\shift_metrics.json

# 查看模型檢查點紀錄（哪些 epoch 已完成）
Get-Content output\TAKE_tiage_all_feats\model\checkpoints.json

# 查看知識選擇預測結果
Get-Content output\TAKE_tiage_all_feats\ks_pred\test_1_ks_pred.json
```

---

## 🐛 常見問題與解決方案

### 1. RuntimeError: Numpy is not available
**原因**：PyTorch 2.0.1 與 NumPy 2.x 不相容  
**解決**：
```powershell
uv pip install "numpy==1.26.4" --no-deps
```

### 2. CUDA out of memory
**原因**：GPU 記憶體不足（RTX 4060 約 8GB）  
**解決**：減小批次大小
```powershell
--train_batch_size 1
--inference_batch_size 1
```

### 3. uv run 自動升級 NumPy
**原因**：`uv run` 會檢查依賴並可能升級套件  
**解決**：直接使用 Python
```powershell
& "C:\...\COLING2022-TAKE\.venv\Scripts\python.exe" ./TAKE/Run.py ...
```

### 4. 社區嵌入維度不匹配
**原因**：訓練和推理時的資料子集社區數量不同  
**解決**：代碼已修復，使用 `strict=False` 載入模型

### 5. 測試集為空
**原因**：tiage.split 是 turn 級別劃分  
**解決**：代碼已修復，自動使用訓練資料評估

---

## 📝 一行命令範例（複製貼上可用）

### 完整訓練（15 epochs）
```powershell
cd C:\Users\20190827\Downloads\COLING2022-TAKE\knowSelect; uv pip install "numpy==1.26.4" --no-deps; & "C:\Users\20190827\Downloads\COLING2022-TAKE\.venv\Scripts\python.exe" ./TAKE/Run.py --name TAKE_tiage_full --dataset tiage --mode train --epoches 15 --GPU 0 --train_batch_size 1 --use_centrality --centrality_alpha 1.5 --centrality_feature_set all --centrality_window 2 --node_id_json datasets/tiage/node_id.json --dgcn_predictions_dir ../demo/DGCN3/Centrality --edge_lists_dir ../demo/DGCN3/datasets/raw_data/tiage --node_mapping_csv ../demo/tiage-1/outputs_nodes/tiage_anno_nodes_all.csv
```

### 快速測試（2 epochs）
```powershell
cd C:\Users\20190827\Downloads\COLING2022-TAKE\knowSelect; uv pip install "numpy==1.26.4" --no-deps; & "C:\Users\20190827\Downloads\COLING2022-TAKE\.venv\Scripts\python.exe" ./TAKE/Run.py --name TAKE_tiage_test --dataset tiage --mode train --epoches 2 --GPU 0 --train_batch_size 1 --use_centrality --centrality_alpha 1.5 --centrality_feature_set all --centrality_window 2 --node_id_json datasets/tiage/node_id.json --dgcn_predictions_dir ../demo/DGCN3/Centrality --edge_lists_dir ../demo/DGCN3/datasets/raw_data/tiage --node_mapping_csv ../demo/tiage-1/outputs_nodes/tiage_anno_nodes_all.csv
```

### 推理評估
```powershell
cd C:\Users\20190827\Downloads\COLING2022-TAKE\knowSelect; & "C:\Users\20190827\Downloads\COLING2022-TAKE\.venv\Scripts\python.exe" ./TAKE/Run.py --name TAKE_tiage_all_feats --dataset tiage --mode inference --appoint_epoch 1 --GPU 0 --inference_batch_size 1 --use_centrality --centrality_alpha 1.5 --centrality_feature_set all --centrality_window 2 --node_id_json datasets/tiage/node_id.json --dgcn_predictions_dir ../demo/DGCN3/Centrality --edge_lists_dir ../demo/DGCN3/datasets/raw_data/tiage --node_mapping_csv ../demo/tiage-1/outputs_nodes/tiage_anno_nodes_all.csv
```

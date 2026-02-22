# TAKE + Tiage 話題轉移檢測系統文檔

> 基於 TAKE 模型與 Tiage 數據集的話題轉移檢測與對話生成系統

---

## 1. 系統架構概覽

```mermaid
graph TB
    subgraph "輸入數據"
        A[("Tiage 對話數據集")]
        B[("節點中心性 CSV")]
    end

    subgraph "knowSelect 模組"
        C["數據預處理<br/>Utils_TAKE.py"]
        D["TAKE 模型<br/>Model.py"]
        E["話題轉移檢測<br/>TeacherTopicShiftDiscriminator"]
        F["知識選擇<br/>TopicShiftedSelector"]
    end

    subgraph "dialogen 模組"
        G["GPT-2 數據集<br/>gpt2Dataset.py"]
        H["GPT-2 對話生成<br/>Model.py"]
    end

    subgraph "輸出結果"
        I[("話題轉移預測<br/>0/1")]
        J[("Top-K 高中心性句子")]
        K[("Precision/Recall/F1")]
        L[("生成的對話回復")]
    end

    A --> C
    B --> D
    C --> D
    D --> E
    D --> F
    E --> I
    E --> J
    E --> K
    F --> G
    G --> H
    H --> L

    style A fill:#e1f5fe
    style B fill:#e1f5fe
    style I fill:#c8e6c9
    style J fill:#c8e6c9
    style K fill:#c8e6c9
    style L fill:#c8e6c9
```

---

## 2. 算法流程

### 2.1 話題轉移檢測流程

```mermaid
flowchart TD
    A["開始"] --> B["載入對話數據"]
    B --> C["載入節點中心性特徵"]
    C --> D["BERT 編碼對話上下文"]
    D --> E["計算 6 維結構特徵"]
    
    subgraph "6 維結構特徵"
        E1["imp_raw: 原始中心性"]
        E2["imp_pct: 重要性分位數"]
        E3["imp_delta_prev: 與前節點差值"]
        E4["imp_delta_next: 與後節點差值"]
        E5["imp_z_local: 局部 Z-score"]
        E6["imp_minus_window_mean: 與窗口均值差"]
    end
    
    E --> E1 & E2 & E3 & E4 & E5 & E6
    E1 & E2 & E3 & E4 & E5 & E6 --> F["特徵融合層"]
    F --> G["TeacherTopicShiftDiscriminator"]
    G --> H{"預測話題轉移?"}
    H -->|"是 (1)"| I["TopicShiftedSelector<br/>話題轉移知識選擇"]
    H -->|"否 (0)"| J["TopicInheritedSelector<br/>話題繼承知識選擇"]
    I --> K["輸出預測結果"]
    J --> K
    K --> L["計算 Precision/Recall/F1"]
    L --> M["結束"]

    style G fill:#ffecb3
    style I fill:#c8e6c9
    style J fill:#bbdefb
```

### 2.2 GPT-2 對話生成流程

```mermaid
flowchart LR
    A["knowSelect<br/>知識選擇結果"] --> B["載入選中的知識片段"]
    B --> C["構建 GPT-2 輸入"]
    C --> D["GPT-2 模型"]
    D --> E["生成對話回復"]
    E --> F["輸出結果"]

    subgraph "GPT-2 輸入格式"
        C1["<knowledge> 知識"]
        C2["<context> 歷史對話"]
        C3["<response> 回復"]
    end

    C --> C1 & C2 & C3
```

---

## 3. 消融實驗設計

```mermaid
graph LR
    subgraph "實驗配置"
        A1["A1: 純文本基線<br/>feature_set=none"]
        A2["A2: 文本 + imp_pct<br/>feature_set=imp_pct"]
        A3["A3: 文本 + 6維特徵<br/>feature_set=all"]
    end

    subgraph "評價指標"
        M1["Precision<br/>精確率"]
        M2["Recall<br/>召回率"]
        M3["F1<br/>調和平均"]
    end

    A1 --> M1 & M2 & M3
    A2 --> M1 & M2 & M3
    A3 --> M1 & M2 & M3

    style A1 fill:#ffcdd2
    style A2 fill:#fff9c4
    style A3 fill:#c8e6c9
```

| 配置 | 描述 | 特徵 |
|------|------|------|
| A1 | 純文本基線 | 不使用中心性特徵 |
| A2 | 文本 + imp_pct | 僅使用重要性分位數 |
| A3 | 文本 + 6維特徵 | 使用全部結構特徵 |

---

## 4. 評價指標說明

| 指標 | 公式 | 說明 |
|------|------|------|
| **Precision** | TP / (TP + FP) | 預測為話題轉移中，真正是話題轉移的比例 |
| **Recall** | TP / (TP + FN) | 所有真實話題轉移中，被正確預測的比例 |
| **F1** | 2 × P × R / (P + R) | Precision 和 Recall 的調和平均數 |

---

## 5. 腳本使用指南

### 5.1 完整流程

```mermaid
sequenceDiagram
    participant U as 用戶
    participant S as 環境設置
    participant K as knowSelect
    participant D as dialogen
    participant R as 結果

    U->>S: setup_env.sh
    S-->>U: 環境準備完成
    
    U->>K: train_take_tiage.sh
    K-->>U: 模型訓練完成
    
    U->>K: infer_take_tiage.sh
    K-->>R: shift_metrics.json
    K-->>R: shift_top3.jsonl
    
    U->>D: train_dialogen_tiage.sh
    D-->>R: 生成對話模型
```

### 5.2 腳本對照表

| 腳本 | 平台 | 功能 |
|------|------|------|
| `setup_env.sh` / `.bat` | Linux/Windows | 環境初始化 |
| `train_take_tiage.sh` / `.bat` | Linux/Windows | knowSelect 訓練 |
| `infer_take_tiage.sh` / `.bat` | Linux/Windows | knowSelect 推論 |
| `ablation_take_tiage.sh` / `.bat` | Linux/Windows | 消融實驗 |
| `train_dialogen_tiage.sh` / `.bat` | Linux/Windows | GPT-2 訓練 |

### 5.3 快速開始

#### Linux / macOS

```bash
# 1. 設置環境
./scripts/setup_env.sh

# 2. 訓練話題轉移檢測模型
./scripts/train_take_tiage.sh

# 3. 執行推論並獲取評價指標
./scripts/infer_take_tiage.sh

# 4. 運行消融實驗（可選）
./scripts/ablation_take_tiage.sh

# 5. 訓練 GPT-2 對話生成模型
./scripts/train_dialogen_tiage.sh
```

#### Windows (CMD / PowerShell)

```batch
REM 1. 設置環境
scripts\setup_env.bat

REM 2. 訓練話題轉移檢測模型
scripts\train_take_tiage.bat

REM 3. 執行推論並獲取評價指標
scripts\infer_take_tiage.bat

REM 4. 運行消融實驗（可選）
scripts\ablation_take_tiage.bat

REM 5. 訓練 GPT-2 對話生成模型
scripts\train_dialogen_tiage.bat
```

> **提示**: Windows 用戶也可以使用 PowerShell 腳本 `train_take_tiage.ps1`

---

## 6. 輸出文件說明

```mermaid
graph TD
    subgraph "knowSelect 輸出"
        A["output/{name}/"]
        A --> B["model/<br/>訓練模型"]
        A --> C["ks_pred/<br/>知識選擇預測"]
        A --> D["metrics/"]
        D --> D1["shift_metrics.json<br/>Precision/Recall/F1"]
        D --> D2["shift_top3.jsonl<br/>Top-K 句子"]
        D --> D3["ablation_results.csv<br/>消融實驗"]
    end

    subgraph "dialogen 輸出"
        E["output/{name}/"]
        E --> F["gen_model/<br/>GPT-2 模型"]
        E --> G["result_gen/<br/>生成結果"]
    end
```

### 6.1 shift_metrics.json 格式

```json
{
  "9_test": {
    "run": "TAKE_tiage_all_feats",
    "dataset": "test",
    "epoch": "9",
    "precision": 75.32,
    "recall": 68.45,
    "f1": 71.72
  }
}
```

### 6.2 shift_top3.jsonl 格式

```json
{
  "run": "TAKE_tiage_all_feats",
  "dataset": "test",
  "dialog_id": "1",
  "shift_found": true,
  "shift_top3": [
    {"node_id": 42, "sentence": "話題轉移句子內容...", "centrality": 0.8523},
    {"node_id": 38, "sentence": "第二高中心性句子...", "centrality": 0.7891},
    {"node_id": 45, "sentence": "第三高中心性句子...", "centrality": 0.7234}
  ]
}
```

---

## 7. 核心模組說明

### 7.1 特徵計算 (CentralityLoader)

```python
# 6 維結構特徵計算
features = [
    imp_raw,              # 原始中心性值
    imp_pct,              # 重要性分位數 (0-1)
    imp_delta_prev,       # 與前一節點的中心性差值
    imp_delta_next,       # 與後一節點的中心性差值
    imp_z_local,          # 局部窗口內的 Z-score
    imp_minus_window_mean # 與窗口均值的差值
]
```

### 7.2 話題判別器 (TeacherTopicShiftDiscriminator)

```mermaid
graph LR
    A["上下文向量"] --> D["MLP"]
    B["知識向量"] --> D
    C["中心性特徵"] --> D
    D --> E["Softmax"]
    E --> F["0: 話題繼承"]
    E --> G["1: 話題轉移"]
```

---

## 8. 數據集結構

```
knowSelect/datasets/tiage/
├── tiage.answer      # 回答數據
├── tiage.query       # 查詢數據
├── tiage.passage     # 知識段落
├── tiage.pool        # 知識池
├── tiage.split       # 訓練/測試劃分
├── node_id.json      # 節點ID映射
└── ID_label.json     # 話題轉移標籤
```

---

## 9. 常見問題

### Q1: 如何查看評價指標？

**Linux/Mac:**
```bash
cat knowSelect/output/TAKE_tiage_all_feats/metrics/shift_metrics.json
```

**Windows:**
```batch
type knowSelect\output\TAKE_tiage_all_feats\metrics\shift_metrics.json
```

### Q2: 如何查看 Top-K 高中心性句子？

**Linux/Mac:**
```bash
head -10 knowSelect/output/TAKE_tiage_all_feats/metrics/shift_top3.jsonl
```

**Windows (PowerShell):**
```powershell
Get-Content knowSelect\output\TAKE_tiage_all_feats\metrics\shift_top3.jsonl -First 10
```

### Q3: 消融實驗結果在哪裡？

**Linux/Mac:**
```bash
cat knowSelect/output/TAKE_tiage_all_feats/metrics/ablation_results.csv
```

**Windows:**
```batch
type knowSelect\output\TAKE_tiage_all_feats\metrics\ablation_results.csv
```

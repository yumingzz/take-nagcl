# COLING2022-TAKE: 主題轉換感知的知識選擇對話生成

本專案為 COLING 2022 論文的程式碼實作：[TAKE: Topic-shift Aware Knowledge sElection for Dialogue Generation](https://aclanthology.org/2022.coling-1.21/)

![TAKE 模型架構圖](https://github.com/iie-ycx/COLING2022-TAKE/raw/main/fig/take-pic.png)

## 專案簡介

TAKE 模型透過感知對話中的主題轉換，實現更精準的知識選擇與對話生成。系統採用兩階段管線架構：

1. **知識選擇（Knowledge Selection）**：使用 BERT 編碼器與主題轉換/繼承選擇器
2. **對話生成（Dialogue Generation）**：使用 GPT-2 根據選擇的知識生成回應

## 環境需求

- Python 3.9+
- PyTorch 1.10.1
- transformers 4.15.0
- CUDA（建議使用 GPU）

## 安裝

### 使用 uv（推薦）

[uv](https://github.com/astral-sh/uv) 是一個極快的 Python 套件管理器。

```bash
# 安裝 uv（如果尚未安裝）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 建立虛擬環境並安裝相依套件
uv venv --python 3.9
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 安裝相依套件
uv pip install torch==1.10.1 transformers==4.15.0 scikit-learn nltk tqdm
```

### 使用 requirements.txt

```bash
uv venv --python 3.9
source .venv/bin/activate
uv pip install -r requirements.txt
```

## 資料集

使用 [Meng et al.](https://dl.acm.org/doi/10.1145/3404835.3462824) 預處理的 Wizard of Wikipedia 資料集。

### 下載資料集

從[這裡](https://share.weiyun.com/rpmIidMZ)下載資料集，並放置於以下目錄：

```
./knowSelect/datasets/wizard_of_wikipedia/
./dialogen/datasets/wizard_of_wikipedia/
./dialogen/datasets/wow_gpt2/
```

### 下載預訓練檢查點（可選）

從[這裡](https://share.weiyun.com/zqoSPsF7)下載預訓練檢查點：

- 知識選擇模型 → `./knowSelect/output/TAKE_WoW/model/`
- 對話生成模型 → `./dialogen/output/TAKE_WoW/gen_model/`

## 使用方式

### 快速推論（使用預訓練模型）

```bash
bash infer_bash.sh
```

### 完整訓練流程

```bash
bash train_bash.sh
```

### 分別執行各模組

#### 知識選擇模組

```bash
cd knowSelect

# 訓練
uv run python ./TAKE/Run.py --name TAKE_WoW --dataset wizard_of_wikipedia --mode train

# 推論
uv run python ./TAKE/Run.py --name TAKE_WoW --dataset wizard_of_wikipedia --mode inference
```

#### 對話生成模組

```bash
cd dialogen

# 訓練
uv run python ./TAKE/Run.py --name TAKE_WoW --dataset wizard_of_wikipedia --mode train

# 推論
uv run python ./TAKE/Run.py --name TAKE_WoW --dataset wizard_of_wikipedia --mode inference
```

### 常用參數

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--mode` | 執行模式（train/inference） | train |
| `--resume` | 從檢查點繼續訓練 | False |
| `--GPU` | GPU 裝置 ID | 0 |
| `--epoches` | 訓練輪數 | 15 |
| `--train_batch_size` | 訓練批次大小 | 2 |
| `--appoint_epoch` | 指定推論的 epoch | -1 |

## 專案結構

```
COLING2022-TAKE/
├── knowSelect/              # 知識選擇模組
│   ├── TAKE/
│   │   ├── Run.py          # 進入點
│   │   ├── Model.py        # TAKE 模型
│   │   └── ...
│   ├── datasets/           # 資料集目錄
│   └── output/             # 輸出目錄
│
├── dialogen/                # 對話生成模組
│   ├── TAKE/
│   │   ├── Run.py          # 進入點
│   │   ├── Model.py        # GPT-2 生成模型
│   │   └── ...
│   ├── datasets/           # 資料集目錄
│   └── output/             # 輸出目錄
│
├── demo/                    # 示範專案
│   └── DGCN3/              # DGCN3 專案
│
├── train_bash.sh           # 完整訓練腳本
├── infer_bash.sh           # 推論腳本
└── requirements.txt        # 相依套件
```

## 評估指標

專案支援以下評估指標：
- **BLEU**：機器翻譯評估
- **ROUGE**：摘要生成評估
- **F1**：知識選擇準確度
- **METEOR**：語義相似度

## 引用

如果您使用了本專案的程式碼，請引用我們的論文：

```bibtex
@inproceedings{yang-etal-2022-take,
    title = "{TAKE}: Topic-shift Aware Knowledge s{E}lection for Dialogue Generation",
    author = "Yang, Chenxu and Lin, Zheng and Li, Jiangnan and Meng, Fandong and Wang, Weiping and Wang, Lanrui and Zhou, Jie",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    pages = "253--265",
}
```

## 聯絡方式

如有任何問題，請聯絡 Chenxu Yang (yangchenxu@iie.ac.cn)

## 授權條款

本專案採用 MIT 授權條款。

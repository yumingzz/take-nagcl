# CLAUDE.md

本文件為 Claude Code (claude.ai/code) 在此程式碼庫中工作時提供指引。

## 專案概述

此程式庫包含 COLING 2022 論文「TAKE: Topic-shift Aware Knowledge sElection for Dialogue Generation」的實作。該模型使用兩階段管線執行知識驅動的對話生成，並考慮對話中的主題轉換。

## 環境需求

- Python 3.9
- PyTorch 1.10.1
- transformers==4.15.0
- BERT (bert-base-uncased) 用於編碼
- GPT-2 用於回應生成

## 指令

### 完整訓練流程
```bash
bash train_bash.sh
```
依序執行知識選擇訓練、推論，接著執行對話生成訓練與推論。

### 僅推論（使用預訓練檢查點）
```bash
bash infer_bash.sh
```

### 個別模組指令

知識選擇：
```bash
cd knowSelect
python ./TAKE/Run.py --name TAKE_WoW --dataset wizard_of_wikipedia --mode train
python ./TAKE/Run.py --name TAKE_WoW --dataset wizard_of_wikipedia --mode inference
```

對話生成：
```bash
cd dialogen
python ./TAKE/Run.py --name TAKE_WoW --dataset wizard_of_wikipedia --mode train
python ./TAKE/Run.py --name TAKE_WoW --dataset wizard_of_wikipedia --mode inference
```

主要參數：
- `--mode`：`train` 或 `inference`
- `--resume`：從上次檢查點繼續訓練
- `--GPU`：GPU 裝置 ID（預設：0）
- `--appoint_epoch`：對指定 epoch 檢查點執行推論

## 架構

系統為兩階段管線：

### 第一階段：知識選擇（`knowSelect/`）
`knowSelect/TAKE/Model.py` 中的 TAKE 模型使用以下元件選擇相關知識：
- **TransformerSeqEncoder**：基於 BERT 的編碼器，用於編碼上下文與知識候選項
- **DivideAndSelfAttention**：分離上下文與知識注意力
- **TopicShiftedSelector**：處理主題轉換 - 根據當前上下文選擇知識
- **TopicInheritedSelector**：處理主題繼承 - 根據對話歷史選擇知識
- **Teacher/Student TopicShiftDiscriminator**：主題轉換偵測的知識蒸餾

模型輸出知識選擇預測至 `knowSelect/output/TAKE_WoW/ks_pred/`。

### 第二階段：對話生成（`dialogen/`）
使用 GPT-2（`dialogen/TAKE/Model.py`）生成回應：
- 接收第一階段選擇的知識
- 使用 `GPT2Summ` 封裝器進行知識條件化生成
- 輸出至 `dialogen/output/TAKE_WoW/result_gen/`

### 資料流程
1. 執行知識選擇（knowSelect）→ 產生 `ks_pred/` 預測結果
2. 複製預測結果：`cp -r ./knowSelect/output/TAKE_WoW/ks_pred ./dialogen/output/TAKE_WoW`
3. 執行對話生成（dialogen）使用知識預測結果

## 關鍵檔案

- `knowSelect/TAKE/Run.py`：知識選擇進入點
- `dialogen/TAKE/Run.py`：對話生成進入點
- `*/TAKE/Model.py`：模型架構
- `*/TAKE/CumulativeTrainer.py`：含梯度累積的訓練迴圈
- `*/dataset/Utils_TAKE.py`：資料載入工具（BERT/GPT2 分詞）
- `*/evaluation/`：評估指標（BLEU、ROUGE、F1、METEOR）

## 資料集

使用 Wizard of Wikipedia 資料集。請將資料檔案放置於：
- `knowSelect/datasets/wizard_of_wikipedia/`
- `dialogen/datasets/wizard_of_wikipedia/`
- `dialogen/datasets/wow_gpt2/`

檢查點下載：https://share.weiyun.com/zqoSPsF7
資料集下載：https://share.weiyun.com/rpmIidMZ

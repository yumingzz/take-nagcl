# TAKE + DGCN3 训练指南

> 本文档说明如何运行 TAKE 模型训练、查看日志、以及常见问题排查。

---

## 目录

1. [环境准备](#一环境准备)
2. [数据准备](#二数据准备)
3. [训练命令](#三训练命令)
4. [日志系统](#四日志系统)
5. [推论与评估](#五推论与评估)
6. [常见问题](#六常见问题)

---

## 一、环境准备

### 1.1 Python 环境

本项目使用 Python 3.9，建议使用虚拟环境：

```bash
# 使用 uv (推荐)
cd /path/to/COLING2022-TAKE
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 或使用 pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1.2 依赖检查

```bash
# 检查核心依赖
python -c "import torch; import transformers; import nltk; print('All OK')"

# 检查 PyTorch 版本和设备
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### 1.3 目录结构

确保以下目录存在：

```
COLING2022-TAKE/
├── .venv/                          # 虚拟环境
├── demo/
│   ├── DGCN3/
│   │   ├── Centrality/             # DGCN3 预测输出
│   │   │   └── alpha_1.5/
│   │   │       └── tiage_0~9.csv
│   │   └── datasets/raw_data/tiage/
│   └── tiage-1/
│       └── outputs_nodes/
│           └── tiage_anno_nodes_all.csv
├── knowSelect/
│   ├── datasets/tiage/             # TAKE 数据集
│   ├── output/TAKE_tiage/          # 训练输出
│   │   ├── model/
│   │   ├── ks_pred/
│   │   └── logs/
│   └── TAKE/                       # 模型代码
└── docs/
```

---

## 二、数据准备

### 2.1 生成 DGCN3 预测

如果 `demo/DGCN3/Centrality/` 不存在，需要先运行 DGCN3：

```bash
cd demo/DGCN3
python main.py --dataset_name tiage
```

输出文件：`Centrality/alpha_1.5/tiage_{0-9}.csv`

### 2.2 生成 TAKE 数据集

如果 `knowSelect/datasets/tiage/` 不存在：

```bash
cd demo/tiage-1
python export_take_dataset.py --out ../../knowSelect/datasets/tiage
```

输出文件：
- `tiage.query`, `tiage.answer`, `tiage.pool`, `tiage.passage`, `tiage.split`
- `ID_label.json`, `node_id.json`

### 2.3 创建输出目录

```bash
mkdir -p knowSelect/output/TAKE_tiage/{model,ks_pred,logs}

# 初始化 checkpoints.json
echo '{"time": []}' > knowSelect/output/TAKE_tiage/model/checkpoints.json
```

---

## 三、训练命令

### 3.1 基本训练命令

```bash
cd knowSelect

# 使用虚拟环境的 Python
.venv/bin/python -u ./TAKE/Run.py \
    --name TAKE_tiage \
    --dataset tiage \
    --mode train \
    --use_centrality \
    --centrality_alpha 1.5
```

**参数说明**:

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--name` | 实验名称 | 必填 |
| `--dataset` | 数据集名称 | 必填 |
| `--mode` | `train` 或 `inference` | 必填 |
| `--use_centrality` | 启用中心性特征 | False |
| `--centrality_alpha` | SIR alpha 参数 | 1.5 |
| `--centrality_feature_set` | 特征集 (`none`/`imp_pct`/`all`) | `all` |
| `--GPU` | GPU 设备 ID | 0 |
| `--epoches` | 训练轮数 | 10 |
| `--train_batch_size` | 批次大小 | 2 |

### 3.2 后台运行

```bash
# 后台运行并将输出重定向到文件
nohup .venv/bin/python -u ./TAKE/Run.py \
    --name TAKE_tiage \
    --dataset tiage \
    --mode train \
    --use_centrality \
    --centrality_alpha 1.5 \
    > train.log 2>&1 &

# 查看进程
ps aux | grep Run.py
```

### 3.3 消融实验

```bash
# A1: 纯文本基线 (不使用中心性)
python -u ./TAKE/Run.py --name TAKE_tiage_text_only --dataset tiage --mode train

# A2: 仅使用 imp_pct 特征
python -u ./TAKE/Run.py --name TAKE_tiage_imp_pct --dataset tiage --mode train \
    --use_centrality --centrality_feature_set imp_pct

# A3: 使用全部结构特征 (默认)
python -u ./TAKE/Run.py --name TAKE_tiage_all_feats --dataset tiage --mode train \
    --use_centrality --centrality_feature_set all
```

### 3.4 从检查点恢复

```bash
python -u ./TAKE/Run.py \
    --name TAKE_tiage \
    --dataset tiage \
    --mode train \
    --use_centrality \
    --resume  # 添加此参数
```

---

## 四、日志系统

### 4.1 日志文件位置

训练日志自动保存到：
```
knowSelect/output/{name}/logs/train_{timestamp}.log
```

例如：`output/TAKE_tiage/logs/train_20260103_165408.log`

### 4.2 日志格式

```
[2026-01-03 16:54:08] === Training session started: TAKE_tiage ===
[2026-01-03 16:54:08] Log file: output/TAKE_tiage/logs/train_20260103_165408.log
[2026-01-03 16:54:08] Using CPU
[2026-01-03 16:54:08] ============================================================
[2026-01-03 16:54:08] Starting Epoch 0 | Total batches: 150 | Batch size: 2
[2026-01-03 16:54:08] ============================================================
[2026-01-03 16:54:47] [Epoch 0] Batch 1/150 (0.7%) | loss_ks: 0.0000 | loss_distill: 0.9490 | loss_ID: 0.4478 | ks_acc: 1.0000 | ID_acc: 0.5000 | elapsed: 38.8s | LR: 0.00e+00
```

### 4.3 实时查看日志

```bash
# 方法1: tail -f (实时跟踪)
tail -f output/TAKE_tiage/logs/train_*.log

# 方法2: 查看最新日志
ls -lt output/TAKE_tiage/logs/ | head -5
cat output/TAKE_tiage/logs/train_20260103_165408.log

# 方法3: 查看控制台输出 (如果使用 nohup)
tail -f train.log
```

### 4.4 日志指标说明

| 指标 | 说明 | 理想趋势 |
|------|------|----------|
| `loss_ks` | 知识选择损失 | 下降 |
| `loss_distill` | 蒸馏损失 | 下降 |
| `loss_ID` | 话题判别损失 | 下降 |
| `ks_acc` | 知识选择准确率 | 上升 |
| `ID_acc` | 话题判别准确率 | 上升 |
| `elapsed` | 已用时间 (秒) | - |
| `LR` | 学习率 | 先升后降 (warmup) |

### 4.5 调整日志频率

在 `CumulativeTrainer.py` 中修改：

```python
# 第199行左右
log_interval = 10  # 改为更小的值如 5 或 1
```

---

## 五、推论与评估

### 5.1 运行推论

```bash
python -u ./TAKE/Run.py \
    --name TAKE_tiage \
    --dataset tiage \
    --mode inference \
    --use_centrality \
    --centrality_alpha 1.5
```

### 5.2 推论输出

输出文件位于：`output/TAKE_tiage/ks_pred/`

```
ks_pred/
├── 0_test.json      # Epoch 0 的测试集预测
├── 1_test.json      # Epoch 1 的测试集预测
└── ...
```

### 5.3 评估指标

推论完成后，终端会输出：

- `final_ks_acc`: 最终知识选择准确率
- `shifted_ks_acc`: 话题转移时的准确率
- `inherited_ks_acc`: 话题继承时的准确率
- `ID_acc`: 话题转移判别准确率

---

## 六、常见问题

### 6.1 CUDA 不可用

**问题**: `AssertionError: Torch not compiled with CUDA enabled`

**解决**: 代码已适配 CPU 模式，会自动回退。CPU 训练较慢，每 batch 约 40 秒。

### 6.2 目录不存在

**问题**: `FileNotFoundError: ... checkpoints.json`

**解决**:
```bash
mkdir -p output/TAKE_tiage/{model,ks_pred,logs}
echo '{"time": []}' > output/TAKE_tiage/model/checkpoints.json
```

### 6.3 内存不足

**问题**: `RuntimeError: out of memory`

**解决**: 减小批次大小
```bash
--train_batch_size 1
```

### 6.4 日志不更新

**问题**: 日志文件没有实时更新

**解决**:
1. 确保使用 `-u` 参数运行 Python (禁用缓冲)
2. 代码已使用 `FlushFileHandler` 即时刷新

### 6.5 中心性特征加载失败

**问题**: 找不到中心性预测文件

**解决**: 确保 DGCN3 预测已生成
```bash
ls demo/DGCN3/Centrality/alpha_1.5/
# 应该有 tiage_0.csv ~ tiage_9.csv
```

### 6.6 终止训练

```bash
# 查找进程
ps aux | grep Run.py

# 终止进程
pkill -f "Run.py --name TAKE_tiage"

# 或指定 PID
kill -9 <PID>
```

---

## 附录: 完整训练脚本

```bash
#!/bin/bash
# train_take_tiage.sh

set -e

PROJECT_ROOT="/path/to/COLING2022-TAKE"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

cd "$PROJECT_ROOT/knowSelect"

# 创建输出目录
mkdir -p output/TAKE_tiage/{model,ks_pred,logs}
echo '{"time": []}' > output/TAKE_tiage/model/checkpoints.json

# 开始训练
echo "[$(date)] Starting training..."
$VENV_PYTHON -u ./TAKE/Run.py \
    --name TAKE_tiage \
    --dataset tiage \
    --mode train \
    --use_centrality \
    --centrality_alpha 1.5 \
    --epoches 10

echo "[$(date)] Training completed!"

# 运行推论
echo "[$(date)] Starting inference..."
$VENV_PYTHON -u ./TAKE/Run.py \
    --name TAKE_tiage \
    --dataset tiage \
    --mode inference \
    --use_centrality \
    --centrality_alpha 1.5

echo "[$(date)] All done!"
```

---

## 快速参考

```bash
# 训练
cd knowSelect
.venv/bin/python -u ./TAKE/Run.py --name TAKE_tiage --dataset tiage --mode train --use_centrality

# 查看日志
tail -f output/TAKE_tiage/logs/train_*.log

# 终止训练
pkill -f "Run.py --name TAKE_tiage"

# 推论
.venv/bin/python -u ./TAKE/Run.py --name TAKE_tiage --dataset tiage --mode inference --use_centrality
```

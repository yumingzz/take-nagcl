# TAKE + DGCN3 算法架构文档

> 本文档说明 TAKE 模型与 DGCN3 中心性预测整合后的算法结构、输入输出，以及与 `IMPLEMENTATION_PLAN.md` 的对应关系。

---

## 目录

1. [整体架构概览](#一整体架构概览)
2. [数据流程](#二数据流程)
3. [模块详解](#三模块详解)
4. [输入输出规格](#四输入输出规格)
5. [与实施计划对应关系](#五与实施计划对应关系)

---

## 一、整体架构概览

### 1.1 系统架构图

```mermaid
flowchart TB
    subgraph DataSource["数据源"]
        tiage1["tiage-1<br/>- 对话文本<br/>- 网络拓扑"]
    end

    subgraph DGCN3["DGCN3 中心性预测"]
        gat["GAT 编码器"]
        pred["重要性预测"]
        gat --> pred
    end

    subgraph CentralityLoader["CentralityLoader 特征加载"]
        feat6["6维结构特征"]
        louvain["Louvain 社团检测"]
    end

    subgraph TAKEModel["TAKE 模型"]
        subgraph Encoders["编码器层"]
            bert["TransformerSeqEncoder<br/>(BERT)<br/>- 文本编码<br/>- 768维向量"]
            centEnc["CentralityCommunityEncoder<br/>- 中心性 MLP<br/>- 社团 Embedding<br/>- 融合层 128维"]
            disc["TopicShiftDiscriminator<br/>- Teacher (训练)<br/>- Student (推论)<br/>- 话题转移判别"]
        end

        subgraph KSLayer["Knowledge Selection Layer"]
            shifted["TopicShiftedSelector<br/>(话题转移时选择)"]
            inherited["TopicInheritedSelector<br/>(话题继承时选择)"]
        end

        bert --> KSLayer
        centEnc --> disc
        disc --> KSLayer
    end

    tiage1 --> DGCN3
    DGCN3 --> CentralityLoader
    CentralityLoader --> TAKEModel
    tiage1 --> TAKEModel
```

### 1.2 核心组件

| 组件 | 文件路径 | 功能 |
|------|----------|------|
| **DGCN3** | `demo/DGCN3/` | 基于 GAT 的节点重要性预测 |
| **CentralityLoader** | `knowSelect/TAKE/CentralityLoader.py` | 加载中心性预测，计算6维结构特征 |
| **CentralityEncoder** | `knowSelect/TAKE/CentralityEncoder.py` | 将结构特征编码为128维向量 |
| **TAKE Model** | `knowSelect/TAKE/Model.py` | 话题感知的知识选择模型 |
| **Dataset** | `knowSelect/TAKE/Dataset.py` | 数据加载与批处理 |

---

## 二、数据流程

### 2.1 训练数据流

```mermaid
flowchart TB
    subgraph Input["输入数据"]
        csv["tiage_anno_nodes_all.csv"]
        dgcn_out["DGCN3 Centrality/alpha_1.5/<br/>tiage_0.csv ~ tiage_9.csv"]
    end

    subgraph DataPrep["数据准备"]
        export["export_take_dataset.py"]
        csv --> export
    end

    subgraph TiageDataset["knowSelect/datasets/tiage/"]
        query["tiage.query"]
        answer["tiage.answer"]
        pool["tiage.pool"]
        passage["tiage.passage"]
        nodeid["node_id.json"]
        idlabel["ID_label.json"]
    end

    export --> TiageDataset

    subgraph DatasetPy["Dataset.py"]
        load["加载对话数据"]
        episode["构建 episode"]
        link["关联 node_id"]
        load --> episode --> link
    end

    TiageDataset --> DatasetPy

    subgraph CentLoader["CentralityLoader.py"]
        input_node["输入: node_id"]
        output_feat["输出:<br/>- centrality_features [batch, 6]<br/>- community_ids [batch]"]
        input_node --> output_feat
    end

    dgcn_out --> CentLoader
    DatasetPy --> CentLoader

    subgraph Model["TAKE Model"]
        fusion["融合: BERT文本编码 + 结构特征编码"]
        output_model["输出: 知识选择预测 + 话题转移判别"]
        fusion --> output_model
    end

    CentLoader --> Model
```

### 2.2 批次数据结构

```python
# collate_fn 返回的 batch 数据结构
batch = {
    'episode_id': [batch],                                    # 对话ID
    'context': [batch, max_episode_length, context_len],      # 上下文编码
    'response': [batch, max_episode_length, max_dec_length],  # 回复编码
    'knowledge_pool': [batch, max_episode_length, max_knowledge_pool, knowledge_sentence_len],
    'knowledge_piece_mask': [batch, max_episode_length, max_knowledge_pool],
    'knowledge_label': [batch, max_episode_length],           # 知识标签
    'Initiative_label': [batch, max_episode_length],          # 话题转移标签 (-1/0/1)
    'episode_mask': [batch, max_episode_length],              # 有效位置掩码
    'node_ids': [batch, max_episode_length]                   # 节点ID (新增)
}
```

---

## 三、模块详解

### 3.1 CentralityLoader - 结构特征加载器

**位置**: `knowSelect/TAKE/CentralityLoader.py`

**功能**: 加载 DGCN3 预测的中心性值，计算6维结构特征

**6维结构特征**:

| 特征名 | 说明 | 计算方式 |
|--------|------|----------|
| `imp_raw` | 原始重要性分数 | DGCN3 预测值 |
| `imp_pct` | 重要性分位数 | 对话内排名 / 对话长度 |
| `imp_delta_prev` | 与上一句的变化 | imp_pct[t] - imp_pct[t-1] |
| `imp_delta_next` | 与下一句的变化 | imp_pct[t+1] - imp_pct[t] |
| `imp_z_local` | 局部窗口 z-score | (x - mean) / (std + 1e-6) |
| `imp_minus_window_mean` | 与窗口均值差 | x - window_mean |

**接口**:

```python
class CentralityCommunityLoader:
    def __init__(
        self,
        dgcn_predictions_dir: str,   # DGCN3 预测目录
        edge_lists_dir: str,          # 边列表目录 (用于社团检测)
        node_mapping_csv: str,        # 节点映射文件
        alpha: float = 1.5,           # SIR alpha 参数
        num_slices: int = 10,         # 时间片数量
        feature_set: str = "all",     # 特征集 (none/imp_pct/all)
        window_size: int = 2          # 局部窗口大小
    )

    def get_batch_features(
        self,
        node_ids: torch.Tensor,       # [batch]
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 返回: (centrality_features [batch, 6], community_ids [batch])
```

### 3.2 CentralityEncoder - 结构特征编码器

**位置**: `knowSelect/TAKE/CentralityEncoder.py`

**功能**: 将6维结构特征 + 社团ID 编码为128维密集向量

**网络结构**:

```mermaid
flowchart TB
    subgraph Input["输入"]
        cent_feat["centrality_features<br/>[batch, 6]"]
        comm_ids["community_ids<br/>[batch]"]
    end

    subgraph CentralityBranch["中心性分支"]
        cent_mlp["centrality_mlp<br/>6 → 64 → 64"]
    end

    subgraph CommunityBranch["社团分支"]
        comm_embed["community_embedding<br/>num_comm → 64"]
    end

    subgraph Fusion["融合层"]
        concat["concat<br/>64 + 64 = 128"]
        fusion_layer["fusion<br/>128 → 128 → 128"]
        ln["layer_norm"]
    end

    subgraph Output["输出"]
        output["output<br/>[batch, 128]"]
    end

    cent_feat --> cent_mlp
    comm_ids --> comm_embed
    cent_mlp --> concat
    comm_embed --> concat
    concat --> fusion_layer
    fusion_layer --> ln
    ln --> output
```

### 3.3 TopicShiftDiscriminator - 话题转移判别器

**位置**: `knowSelect/TAKE/Model.py`

**修改**: 融合层输入维度增加128维 (中心性特征)

**Teacher Discriminator** (训练时使用):
- 输入: `5 * hidden_size + 128` = `5 * 768 + 128 = 3968`
- 使用 gold knowledge 辅助判别

**Student Discriminator** (推论时使用):
- 输入: `4 * hidden_size + 128` = `4 * 768 + 128 = 3200`
- 不使用 gold knowledge

### 3.4 TAKE 主模型

**位置**: `knowSelect/TAKE/Model.py`

**知识选择流程**:

```mermaid
flowchart TB
    step1["1. BERT 编码上下文和知识候选"]
    step2["2. DivideAndSelfAttention 分离注意力"]
    step3["3. TopicShiftDiscriminator<br/>判别话题是否转移"]

    step1 --> step2 --> step3

    step3 --> shift_prob["shift_prob"]
    step3 --> inherit_prob["inherit_prob"]

    shift_prob --> step4a["4a. TopicShiftedSelector<br/>(基于当前上下文)"]
    inherit_prob --> step4b["4b. TopicInheritedSelector<br/>(基于对话历史)"]

    step4a --> step5["5. 加权融合得到最终知识选择分布"]
    step4b --> step5
```

### 3.5 完整模型数据流

```mermaid
flowchart LR
    subgraph Input["输入"]
        context["context<br/>[batch, seq_len]"]
        knowledge["knowledge_pool<br/>[batch, K, seq_len]"]
        node_ids["node_ids<br/>[batch]"]
    end

    subgraph BERT["BERT Encoder"]
        ctx_enc["context_encoding<br/>[batch, 768]"]
        know_enc["knowledge_encoding<br/>[batch, K, 768]"]
    end

    subgraph Centrality["Centrality Module"]
        loader["CentralityLoader<br/>6维特征 + 社团ID"]
        encoder["CentralityEncoder<br/>[batch, 128]"]
        loader --> encoder
    end

    subgraph Discriminator["Topic Shift Discriminator"]
        teacher["Teacher<br/>(5×768 + 128)"]
        student["Student<br/>(4×768 + 128)"]
    end

    subgraph Selector["Knowledge Selector"]
        shifted["TopicShiftedSelector"]
        inherited["TopicInheritedSelector"]
        final["Final Distribution"]
    end

    context --> ctx_enc
    knowledge --> know_enc
    node_ids --> loader

    ctx_enc --> Discriminator
    know_enc --> Discriminator
    encoder --> Discriminator

    Discriminator --> shifted
    Discriminator --> inherited
    shifted --> final
    inherited --> final
```

---

## 四、输入输出规格

### 4.1 DGCN3 输出

**路径**: `demo/DGCN3/Centrality/alpha_1.5/tiage_{0-9}.csv`

**格式** (无表头):
```csv
node_id,centrality
0,0.8234
741,0.5621
...
```

### 4.2 tiage 数据集文件

**路径**: `knowSelect/datasets/tiage/`

| 文件 | 格式 | 说明 |
|------|------|------|
| `tiage.query` | 文本 | 每行一个查询 |
| `tiage.answer` | TSV | `history_ids\tcurrent_id\tknowledge_ids\tresponse` |
| `tiage.pool` | 每行ID列表 | 知识候选池 |
| `tiage.passage` | 文本 | 每行一个知识片段 |
| `tiage.split` | TSV | `query_id\ttrain/test` |
| `ID_label.json` | JSON | `{query_id: shift_label}` |
| `node_id.json` | JSON | `{query_id: node_id}` |

### 4.3 训练输出

**路径**: `knowSelect/output/TAKE_tiage/`

```mermaid
flowchart TB
    subgraph OutputDir["output/TAKE_tiage/"]
        subgraph ModelDir["model/"]
            ckpt["checkpoints.json"]
            pkl["{epoch}.pkl"]
        end
        subgraph KSPred["ks_pred/"]
            pred["{epoch}_{split}.json"]
        end
        subgraph Logs["logs/"]
            log["train_{timestamp}.log"]
        end
    end
```

### 4.4 训练指标

| 指标 | 说明 |
|------|------|
| `loss_ks` | 知识选择损失 |
| `loss_distill` | 知识蒸馏损失 (Teacher → Student) |
| `loss_ID` | 话题转移判别损失 |
| `ks_acc` | 知识选择准确率 |
| `ID_acc` | 话题转移判别准确率 |

---

## 五、与实施计划对应关系

### 5.1 任务对应表

| IMPLEMENTATION_PLAN 任务 | 当前实现状态 | 对应文件 |
|--------------------------|--------------|----------|
| **任务A**: DGCN3 预测保存 | ✅ 已完成 | `demo/DGCN3/Centrality/` |
| **任务B**: tiage → TAKE 数据准备 | ✅ 已完成 | `knowSelect/datasets/tiage/` |
| **任务C**: 中心性特征整合 | ✅ 已完成 | `CentralityLoader.py`, `CentralityEncoder.py` |
| **任务C**: 话题判别器融合 | ✅ 已完成 | `Model.py` (修改 Discriminator) |
| **任务C**: Dataset 添加 node_id | ✅ 已完成 | `Dataset.py` |
| **任务C**: Run.py 参数扩展 | ✅ 已完成 | `Run.py` |
| **日志系统** | ✅ 已完成 | `CumulativeTrainer.py` |

### 5.2 文件修改清单

```mermaid
flowchart LR
    subgraph NewFiles["新建文件"]
        cl["CentralityLoader.py<br/>中心性/社团特征加载"]
        ce["CentralityEncoder.py<br/>结构特征编码网络"]
    end

    subgraph ModifiedFiles["修改文件"]
        model["Model.py<br/>判别器融合中心性特征"]
        dataset["Dataset.py<br/>添加 node_id 支持"]
        run["Run.py<br/>新增命令行参数"]
        trainer["CumulativeTrainer.py<br/>添加时间戳日志系统"]
    end
```

### 5.3 消融实验配置

根据 `IMPLEMENTATION_PLAN.md` 第5.1.3节:

| 配置 | `--centrality_feature_set` | 说明 |
|------|---------------------------|------|
| A1 | 不使用 `--use_centrality` | 纯文本基线 |
| A2 | `imp_pct` | 仅使用重要性分位数 |
| A3 | `all` | 使用全部6维结构特征 |

---

## 附录: 关键代码位置

| 功能 | 文件 | 行号 (约) |
|------|------|-----------|
| TAKE 主类初始化 | `Model.py` | 267+ |
| 混合知识选择层 | `Model.py` | 300-400 |
| 中心性特征注入 | `Model.py` | 330-350 |
| Teacher Discriminator | `Model.py` | 165-206 |
| Student Discriminator | `Model.py` | 210-249 |
| Dataset collate_fn | `Dataset.py` | 198-210 |
| 训练日志输出 | `CumulativeTrainer.py` | 198-230 |

#!/usr/bin/env python3
"""
tiage → TAKE 数据格式转换脚本

将 tiage_anno_nodes_all.csv 转换为 TAKE 模型所需的数据格式：
- tiage.answer
- tiage.query
- tiage.pool
- tiage.passage
- tiage.split
- ID_label.json
- node_id.json
"""

import os
import json
import argparse
import pandas as pd
from typing import Dict, List, Tuple


def load_tiage_nodes(csv_path: str) -> pd.DataFrame:
    """加载 tiage 节点数据"""
    df = pd.read_csv(csv_path)
    # 确保必要的列存在
    required_cols = ['node_id', 'split', 'dialog_id', 'turn_id', 'text', 'shift_label']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要列: {missing}")
    return df


def assign_slice_id(df: pd.DataFrame, num_slices: int = 10) -> pd.DataFrame:
    """分配时间片 ID"""
    lens = df.groupby('dialog_id')['turn_id'].max() + 1
    df = df.copy()
    df['dialog_len'] = df['dialog_id'].map(lens).astype(int)
    df['slice_id'] = (num_slices * df['turn_id'] / df['dialog_len']).astype(int)
    df['slice_id'] = df['slice_id'].clip(0, num_slices - 1)
    return df


def generate_query_id(row) -> str:
    """生成 query_id: dialog_id_turn_id"""
    return f"{row['dialog_id']}_{row['turn_id']}"


def resolve_numeric_series(
    df: pd.DataFrame,
    candidates: List[str],
    default_value: float = 0.0
) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors='coerce').fillna(default_value).astype(float)
    return pd.Series([default_value] * len(df), index=df.index, dtype=float)


def load_ngacl_centrality(centrality_dir: str, train_slices: List[int]) -> Dict[int, float]:
    records = []
    for sid in train_slices:
        csv_path = os.path.join(centrality_dir, f"tiage_{sid}.csv")
        if not os.path.exists(csv_path):
            continue

        tmp = pd.read_csv(csv_path, header=None)
        if tmp.shape[1] < 2:
            continue

        tmp = tmp.iloc[:, :2].copy()
        tmp.columns = ['node_id', 'c_global']
        tmp['node_id'] = pd.to_numeric(tmp['node_id'], errors='coerce')
        tmp['c_global'] = pd.to_numeric(tmp['c_global'], errors='coerce')
        tmp = tmp.dropna(subset=['node_id', 'c_global'])
        if not tmp.empty:
            records.append(tmp)

    if not records:
        return {}

    merged = pd.concat(records, ignore_index=True)
    merged['node_id'] = merged['node_id'].astype(int)
    score_map = merged.groupby('node_id', as_index=True)['c_global'].mean().to_dict()
    return {int(k): float(v) for k, v in score_map.items()}


# def export_take_dataset(
#     df: pd.DataFrame,
#     output_dir: str,
#     train_slices: List[int] = None,
#     test_slices: List[int] = None
# ) -> None:
def export_take_dataset(
    df: pd.DataFrame,
    output_dir: str,
    train_slices: List[int] = None,
    dev_slices: List[int] = None,
    test_slices: List[int] = None
) -> None:
    """导出 TAKE 格式数据集"""

    # if train_slices is None:
    #     train_slices = list(range(7))  # slice 0-6
    # if test_slices is None:
    #     test_slices = [7, 8, 9]  # slice 7-9
    if train_slices is None:
        train_slices = list(range(7))  # slice 0-6
    if dev_slices is None:
        dev_slices = [7]               # slice 7
    if test_slices is None:
        test_slices = [8, 9]           # slice 8-9

    os.makedirs(output_dir, exist_ok=True)

    # 分配 slice_id
    df = assign_slice_id(df)

    # ==============================
    # 修正 dialog_id 使其全局唯一
    # ==============================

    df = df.copy()

    # 统计每个 split 的 dialog 数量
    train_dialogs = df[df['split'] == 'train']['dialog_id'].unique()
    dev_dialogs = df[df['split'] == 'dev']['dialog_id'].unique()
    test_dialogs = df[df['split'] == 'test']['dialog_id'].unique()

    train_dialog_count = len(train_dialogs)
    dev_dialog_count = len(dev_dialogs)

    # 构造映射表
    train_map = {d: i for i, d in enumerate(sorted(train_dialogs))}
    dev_map = {d: i + train_dialog_count for i, d in enumerate(sorted(dev_dialogs))}
    test_map = {d: i + train_dialog_count + dev_dialog_count for i, d in enumerate(sorted(test_dialogs))}

    def remap_dialog_id(row):
        if row['split'] == 'train':
            return train_map[row['dialog_id']]
        elif row['split'] == 'dev':
            return dev_map[row['dialog_id']]
        elif row['split'] == 'test':
            return test_map[row['dialog_id']]
        else:
            return row['dialog_id']

    df['dialog_id'] = df.apply(remap_dialog_id, axis=1)




    # 生成 query_id
    df['query_id'] = df.apply(generate_query_id, axis=1)

    # 按 dialog_id 和 turn_id 排序
    df = df.sort_values(['dialog_id', 'turn_id']).reset_index(drop=True)

    # 1. 生成 tiage.query (query_id\tquery_content)
    query_path = os.path.join(output_dir, 'tiage.query')
    with open(query_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            f.write(f"{row['query_id']}\t{row['text']}\n")
    print(f"[OK] Generated {query_path}")

    # 2. 生成 tiage.passage (passage_id\tpassage_content)
    # 最小版本：每个节点的 passage 就是自己
    passage_path = os.path.join(output_dir, 'tiage.passage')
    with open(passage_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            f.write(f"{row['query_id']}\t{row['text']}\n")
    print(f"[OK] Generated {passage_path}")

    # 3. 生成 tiage.pool (query_id Q0 passage_id rank score model_name)
    # 最小版本：每个 query 的 pool 只包含自己
    # C_transfer(v) = alpha * C_global(v) + beta * P(v)
    # - C_global(v): NGACL ?????? demo/DGCN3/Centrality/alpha_1.5/tiage_0~5.csv?
    # - P(v): participation
    alpha = 1.0
    beta = 1.0
    # top_k_pool = 128
    top_k_pool = 64

    centrality_dir = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'DGCN3', 'Centrality', 'alpha_1.5')
    )
    ngacl_map = load_ngacl_centrality(centrality_dir, train_slices=list(range(6)))

    df['c_global'] = df['node_id'].map(ngacl_map).fillna(0.0).astype(float)
    df['participation'] = resolve_numeric_series(
        df,
        candidates=['participation', 'participation_coeff', 'participation_coefficient', 'P'],
        default_value=0.0,
    )
    df['c_transfer'] = alpha * df['c_global'] + beta * df['participation']

    # pool_path = os.path.join(output_dir, 'tiage.pool')
    # with open(pool_path, 'w', encoding='utf-8') as f:
    #     for _, row in df.iterrows():
    #         f.write(f"{row['query_id']} Q0 {row['query_id']} 1 1.0 tiage\n")
    # print(f"[OK] Generated {pool_path}")

    # tiage_0~5
    train_pool_node_ids = set(ngacl_map.keys())
    pool_candidates = df[df['node_id'].isin(train_pool_node_ids)].copy()

    # centrality
    if pool_candidates.empty:
        pool_candidates = df.copy()

    pool_candidates = pool_candidates.sort_values('c_transfer', ascending=False)
    if top_k_pool > 0:
        # Reserve one slot for self knowledge so total pool size equals top_k_pool.
        pool_candidates = pool_candidates.head(max(top_k_pool - 1, 0))

    pool_path = os.path.join(output_dir, 'tiage.pool')
    with open(pool_path, 'w', encoding='utf-8') as f:
        for _, qrow in df.iterrows():
            query_id = qrow['query_id']

            # Ensure gold knowledge is always in pool for TAKE loader check.
            self_score = float(qrow['c_transfer'])
            f.write(
                f"{query_id} Q0 {query_id} 1 {self_score:.6f} "
                f"tiage_transfer_a{alpha}_b{beta}\n"
            )

            rank = 2
            for cand in pool_candidates.itertuples(index=False):
                if cand.query_id == query_id:
                    continue
                if top_k_pool > 0 and rank > top_k_pool:
                    break
                f.write(
                    f"{query_id} Q0 {cand.query_id} {rank} {float(cand.c_transfer):.6f} "
                    f"tiage_transfer_a{alpha}_b{beta}\n"
                )
                rank += 1
    print(f"[OK] Generated {pool_path}")

    # 4. 生成 tiage.answer (prev_ids\tcurrent_id\tpassage_ids\tresponse)
    # prev_ids: 同一对话内的历史 query_id（分号分隔）
    # passage_ids: 当前使用的知识（分号分隔）
    answer_path = os.path.join(output_dir, 'tiage.answer')
    with open(answer_path, 'w', encoding='utf-8') as f:
        for dialog_id, group in df.groupby('dialog_id'):
            group = group.sort_values('turn_id')
            prev_ids = []
            for _, row in group.iterrows():
                prev_str = ';'.join(prev_ids) if prev_ids else ''
                # 最小版本：passage_id 就是当前 query_id
                f.write(f"{prev_str}\t{row['query_id']}\t{row['query_id']}\t{row['text']}\n")
                prev_ids.append(row['query_id'])
    print(f"[OK] Generated {answer_path}")

    # 5. 生成 tiage.split (query_id\ttrain/dev/test)
    split_path = os.path.join(output_dir, 'tiage.split')
    with open(split_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            split_label = row['split']  # 直接用原 CSV 的 split
            if split_label not in ['train', 'dev', 'test']:
                continue
            f.write(f"{row['query_id']}\t{split_label}\n")
    print(f"[OK] Generated {split_path}")
    
    # 6. 生成 ID_label.json (话题转移标签数组)
    # 按 query_id 顺序排列，-1 -> -1 (对话开头), 0 -> 0 (非转移), 1 -> 1 (转移)
    id_labels = []
    for _, row in df.iterrows():
        label = row['shift_label']
        if pd.isna(label):
            id_labels.append(-1)
        else:
            id_labels.append(int(label))

    id_label_path = os.path.join(output_dir, 'ID_label.json')
    with open(id_label_path, 'w', encoding='utf-8') as f:
        json.dump(id_labels, f)
    print(f"[OK] Generated {id_label_path}")

    # 7. 生成 node_id.json (query_id -> node_id 映射)
    node_id_map = {}
    for _, row in df.iterrows():
        node_id_map[row['query_id']] = int(row['node_id'])

    node_id_path = os.path.join(output_dir, 'node_id.json')
    with open(node_id_path, 'w', encoding='utf-8') as f:
        json.dump(node_id_map, f, indent=2)
    print(f"[OK] Generated {node_id_path}")

    # 打印统计信息
    # train_count = len(df[df['slice_id'].isin(train_slices)])
    # test_count = len(df[df['slice_id'].isin(test_slices)])
    train_count = len(df[df['slice_id'].isin(train_slices)])
    dev_count = len(df[df['slice_id'].isin(dev_slices)])
    test_count = len(df[df['slice_id'].isin(test_slices)])
    
    print(f"\n[统计]")
    print(f"  总节点数: {len(df)}")
    print(f"  训练集: {train_count}")
    print(f"  验证集: {dev_count}")
    print(f"  测试集: {test_count}")
    print(f"  对话数: {df['dialog_id'].nunique()}")
    


def main():
    parser = argparse.ArgumentParser(description='导出 tiage → TAKE 格式数据')
    parser.add_argument('--input', type=str,
                        default='outputs_nodes/tiage_anno_nodes_all.csv',
                        help='输入 CSV 文件路径')
    parser.add_argument('--out', type=str,
                        default='../../knowSelect/datasets/tiage',
                        help='输出目录')
    args = parser.parse_args()

    # 解析相对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, args.input)
    output_dir = os.path.join(script_dir, args.out)

    print(f"[*] 输入文件: {input_path}")
    print(f"[*] 输出目录: {output_dir}")

    df = load_tiage_nodes(input_path)
    export_take_dataset(df, output_dir)

    print(f"\n[完成] TAKE 数据已导出到 {output_dir}")


if __name__ == '__main__':
    main()

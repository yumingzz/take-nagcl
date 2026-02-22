#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
話題轉移檢測評估腳本

功能：
1. 加載訓練好的模型進行推論
2. 輸出話題轉移預測結果 (0=不轉移, 1=轉移)
3. 輸出 Top-K 高中心性句子及內容
4. 計算 Precision、Recall、F1 評價指標
5. 保存消融實驗結果

使用方法：
    python evaluate_topic_shift.py --name TAKE_tiage --epoch 9
"""

import argparse
import json
import os
import sys

sys.path.append('./')

import pandas as pd
from datetime import datetime


def load_shift_metrics(output_path: str) -> dict:
    """加載話題轉移評價指標"""
    metrics_file = os.path.join(output_path, "metrics", "shift_metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def load_shift_top3(output_path: str) -> list:
    """加載話題轉移 Top-3 句子記錄"""
    top3_file = os.path.join(output_path, "metrics", "shift_top3.jsonl")
    records = []
    if os.path.exists(top3_file):
        with open(top3_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def load_ablation_results(output_path: str) -> pd.DataFrame:
    """加載消融實驗結果"""
    csv_file = os.path.join(output_path, "metrics", "ablation_results.csv")
    if os.path.exists(csv_file):
        return pd.read_csv(csv_file)
    return pd.DataFrame()


def generate_report(args):
    """生成評估報告"""
    output_path = os.path.join(args.base_output_path, args.name)
    
    print("=" * 70)
    print("話題轉移檢測評估報告")
    print("=" * 70)
    print(f"實驗名稱: {args.name}")
    print(f"輸出路徑: {output_path}")
    print(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 1. 評價指標
    print("-" * 70)
    print("1. 話題轉移檢測評價指標 (Precision / Recall / F1)")
    print("-" * 70)
    
    metrics = load_shift_metrics(output_path)
    if metrics:
        for key, value in metrics.items():
            print(f"  {key}:")
            print(f"    Precision: {value.get('precision', 'N/A')}%")
            print(f"    Recall:    {value.get('recall', 'N/A')}%")
            print(f"    F1:        {value.get('f1', 'N/A')}%")
    else:
        print("  尚無評價指標數據，請先運行推論。")
    print()
    
    # 2. Top-K 高中心性句子
    print("-" * 70)
    print(f"2. 話題轉移 Top-{args.top_k} 高中心性句子")
    print("-" * 70)
    
    top3_records = load_shift_top3(output_path)
    if top3_records:
        # 按 run 和 epoch 過濾
        filtered_records = [
            r for r in top3_records
            if r.get('run') == args.name and (args.epoch == -1 or r.get('epoch') == str(args.epoch))
        ]
        
        if filtered_records:
            for i, record in enumerate(filtered_records[:args.max_display]):
                print(f"\n  對話 {i+1}: {record.get('dialog_id', 'N/A')}")
                print(f"  話題轉移: {'是' if record.get('shift_found') else '否'}")
                
                top3 = record.get('shift_top3', [])
                if top3:
                    print(f"  Top-{len(top3[:args.top_k])} 高中心性句子:")
                    for j, item in enumerate(top3[:args.top_k]):
                        print(f"    [{j+1}] 中心性: {item.get('centrality', 0):.4f}")
                        print(f"        節點ID: {item.get('node_id', 'N/A')}")
                        print(f"        句子: {item.get('sentence', 'N/A')[:100]}...")
        else:
            print("  未找到匹配的記錄。")
    else:
        print("  尚無 Top-K 句子數據，請先運行推論。")
    print()
    
    # 3. 消融實驗結果
    print("-" * 70)
    print("3. 消融實驗結果對比")
    print("-" * 70)
    
    ablation_df = load_ablation_results(output_path)
    if not ablation_df.empty:
        print(ablation_df.to_string(index=False))
    else:
        print("  尚無消融實驗數據。")
    print()
    
    # 4. 保存報告
    report_file = os.path.join(output_path, "evaluation_report.txt")
    save_report_to_file(args, output_path, metrics, top3_records, ablation_df, report_file)
    print("-" * 70)
    print(f"報告已保存至: {report_file}")
    print("=" * 70)


def save_report_to_file(args, output_path, metrics, top3_records, ablation_df, report_file):
    """保存報告到文件"""
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("話題轉移檢測評估報告\n")
        f.write("=" * 70 + "\n")
        f.write(f"實驗名稱: {args.name}\n")
        f.write(f"輸出路徑: {output_path}\n")
        f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 評價指標
        f.write("-" * 70 + "\n")
        f.write("1. 話題轉移檢測評價指標\n")
        f.write("-" * 70 + "\n")
        if metrics:
            for key, value in metrics.items():
                f.write(f"  {key}:\n")
                f.write(f"    Precision: {value.get('precision', 'N/A')}%\n")
                f.write(f"    Recall:    {value.get('recall', 'N/A')}%\n")
                f.write(f"    F1:        {value.get('f1', 'N/A')}%\n")
        f.write("\n")
        
        # Top-K 句子
        f.write("-" * 70 + "\n")
        f.write(f"2. 話題轉移 Top-{args.top_k} 高中心性句子 (共 {len(top3_records)} 條記錄)\n")
        f.write("-" * 70 + "\n")
        for i, record in enumerate(top3_records[:args.max_display]):
            f.write(f"\n  對話 {i+1}: {record.get('dialog_id', 'N/A')}\n")
            f.write(f"  話題轉移: {'是' if record.get('shift_found') else '否'}\n")
            top3 = record.get('shift_top3', [])
            for j, item in enumerate(top3[:args.top_k]):
                f.write(f"    [{j+1}] 中心性: {item.get('centrality', 0):.4f}, ")
                f.write(f"節點ID: {item.get('node_id', 'N/A')}\n")
                f.write(f"        句子: {item.get('sentence', 'N/A')}\n")
        f.write("\n")
        
        # 消融實驗
        f.write("-" * 70 + "\n")
        f.write("3. 消融實驗結果對比\n")
        f.write("-" * 70 + "\n")
        if not ablation_df.empty:
            f.write(ablation_df.to_string(index=False))
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="話題轉移檢測評估腳本")
    parser.add_argument("--name", type=str, default="TAKE_tiage", help="實驗名稱")
    parser.add_argument("--base_output_path", type=str, default="output/", help="輸出路徑")
    parser.add_argument("--epoch", type=int, default=-1, help="指定 epoch (-1 表示所有)")
    parser.add_argument("--top_k", type=int, default=3, help="Top-K 句子數量")
    parser.add_argument("--max_display", type=int, default=10, help="最大顯示對話數")
    
    args = parser.parse_args()
    generate_report(args)


if __name__ == "__main__":
    main()

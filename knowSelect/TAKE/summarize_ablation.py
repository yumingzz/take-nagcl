#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融實驗彙總腳本

功能：
1. 收集所有消融實驗的結果
2. 生成對比表格 (純文本 / 文本+imp_pct / 文本+6維特徵)
3. 輸出 Markdown 格式報告

使用方法：
    python summarize_ablation.py --output_dir output/
"""

import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd

# 實驗配置
ABLATION_CONFIGS = [
    {"name": "TAKE_tiage_text_only", "feature_set": "none", "description": "純文本基線"},
    {"name": "TAKE_tiage_imp_pct", "feature_set": "imp_pct", "description": "文本 + imp_pct"},
    {"name": "TAKE_tiage_all_feats", "feature_set": "all", "description": "文本 + 6維結構特徵"},
]


def collect_results(output_dir: str) -> pd.DataFrame:
    """收集所有實驗的結果"""
    results = []
    
    for config in ABLATION_CONFIGS:
        exp_path = os.path.join(output_dir, config["name"])
        metrics_file = os.path.join(exp_path, "metrics", "shift_metrics.json")
        result_file = os.path.join(exp_path, "test_result.json")
        
        entry = {
            "實驗名稱": config["name"],
            "特徵配置": config["description"],
            "feature_set": config["feature_set"],
            "Precision": "-",
            "Recall": "-",
            "F1": "-",
            "ks_acc": "-",
            "ID_acc": "-",
        }
        
        # 讀取話題轉移指標
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
                # 取最後一個 epoch 的結果
                if metrics:
                    last_key = list(metrics.keys())[-1]
                    last_metrics = metrics[last_key]
                    entry["Precision"] = f"{last_metrics.get('precision', '-')}%"
                    entry["Recall"] = f"{last_metrics.get('recall', '-')}%"
                    entry["F1"] = f"{last_metrics.get('f1', '-')}%"
        
        # 讀取知識選擇和話題判別準確率
        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
                if result:
                    last_key = list(result.keys())[-1]
                    last_result = result[last_key]
                    entry["ks_acc"] = f"{last_result.get('ks_acc', '-')}%"
                    entry["ID_acc"] = f"{last_result.get('ID_acc', '-')}%"
        
        results.append(entry)
    
    return pd.DataFrame(results)


def generate_markdown_report(df: pd.DataFrame, output_file: str):
    """生成 Markdown 格式報告"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 消融實驗結果報告\n\n")
        f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 實驗配置\n\n")
        f.write("| 配置 | 描述 | 特徵集 |\n")
        f.write("|------|------|--------|\n")
        f.write("| A1 | 純文本基線 | none |\n")
        f.write("| A2 | 文本 + 重要性分位數 | imp_pct |\n")
        f.write("| A3 | 文本 + 6維結構特徵 | all |\n\n")
        
        f.write("## 話題轉移檢測指標\n\n")
        f.write("| 實驗 | Precision | Recall | F1 |\n")
        f.write("|------|-----------|--------|----|\n")
        for _, row in df.iterrows():
            f.write(f"| {row['特徵配置']} | {row['Precision']} | {row['Recall']} | {row['F1']} |\n")
        f.write("\n")
        
        f.write("## 知識選擇與話題判別準確率\n\n")
        f.write("| 實驗 | ks_acc | ID_acc |\n")
        f.write("|------|--------|--------|\n")
        for _, row in df.iterrows():
            f.write(f"| {row['特徵配置']} | {row['ks_acc']} | {row['ID_acc']} |\n")
        f.write("\n")
        
        f.write("## 指標說明\n\n")
        f.write("- **Precision (精確率)**: 預測為話題轉移的樣本中，真正是話題轉移的比例\n")
        f.write("- **Recall (召回率)**: 所有真實話題轉移樣本中，被正確預測的比例\n")
        f.write("- **F1**: Precision 和 Recall 的調和平均數\n")
        f.write("- **ks_acc**: 知識選擇準確率\n")
        f.write("- **ID_acc**: 話題轉移判別準確率\n")


def main():
    parser = argparse.ArgumentParser(description="消融實驗彙總腳本")
    parser.add_argument("--output_dir", type=str, default="output/", help="實驗輸出目錄")
    parser.add_argument("--report_file", type=str, default="", help="報告輸出文件")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("消融實驗結果彙總")
    print("=" * 70)
    print(f"輸出目錄: {args.output_dir}")
    print()
    
    # 收集結果
    df = collect_results(args.output_dir)
    
    # 顯示表格
    print("-" * 70)
    print("話題轉移檢測評價指標對比")
    print("-" * 70)
    display_cols = ["特徵配置", "Precision", "Recall", "F1", "ks_acc", "ID_acc"]
    print(df[display_cols].to_string(index=False))
    print()
    
    # 生成報告
    if args.report_file:
        report_file = args.report_file
    else:
        report_file = os.path.join(args.output_dir, "ablation_summary.md")
    
    generate_markdown_report(df, report_file)
    print(f"Markdown 報告已保存至: {report_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()

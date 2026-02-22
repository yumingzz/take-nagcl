#!/usr/bin/env python3
"""
TAKE-tiage 完整运行脚本 - 带时间戳输出

本脚本执行以下步骤：
1. 安装必要依赖
2. 生成 tiage → TAKE 格式数据
3. 导出 DGCN3 中心性预测
4. 运行 TAKE 训练和推论
5. 将结果保存到时间戳文件夹

用法:
    python run_tiage_with_timestamp.py [--skip-install] [--skip-data-prep] [--epochs N]
"""

import os
import sys
import subprocess
import shutil
import argparse
from datetime import datetime


def get_timestamp() -> str:
    """获取时间戳字符串"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_command(cmd: list, cwd: str = None, check: bool = True):
    """运行命令并打印输出"""
    print(f"\n[CMD] {' '.join(cmd)}")
    if cwd:
        print(f"[CWD] {cwd}")
    result = subprocess.run(cmd, cwd=cwd, check=check)
    return result


def install_dependencies(project_root: str):
    """安装依赖"""
    print("\n" + "=" * 60)
    print("Step 1: 安装 Python 依赖")
    print("=" * 60)

    requirements_path = os.path.join(project_root, "requirements.txt")
    run_command([sys.executable, "-m", "pip", "install", "-r", requirements_path])

    # 安装额外依赖（IMPLEMENTATION_PLAN.md 中提到的）
    extra_deps = [
        "python-louvain",
        "networkx",
        "scikit-learn",
        "sentence-transformers",
        "umap-learn",
        "pandas",
        "numpy"
    ]
    for dep in extra_deps:
        try:
            run_command([sys.executable, "-m", "pip", "install", dep], check=False)
        except Exception as e:
            print(f"[WARN] 安装 {dep} 时出错: {e}")


def prepare_tiage_data(project_root: str):
    """准备 tiage 数据"""
    print("\n" + "=" * 60)
    print("Step 2: 生成 tiage → TAKE 格式数据")
    print("=" * 60)

    tiage_dir = os.path.join(project_root, "demo", "tiage-1")
    export_script = os.path.join(tiage_dir, "export_take_dataset.py")

    if not os.path.exists(export_script):
        raise FileNotFoundError(f"找不到数据导出脚本: {export_script}")

    output_dir = os.path.join(project_root, "knowSelect", "datasets", "tiage")
    run_command([
        sys.executable, export_script,
        "--out", output_dir
    ], cwd=tiage_dir)


def export_centrality_predictions(project_root: str, alpha: float = 1.5):
    """导出 DGCN3 中心性预测"""
    print("\n" + "=" * 60)
    print("Step 3: 导出 DGCN3 中心性预测")
    print("=" * 60)

    export_script = os.path.join(project_root, "dgcn3_export_predictions.py")

    if not os.path.exists(export_script):
        raise FileNotFoundError(f"找不到导出脚本: {export_script}")

    run_command([
        sys.executable, export_script,
        "--dataset_name", "tiage",
        "--alphas", str(alpha)
    ], cwd=project_root)


def run_take_training(project_root: str, name: str, epochs: int = 5, use_centrality: bool = True):
    """运行 TAKE 训练"""
    print("\n" + "=" * 60)
    print("Step 4: 运行 TAKE 训练")
    print("=" * 60)

    knowselect_dir = os.path.join(project_root, "knowSelect")

    cmd = [
        sys.executable, "./TAKE/Run.py",
        "--name", name,
        "--dataset", "tiage",
        "--mode", "train",
        "--epoches", str(epochs)
    ]

    if use_centrality:
        cmd.extend([
            "--use_centrality",
            "--centrality_alpha", "1.5",
            "--centrality_feature_set", "all",
            "--centrality_window", "2",
            "--node_id_json", "datasets/tiage/node_id.json",
            "--dgcn_predictions_dir", "../demo/DGCN3/Centrality",
            "--edge_lists_dir", "../demo/DGCN3/datasets/raw_data/tiage",
            "--node_mapping_csv", "../demo/tiage-1/outputs_nodes/tiage_anno_nodes_all.csv"
        ])

    run_command(cmd, cwd=knowselect_dir)


def run_take_inference(project_root: str, name: str, use_centrality: bool = True):
    """运行 TAKE 推论"""
    print("\n" + "=" * 60)
    print("Step 5: 运行 TAKE 推论")
    print("=" * 60)

    knowselect_dir = os.path.join(project_root, "knowSelect")

    cmd = [
        sys.executable, "./TAKE/Run.py",
        "--name", name,
        "--dataset", "tiage",
        "--mode", "inference"
    ]

    if use_centrality:
        cmd.extend([
            "--use_centrality",
            "--centrality_alpha", "1.5",
            "--centrality_feature_set", "all",
            "--centrality_window", "2",
            "--node_id_json", "datasets/tiage/node_id.json",
            "--dgcn_predictions_dir", "../demo/DGCN3/Centrality",
            "--edge_lists_dir", "../demo/DGCN3/datasets/raw_data/tiage",
            "--node_mapping_csv", "../demo/tiage-1/outputs_nodes/tiage_anno_nodes_all.csv"
        ])

    run_command(cmd, cwd=knowselect_dir)


def copy_results_to_timestamp_folder(project_root: str, name: str, timestamp: str):
    """将结果复制到时间戳文件夹"""
    print("\n" + "=" * 60)
    print("Step 6: 保存结果到时间戳文件夹")
    print("=" * 60)

    # 源目录
    source_dir = os.path.join(project_root, "knowSelect", "output", name)

    # 目标目录（带时间戳）
    results_base = os.path.join(project_root, "results")
    timestamp_dir = os.path.join(results_base, f"run_{timestamp}")

    os.makedirs(timestamp_dir, exist_ok=True)

    if os.path.exists(source_dir):
        # 复制整个输出目录
        dest_dir = os.path.join(timestamp_dir, name)
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        shutil.copytree(source_dir, dest_dir)
        print(f"[OK] 结果已保存到: {dest_dir}")
    else:
        print(f"[WARN] 源目录不存在: {source_dir}")

    # 复制中心性预测结果
    centrality_dir = os.path.join(project_root, "demo", "DGCN3", "Centrality")
    if os.path.exists(centrality_dir):
        dest_centrality = os.path.join(timestamp_dir, "Centrality")
        if os.path.exists(dest_centrality):
            shutil.rmtree(dest_centrality)
        shutil.copytree(centrality_dir, dest_centrality)
        print(f"[OK] 中心性预测已保存到: {dest_centrality}")

    # 创建运行信息文件
    info_path = os.path.join(timestamp_dir, "run_info.txt")
    with open(info_path, 'w') as f:
        f.write(f"Run Timestamp: {timestamp}\n")
        f.write(f"Model Name: {name}\n")
        f.write(f"Dataset: tiage\n")
        f.write(f"Centrality Alpha: 1.5\n")
        f.write(f"Feature Set: all\n")
    print(f"[OK] 运行信息已保存到: {info_path}")

    return timestamp_dir


def main():
    parser = argparse.ArgumentParser(description='TAKE-tiage 完整运行脚本')
    parser.add_argument('--skip-install', action='store_true',
                        help='跳过依赖安装')
    parser.add_argument('--skip-data-prep', action='store_true',
                        help='跳过数据准备（假设数据已存在）')
    parser.add_argument('--skip-centrality', action='store_true',
                        help='跳过中心性预测导出')
    parser.add_argument('--epochs', type=int, default=5,
                        help='训练轮数（默认: 5）')
    parser.add_argument('--no-centrality', action='store_true',
                        help='不使用中心性特征')
    args = parser.parse_args()

    # 获取项目根目录
    project_root = os.path.dirname(os.path.abspath(__file__))
    timestamp = get_timestamp()
    model_name = f"TAKE_tiage_{timestamp}"

    print("=" * 60)
    print("TAKE-tiage 完整运行脚本")
    print(f"时间戳: {timestamp}")
    print(f"模型名称: {model_name}")
    print("=" * 60)

    try:
        # Step 1: 安装依赖
        if not args.skip_install:
            install_dependencies(project_root)
        else:
            print("\n[跳过] 依赖安装")

        # Step 2: 准备数据
        if not args.skip_data_prep:
            prepare_tiage_data(project_root)
        else:
            print("\n[跳过] 数据准备")

        # Step 3: 导出中心性预测
        if not args.skip_centrality and not args.no_centrality:
            export_centrality_predictions(project_root)
        else:
            print("\n[跳过] 中心性预测导出")

        # Step 4: 训练
        use_centrality = not args.no_centrality
        run_take_training(project_root, model_name, epochs=args.epochs,
                         use_centrality=use_centrality)

        # Step 5: 推论
        run_take_inference(project_root, model_name, use_centrality=use_centrality)

        # Step 6: 保存结果
        results_dir = copy_results_to_timestamp_folder(project_root, model_name, timestamp)

        print("\n" + "=" * 60)
        print("运行完成！")
        print(f"结果保存在: {results_dir}")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] 运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

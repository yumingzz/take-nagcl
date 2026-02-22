#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys


def _run(cmd, cwd=None):
    print("[*] " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def _add_common_take_args(parser):
    parser.add_argument("--dataset", type=str, default="tiage")
    parser.add_argument("--name", type=str, default="TAKE_tiage_all_feats")
    parser.add_argument("--use-centrality", dest="use_centrality", action="store_true")
    parser.add_argument("--no-centrality", dest="use_centrality", action="store_false")
    parser.set_defaults(use_centrality=True)
    parser.add_argument("--centrality-alpha", type=float, default=1.5)
    parser.add_argument("--centrality-feature-set", type=str, default="all", choices=["none", "imp_pct", "all"])
    parser.add_argument("--centrality-window", type=int, default=2)
    parser.add_argument("--node-id-json", type=str, default="datasets/tiage/node_id.json")
    parser.add_argument("--dgcn-predictions-dir", type=str, default=None)
    parser.add_argument("--edge-lists-dir", type=str, default=None)
    parser.add_argument("--node-mapping-csv", type=str, default=None)
    parser.add_argument("--base-data-path", type=str, default=None)
    parser.add_argument("--base-output-path", type=str, default=None)


def _build_take_cmd(args, mode):
    cmd = [sys.executable, "./TAKE/Run.py", "--name", args.name, "--dataset", args.dataset, "--mode", mode]
    if args.use_centrality:
        cmd.append("--use_centrality")
        cmd += ["--centrality_alpha", str(args.centrality_alpha)]
        cmd += ["--centrality_feature_set", args.centrality_feature_set]
        cmd += ["--centrality_window", str(args.centrality_window)]
        cmd += ["--node_id_json", args.node_id_json]
        if args.dgcn_predictions_dir:
            cmd += ["--dgcn_predictions_dir", args.dgcn_predictions_dir]
        if args.edge_lists_dir:
            cmd += ["--edge_lists_dir", args.edge_lists_dir]
        if args.node_mapping_csv:
            cmd += ["--node_mapping_csv", args.node_mapping_csv]
    if args.base_data_path:
        cmd += ["--base_data_path", args.base_data_path]
    if args.base_output_path:
        cmd += ["--base_output_path", args.base_output_path]
    return cmd


def cmd_export_centrality(args):
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "dgcn3_export_predictions.py"),
        "--dataset_name",
        args.dataset_name,
        "--alphas",
        args.alphas,
    ]
    if args.model_path:
        cmd += ["--model_path", args.model_path]
    if args.output_dir:
        cmd += ["--output_dir", args.output_dir]
    _run(cmd)


def cmd_train_take(args):
    cmd = _build_take_cmd(args, "train")
    _run(cmd, cwd=os.path.join(os.path.dirname(__file__), "knowSelect"))


def cmd_infer_take(args):
    cmd = _build_take_cmd(args, "inference")
    _run(cmd, cwd=os.path.join(os.path.dirname(__file__), "knowSelect"))


def cmd_ablation(args):
    base = dict(
        dataset=args.dataset,
        centrality_alpha=args.centrality_alpha,
        centrality_window=args.centrality_window,
        node_id_json=args.node_id_json,
        dgcn_predictions_dir=args.dgcn_predictions_dir,
        edge_lists_dir=args.edge_lists_dir,
        node_mapping_csv=args.node_mapping_csv,
        base_data_path=args.base_data_path,
        base_output_path=args.base_output_path,
    )

    configs = [
        ("TAKE_tiage_text_only", "none", False),
        ("TAKE_tiage_imp_pct", "imp_pct", True),
        ("TAKE_tiage_all_feats", "all", True),
    ]

    for name, feature_set, use_centrality in configs:
        args.name = name
        args.centrality_feature_set = feature_set
        args.use_centrality = use_centrality
        cmd_train_take(args)
        cmd_infer_take(args)


def cmd_pipeline(args):
    cmd_export_centrality(args)
    cmd_train_take(args)
    cmd_infer_take(args)


def cmd_generate_shift_answers(args):
    # 以 knowSelect 推論輸出的 metrics/shift_top3.jsonl 生成每個 shift 事件的 GPT-2 回答文字檔
    # 注意：knowSelect/TAKE 不是 Python package；這裡用 sys.path 動態加入後再 import
    knowselect_dir = os.path.join(os.path.dirname(__file__), "knowSelect")
    if knowselect_dir not in sys.path:
        sys.path.insert(0, knowselect_dir)
    from TAKE.ShiftAnswerGenerator import GeneratorConfig, generate_shift_answers_txt  # type: ignore

    base_output = args.base_output_path or "output/"
    run_dir = os.path.join(os.path.dirname(__file__), "knowSelect", base_output, args.name)
    metrics_dir = os.path.join(run_dir, "metrics")
    shift_top3_jsonl = os.path.join(metrics_dir, "shift_top3.jsonl")
    out_txt = os.path.join(metrics_dir, f"shift_answers_{args.split}_{args.epoch}.txt")

    cfg = GeneratorConfig(
        model_name_or_path=args.gpt2_model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.greedy,
    )

    generate_shift_answers_txt(
        shift_top3_jsonl=shift_top3_jsonl,
        out_txt=out_txt,
        cfg=cfg,
        only_split=args.split,
        only_epoch=str(args.epoch),
    )
    print(f"[*] Saved shift answers to {out_txt}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export-centrality")
    export_parser.add_argument("--dataset-name", type=str, default="tiage")
    export_parser.add_argument("--alphas", type=str, default="1.5")
    export_parser.add_argument("--model-path", type=str, default="")
    export_parser.add_argument("--output-dir", type=str, default="")
    export_parser.set_defaults(func=cmd_export_centrality)

    train_parser = subparsers.add_parser("train-take")
    _add_common_take_args(train_parser)
    train_parser.set_defaults(func=cmd_train_take)

    infer_parser = subparsers.add_parser("infer-take")
    _add_common_take_args(infer_parser)
    infer_parser.set_defaults(func=cmd_infer_take)

    ablation_parser = subparsers.add_parser("ablation")
    _add_common_take_args(ablation_parser)
    ablation_parser.set_defaults(func=cmd_ablation)

    pipeline_parser = subparsers.add_parser("pipeline")
    _add_common_take_args(pipeline_parser)
    pipeline_parser.add_argument("--dataset-name", type=str, default="tiage")
    pipeline_parser.add_argument("--alphas", type=str, default="1.5")
    pipeline_parser.add_argument("--model-path", type=str, default="")
    pipeline_parser.add_argument("--output-dir", type=str, default="")
    pipeline_parser.set_defaults(func=cmd_pipeline)

    gen_shift_parser = subparsers.add_parser("generate-shift-answers")
    _add_common_take_args(gen_shift_parser)
    gen_shift_parser.add_argument("--split", type=str, default="test", help="對應 knowSelect 推論輸出 record_out 的 dataset 欄位（實際為 split，例如 test）")
    gen_shift_parser.add_argument("--epoch", type=str, default="all", help="對應 knowSelect 推論輸出 record_out 的 epoch（字串）；all 表示不過濾")
    gen_shift_parser.add_argument("--gpt2-model", type=str, default="gpt2", help="GPT-2 模型名稱或本地路徑（transformers 可讀）")
    gen_shift_parser.add_argument("--max-new-tokens", type=int, default=80)
    gen_shift_parser.add_argument("--temperature", type=float, default=0.7)
    gen_shift_parser.add_argument("--top-p", type=float, default=0.9)
    gen_shift_parser.add_argument("--greedy", action="store_true", help="不採樣（greedy decoding）")
    gen_shift_parser.set_defaults(func=cmd_generate_shift_answers)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

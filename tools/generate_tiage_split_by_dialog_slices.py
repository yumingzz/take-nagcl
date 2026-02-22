#!/usr/bin/env python3
"""
依 dialog_id 分箱產生 knowSelect 的 tiage.split

規則（依使用者已批准）：
- dialog_id 以數值排序（int(dialog_id)）
- 每 50 個 dialogs 為一個 slice：slice_id = floor(index / 50)
- TAKE 切分（選項 B）：
  - train：slice 0..7
  - test：slice >= 8
- 允許：
  - 最後一片 dialogs < 50
  - slices > 10（此時 slice>=8 仍視為 test）

輸出：
- knowSelect/datasets/tiage/tiage.split（每行：<dialog_id>_0 <TAB> train|test）

注意：
- knowSelect 的 split_data(tiage) 使用 episode[0]['query_id'] 與 split 檔比對，
  tiage 的 episode 起始 query_id 通常為 "<dialog_id>_0"。
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class SplitConfig:
    dialogs_per_slice: int = 50
    train_max_slice_inclusive: int = 7


def _read_dialog_ids_from_csv(path: str) -> List[int]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 tiage_anno_nodes_all.csv：{path}")
    dialog_ids = set()
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "dialog_id" not in (reader.fieldnames or []):
            raise ValueError(f"CSV 缺少 dialog_id 欄位：{path}")
        for row in reader:
            raw = row.get("dialog_id", "")
            if raw is None or str(raw).strip() == "":
                continue
            try:
                dialog_ids.add(int(str(raw)))
            except ValueError as e:
                raise ValueError(f"dialog_id 無法轉成整數：{raw}") from e
    return sorted(dialog_ids)


def _build_dialog_split(dialog_ids: List[int], cfg: SplitConfig) -> Dict[int, str]:
    split: Dict[int, str] = {}
    for idx, did in enumerate(dialog_ids):
        slice_id = idx // cfg.dialogs_per_slice
        split[did] = "train" if slice_id <= cfg.train_max_slice_inclusive else "test"
    return split


def _write_split(path: str, dialog_split: Dict[int, str]) -> Tuple[int, int]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    train_n = 0
    test_n = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        for did in sorted(dialog_split.keys()):
            q0 = f"{did}_0"
            label = dialog_split[did]
            if label == "train":
                train_n += 1
            else:
                test_n += 1
            f.write(f"{q0}\t{label}\n")
    return train_n, test_n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--anno-csv",
        required=True,
        help="demo/tiage-1/outputs_nodes/tiage_anno_nodes_all.csv",
    )
    parser.add_argument(
        "--out-split",
        required=True,
        help="knowSelect/datasets/tiage/tiage.split",
    )
    parser.add_argument("--dialogs-per-slice", type=int, default=50)
    parser.add_argument("--train-max-slice", type=int, default=7, help="slice<=N 視為 train，其餘 test")
    args = parser.parse_args()

    cfg = SplitConfig(dialogs_per_slice=args.dialogs_per_slice, train_max_slice_inclusive=args.train_max_slice)
    dialog_ids = _read_dialog_ids_from_csv(args.anno_csv)
    dialog_split = _build_dialog_split(dialog_ids, cfg)
    train_n, test_n = _write_split(args.out_split, dialog_split)

    total = train_n + test_n
    slices = (total + cfg.dialogs_per_slice - 1) // cfg.dialogs_per_slice if total > 0 else 0
    print(f"[*] dialogs: total={total} train={train_n} test={test_n}")
    print(f"[*] slice config: dialogs_per_slice={cfg.dialogs_per_slice} slices≈{slices} train_max_slice={cfg.train_max_slice_inclusive}")
    print(f"[*] wrote split: {args.out_split}")


if __name__ == "__main__":
    main()


#!/usr/bin/env bash
# 依 dialog_id（數值排序）每 50 dialogs 分箱產生 tiage.split（TAKE：train=0..7，test>=8）
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "錯誤：找不到 uv。請先執行：bash scripts/unix/uv_setup.sh"
  exit 1
fi

# 若非一鍵流程呼叫，單獨執行時也要建立本次 run 的 logs/outputs
LOG_FILE=""
if [ "${ORCHESTRATED:-0}" != "1" ]; then
  if [ -z "${RUN_ID:-}" ]; then
    RUN_ID="$(date '+%Y-%m-%d_%H-%M-%S')"
  fi
  if [ -z "${RUN_LOG_DIR:-}" ]; then
    RUN_LOG_DIR="${PROJECT_ROOT}/logs/${RUN_ID}"
  fi
  if [ -z "${RUN_OUTPUT_DIR:-}" ]; then
    RUN_OUTPUT_DIR="${PROJECT_ROOT}/outputs/${RUN_ID}"
  fi
  mkdir -p "$RUN_LOG_DIR" "$RUN_OUTPUT_DIR"
  export RUN_ID RUN_LOG_DIR RUN_OUTPUT_DIR
  LOG_FILE="${RUN_LOG_DIR}/01_generate_split.log"
fi

ANNO_CSV="${PROJECT_ROOT}/demo/tiage-1/outputs_nodes/tiage_anno_nodes_all.csv"
OUT_SPLIT="${PROJECT_ROOT}/knowSelect/datasets/tiage/tiage.split"

if [ -n "$LOG_FILE" ]; then
  uv run python tools/generate_tiage_split_by_dialog_slices.py \
    --anno-csv "$ANNO_CSV" \
    --out-split "$OUT_SPLIT" \
    --dialogs-per-slice 50 \
    --train-max-slice 7 2>&1 | tee -a "$LOG_FILE"
else
  uv run python tools/generate_tiage_split_by_dialog_slices.py \
    --anno-csv "$ANNO_CSV" \
    --out-split "$OUT_SPLIT" \
    --dialogs-per-slice 50 \
    --train-max-slice 7
fi

echo "[OK] 已更新：$OUT_SPLIT"

# 若由一鍵腳本執行，將 split 複製到本次 outputs 目錄以便回溯
if [ -n "${RUN_OUTPUT_DIR:-}" ]; then
  mkdir -p "${RUN_OUTPUT_DIR}/splits"
  cp -f "$OUT_SPLIT" "${RUN_OUTPUT_DIR}/splits/tiage.split"
  echo "[OK] 已保存本次 split：${RUN_OUTPUT_DIR}/splits/tiage.split"
fi

if [ -n "$LOG_FILE" ]; then
  echo "[OK] logs:    ${RUN_LOG_DIR}"
  echo "[OK] outputs: ${RUN_OUTPUT_DIR}"
fi


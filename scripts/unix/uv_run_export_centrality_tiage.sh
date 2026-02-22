#!/usr/bin/env bash
# 使用 uv 匯出 tiage 各時間片中心性預測（DGCN3，對所有 slices）
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "錯誤：找不到 uv。請先執行：bash scripts/unix/uv_setup.sh"
  exit 1
fi

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
  LOG_FILE="${RUN_LOG_DIR}/02_export_centrality.log"
fi

OUT_DIR="${PROJECT_ROOT}/demo/DGCN3/Centrality"
if [ -n "${RUN_OUTPUT_DIR:-}" ]; then
  OUT_DIR="${RUN_OUTPUT_DIR}/dgcn3/Centrality"
fi

mkdir -p "$OUT_DIR"
if [ -n "$LOG_FILE" ]; then
  uv run python main.py export-centrality --dataset-name tiage --alphas 1.5 --output-dir "$OUT_DIR" 2>&1 | tee -a "$LOG_FILE"
else
  uv run python main.py export-centrality --dataset-name tiage --alphas 1.5 --output-dir "$OUT_DIR"
fi

if [ -n "$LOG_FILE" ]; then
  echo "[OK] logs:    ${RUN_LOG_DIR}"
  echo "[OK] outputs: ${RUN_OUTPUT_DIR}"
fi


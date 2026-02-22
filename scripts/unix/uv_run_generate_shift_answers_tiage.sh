#!/usr/bin/env bash
# 使用 uv 生成「每個 shift 事件」的 GPT-2 回答文字檔（tiage）
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
  LOG_FILE="${RUN_LOG_DIR}/05_gpt2_answers.log"
fi

DATASET="tiage"
EXPERIMENT_NAME="TAKE_tiage_all_feats"
SPLIT="test"
EPOCH="all"
GPT2_MODEL="gpt2"

CMD=(uv run python main.py generate-shift-answers \
  --dataset "$DATASET" \
  --name "$EXPERIMENT_NAME" \
  --split "$SPLIT" \
  --epoch "$EPOCH" \
  --gpt2-model "$GPT2_MODEL" \
  $(if [ -n "${RUN_OUTPUT_DIR:-}" ]; then echo --base-output-path "${RUN_OUTPUT_DIR}/knowSelect/output/"; fi))

if [ -n "$LOG_FILE" ]; then
  "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
else
  "${CMD[@]}"
fi

if [ -n "$LOG_FILE" ]; then
  echo "[OK] logs:    ${RUN_LOG_DIR}"
  echo "[OK] outputs: ${RUN_OUTPUT_DIR}"
fi

echo "完成：已生成 shift 回答文字檔。"


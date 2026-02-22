#!/usr/bin/env bash
# 使用 uv 訓練 tiage 的 TAKE（knowSelect，含中心性/社團/6維特徵）
set -e

PROJECT_ROOT="$(cd "$(dirname "$0  ")/../.." && pwd)"
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
  LOG_FILE="${RUN_LOG_DIR}/03_take_train.log"
fi

# 若切分規則或 tiage.split 更新，需刪除舊 pkl 讓其重新建構資料。
DATA_DIR="knowSelect/datasets/tiage"
rm -f \
  "${DATA_DIR}/train_TAKE.pkl" \
  "${DATA_DIR}/test_TAKE.pkl" \
  "${DATA_DIR}/query_TAKE.pkl" \
  "${DATA_DIR}/passage_TAKE.pkl"

CMD=(uv run python main.py train-take \
  --dataset tiage \
  --name TAKE_tiage_all_feats \
  --use-centrality \
  --centrality-alpha 1.5 \
  --centrality-feature-set all \
  --centrality-window 2 \
  --node-id-json datasets/tiage/node_id.json \
  --edge-lists-dir "${PROJECT_ROOT}/demo/DGCN3/datasets/raw_data/tiage" \
  --node-mapping-csv "${PROJECT_ROOT}/demo/tiage-1/outputs_nodes/tiage_anno_nodes_all.csv" \
  $(if [ -n "${RUN_OUTPUT_DIR:-}" ]; then echo --base-output-path "${RUN_OUTPUT_DIR}/knowSelect/output/"; fi) \
  $(if [ -n "${RUN_OUTPUT_DIR:-}" ]; then echo --dgcn-predictions-dir "${RUN_OUTPUT_DIR}/dgcn3/Centrality"; fi))

if [ -n "$LOG_FILE" ]; then
  "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
else
  "${CMD[@]}"
fi

if [ -n "$LOG_FILE" ]; then
  echo "[OK] logs:    ${RUN_LOG_DIR}"
  echo "[OK] outputs: ${RUN_OUTPUT_DIR}"
fi


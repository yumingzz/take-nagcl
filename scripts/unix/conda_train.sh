#!/usr/bin/env bash
# 使用 conda 訓練 tiage 的 TAKE（knowSelect，含中心性/社團/6維特徵）
set -euo pipefail

export NLTK_DATA="/root/autodl-tmp/nltk_data"

# -----------------------------
# Resolve PROJECT_ROOT
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

# -----------------------------
# Initialize conda for non-interactive shells
# -----------------------------
# Prefer explicit known path (from your logs: /root/miniconda3/envs/take1/...)
CONDA_BASE="${CONDA_BASE:-/root/miniconda3}"

if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  # Fallback: try to discover conda base if conda exists in PATH
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
  else
    echo "錯誤：找不到 conda。請確認 conda 已安裝，或設置 CONDA_BASE（例如 /root/miniconda3）。"
    exit 1
  fi
fi

CONDA_ENV="${CONDA_ENV:-take1}"
conda activate "$CONDA_ENV"

# -----------------------------
# (Optional but recommended) avoid any online model download attempts
# You said model is already downloaded locally.
# -----------------------------
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

# -----------------------------
# Logging / output dirs (same logic as your uv script)
# -----------------------------
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

# -----------------------------
# Clean old pkl to rebuild data (same as original)
# -----------------------------
DATA_DIR="knowSelect/datasets/tiage"
rm -f \
  "${DATA_DIR}/train_TAKE.pkl" \
  "${DATA_DIR}/test_TAKE.pkl" \
  "${DATA_DIR}/query_TAKE.pkl" \
  "${DATA_DIR}/passage_TAKE.pkl"

# -----------------------------
# Build command
# NOTE:
# - We use python from the activated conda env
# - Keep your original args intact
# -----------------------------
CMD=(python main.py train-take \
  --dataset tiage \
  --name TAKE_tiage_all_feats \
  --use-centrality \
  --centrality-alpha 1.5 \
  --centrality-feature-set all \
  --centrality-window 2 \
  --node-id-json datasets/tiage/node_id.json \
  --edge-lists-dir "${PROJECT_ROOT}/demo/DGCN3/datasets/raw_data/tiage" \
  --node-mapping-csv "${PROJECT_ROOT}/demo/tiage-1/outputs_nodes/tiage_anno_nodes_all.csv")

# Optional output dirs (same behavior as before)
if [ -n "${RUN_OUTPUT_DIR:-}" ]; then
  CMD+=(--base-output-path "${RUN_OUTPUT_DIR}/knowSelect/output/")
  CMD+=(--dgcn-predictions-dir "${RUN_OUTPUT_DIR}/dgcn3/Centrality")
fi

# -----------------------------
# Run
# -----------------------------
echo "[INFO] PROJECT_ROOT: ${PROJECT_ROOT}"
echo "[INFO] CONDA_BASE:   ${CONDA_BASE}"
echo "[INFO] CONDA_ENV:    ${CONDA_ENV}"
echo "[INFO] PYTHON:       $(command -v python)"
echo "[INFO] OFFLINE:      TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"
if [ -n "${RUN_OUTPUT_DIR:-}" ]; then
  echo "[INFO] RUN_OUTPUT_DIR: ${RUN_OUTPUT_DIR}"
fi

if [ -n "$LOG_FILE" ]; then
  "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
else
  "${CMD[@]}"
fi

if [ -n "$LOG_FILE" ]; then
  echo "[OK] logs:    ${RUN_LOG_DIR}"
  echo "[OK] outputs: ${RUN_OUTPUT_DIR}"
fi
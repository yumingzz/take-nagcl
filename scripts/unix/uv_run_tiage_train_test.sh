#!/usr/bin/env bash
# Tiage 完整訓練 + 測試流程（uv 版）
# - uv sync（依 uv.lock）
# - 產生 tiage.split（時間片切分）
# - DGCN3 匯出中心性（所有 slices）
# - TAKE 訓練（train）
# - TAKE 測試推論（test）
# - 生成 shift 事件 GPT-2 回答
# - Smoke Check
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "錯誤：找不到 uv。請先安裝 uv。"
  exit 1
fi

RUN_ID="$(date '+%Y-%m-%d_%H-%M-%S')"
RUN_LOG_DIR="${PROJECT_ROOT}/logs/${RUN_ID}"
RUN_OUTPUT_DIR="${PROJECT_ROOT}/outputs/${RUN_ID}"
mkdir -p "$RUN_LOG_DIR" "$RUN_OUTPUT_DIR"

export RUN_ID
export RUN_LOG_DIR
export RUN_OUTPUT_DIR
export ORCHESTRATED=1

run_step() {
  local step_id="$1"
  local title="$2"
  shift 2
  local log_file="${RUN_LOG_DIR}/${step_id}_${title}.log"
  local start_ts
  start_ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "=== [${start_ts}] START ${step_id} ${title} ===" | tee -a "$log_file"
  local t0
  t0="$(date +%s)"
  set +e
  "$@" 2>&1 | tee -a "$log_file"
  local rc=${PIPESTATUS[0]}
  set -e
  local t1
  t1="$(date +%s)"
  local dur=$((t1 - t0))
  local end_ts
  end_ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "=== [${end_ts}] END ${step_id} ${title} rc=${rc} duration_sec=${dur} ===" | tee -a "$log_file"
  if [ "$rc" -ne 0 ]; then
    echo "[ERROR] step failed: ${step_id} ${title}. See: ${log_file}"
    exit "$rc"
  fi
}

run_step "00" "uv_setup" bash scripts/unix/uv_setup.sh
run_step "01" "generate_split" bash scripts/unix/uv_generate_tiage_split.sh
run_step "02" "export_centrality" bash scripts/unix/uv_run_export_centrality_tiage.sh
run_step "03" "take_train" bash scripts/unix/uv_run_take_tiage_train.sh
run_step "04" "take_infer" bash scripts/unix/uv_run_take_tiage_infer.sh
run_step "05" "gpt2_answers" bash scripts/unix/uv_run_generate_shift_answers_tiage.sh
run_step "06" "smoke_check" bash scripts/unix/uv_smoke_check_tiage_outputs.sh

echo "[OK] Tiage 完整訓練/測試流程完成"
echo "[OK] logs:    ${RUN_LOG_DIR}"
echo "[OK] outputs: ${RUN_OUTPUT_DIR}"


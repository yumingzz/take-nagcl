#!/usr/bin/env bash
# 使用 uv 執行 Tiage 輸出 Smoke Check（不跑訓練，只檢查輸出是否齊全、欄位是否存在）
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "錯誤：找不到 uv。請先執行：bash scripts/unix/uv_setup.sh"
  exit 1
fi

if [ "${ORCHESTRATED:-0}" != "1" ]; then
  if [ -z "${RUN_ID:-}" ]; then
    RUN_ID="$(date '+%Y-%m-%d_%H-%M-%S')"
  fi
  if [ -z "${RUN_LOG_DIR:-}" ]; then
    RUN_LOG_DIR="${PROJECT_ROOT}/logs/${RUN_ID}"
  fi
  mkdir -p "$RUN_LOG_DIR"
  export RUN_ID RUN_LOG_DIR
  LOG_FILE="${RUN_LOG_DIR}/06_smoke_check.log"
  exec > >(tee -a "$LOG_FILE") 2>&1
fi

NAME="TAKE_tiage_all_feats"
BASE_OUTPUT_DIR="${PROJECT_ROOT}/knowSelect/output"
if [ -n "${RUN_OUTPUT_DIR:-}" ]; then
  BASE_OUTPUT_DIR="${RUN_OUTPUT_DIR}/knowSelect/output"
fi
export BASE_OUTPUT_DIR
METRICS_DIR="${BASE_OUTPUT_DIR}/${NAME}/metrics"

echo "[*] 檢查輸出目錄：${METRICS_DIR}"
test -d "$METRICS_DIR"

echo "[*] 檢查 tiage.split 是否存在且包含 train/test"
test -f "${PROJECT_ROOT}/knowSelect/datasets/tiage/tiage.split"

test -f "${METRICS_DIR}/shift_metrics.json"
test -f "${METRICS_DIR}/shift_top3.jsonl"
test -f "${METRICS_DIR}/shift_pred.jsonl"

echo "[*] 檢查 shift_pred.jsonl 欄位（dialog_id/query_id/turn_id/node_id/pred_shift）"
uv run python - << 'PY'
import json, os
base = os.environ.get("BASE_OUTPUT_DIR") or os.path.join("knowSelect","output")
path = os.path.join(base, "TAKE_tiage_all_feats", "metrics", "shift_pred.jsonl")
need = {"dialog_id","query_id","turn_id","node_id","pred_shift"}
with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        line=line.strip()
        if not line:
            continue
        obj=json.loads(line)
        miss=need-obj.keys()
        if miss:
            raise SystemExit(f"缺少欄位：{miss} @ line {i+1}")
        if obj["pred_shift"] not in (0,1):
            raise SystemExit(f"pred_shift 非 0/1 @ line {i+1}")
        break
print("[OK] shift_pred.jsonl 欄位與取值正常（抽樣第一筆）")
PY

echo "[*] 檢查 shift_top3.jsonl 是否包含 shift_events 與 interval_top3.turn_id"
uv run python - << 'PY'
import json, os
base = os.environ.get("BASE_OUTPUT_DIR") or os.path.join("knowSelect","output")
path = os.path.join(base, "TAKE_tiage_all_feats", "metrics", "shift_top3.jsonl")
with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        line=line.strip()
        if not line:
            continue
        obj=json.loads(line)
        events=obj.get("shift_events") or []
        if events:
            ev=events[0]
            top3=ev.get("interval_top3") or []
            if top3 and "turn_id" not in top3[0]:
                raise SystemExit("interval_top3 缺少 turn_id")
        break
print("[OK] shift_top3.jsonl 結構正常（抽樣第一筆）")
PY

echo "[OK] Smoke check 完成"


<#
使用 uv 執行 Tiage 輸出 Smoke Check（不跑訓練，只檢查輸出是否齊全、欄位是否存在）
#>
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $projectRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Error "找不到 uv。請先執行 scripts/windows/uv_setup.ps1"
}

# 若非一鍵流程呼叫，單獨執行時也要建立本次 run 的 logs/（並把本腳本輸出寫入 log）
$logFile = $null
$useTranscript = $false
if ($env:ORCHESTRATED -ne "1") {
  if (-not $env:RUN_ID) { $env:RUN_ID = Get-Date -Format "yyyy-MM-dd_HH-mm-ss" }
  if (-not $env:RUN_LOG_DIR) { $env:RUN_LOG_DIR = Join-Path $projectRoot (Join-Path "logs" $env:RUN_ID) }
  New-Item -ItemType Directory -Force -Path $env:RUN_LOG_DIR | Out-Null
  $logFile = Join-Path $env:RUN_LOG_DIR "06_smoke_check.log"
  $useTranscript = $true
  ("=== [{0}] START 06 smoke_check ===" -f (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")) | Tee-Object -FilePath $logFile -Append
  Start-Transcript -Path $logFile -Append | Out-Null
}

try {
  $name = "TAKE_tiage_all_feats"

  $baseOutputDir = Join-Path $projectRoot "knowSelect\output"
  if ($env:RUN_OUTPUT_DIR) {
    $baseOutputDir = Join-Path $env:RUN_OUTPUT_DIR "knowSelect\output"
  }
  $env:BASE_OUTPUT_DIR = $baseOutputDir

  $metricsDir = Join-Path $baseOutputDir "$name\metrics"
  $splitPath = Join-Path $projectRoot "knowSelect\datasets\tiage\tiage.split"

  Write-Host "[*] 檢查輸出目錄：$metricsDir"
  if (-not (Test-Path $metricsDir)) { throw "找不到 metrics 目錄：$metricsDir" }

  Write-Host "[*] 檢查 tiage.split 是否存在"
  if (-not (Test-Path $splitPath)) { throw "找不到 split：$splitPath" }

  foreach ($f in @("shift_metrics.json","shift_top3.jsonl","shift_pred.jsonl")) {
    $p = Join-Path $metricsDir $f
    if (-not (Test-Path $p)) { throw "缺少輸出檔：$p" }
  }

  Write-Host "[*] 檢查 shift_pred.jsonl 欄位"
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

  Write-Host "[*] 檢查 shift_top3.jsonl 結構"
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

  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  Write-Host "[OK] Smoke check 完成"
}
finally {
  if ($useTranscript) {
    Stop-Transcript | Out-Null
    ("=== [{0}] END 06 smoke_check ===" -f (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")) | Tee-Object -FilePath $logFile -Append
  }
}


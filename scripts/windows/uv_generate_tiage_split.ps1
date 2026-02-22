<#
依 dialog_id（數值排序）每 50 dialogs 分箱產生 tiage.split（TAKE：train=0..7，test>=8）
#>

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $projectRoot

# Check uv command
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Error "uv not found. Please run scripts/windows/uv_setup.ps1 first."
}

$logFile = $null
if ($env:ORCHESTRATED -ne "1") {
  if (-not $env:RUN_ID) { $env:RUN_ID = Get-Date -Format "yyyy-MM-dd_HH-mm-ss" }
  if (-not $env:RUN_LOG_DIR) { $env:RUN_LOG_DIR = Join-Path $projectRoot (Join-Path "logs" $env:RUN_ID) }
  if (-not $env:RUN_OUTPUT_DIR) { $env:RUN_OUTPUT_DIR = Join-Path $projectRoot (Join-Path "outputs" $env:RUN_ID) }
  New-Item -ItemType Directory -Force -Path $env:RUN_LOG_DIR | Out-Null
  New-Item -ItemType Directory -Force -Path $env:RUN_OUTPUT_DIR | Out-Null
  $logFile = Join-Path $env:RUN_LOG_DIR "01_generate_split.log"
}

$annoCsv = Join-Path $projectRoot "demo\tiage-1\outputs_nodes\tiage_anno_nodes_all.csv"
$outSplit = Join-Path $projectRoot "knowSelect\datasets\tiage\tiage.split"

if ($logFile) {
  uv run python tools/generate_tiage_split_by_dialog_slices.py `
    --anno-csv "$annoCsv" `
    --out-split "$outSplit" `
    --dialogs-per-slice 50 `
    --train-max-slice 7 2>&1 | Tee-Object -FilePath $logFile -Append
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
  uv run python tools/generate_tiage_split_by_dialog_slices.py `
    --anno-csv "$annoCsv" `
    --out-split "$outSplit" `
    --dialogs-per-slice 50 `
    --train-max-slice 7
}

Write-Host "[OK] Split updated: $outSplit"

if ($env:RUN_OUTPUT_DIR) {
  $splitDir = Join-Path $env:RUN_OUTPUT_DIR "splits"
  New-Item -ItemType Directory -Force -Path $splitDir | Out-Null
  Copy-Item -Force -Path $outSplit -Destination (Join-Path $splitDir "tiage.split")
  Write-Host "[OK] split saved: " (Join-Path $splitDir "tiage.split")

}

if ($logFile) {
  Write-Host ("[OK] logs:    {0}" -f $env:RUN_LOG_DIR)
  Write-Host ("[OK] outputs: {0}" -f $env:RUN_OUTPUT_DIR)
}


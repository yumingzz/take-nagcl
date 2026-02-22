<#
使用 uv 生成「每個 shift 事件」的 GPT-2 回答文字檔（tiage）
#>
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $projectRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Error "找不到 uv。請先執行 scripts/windows/uv_setup.ps1"
}

$logFile = $null
if ($env:ORCHESTRATED -ne "1") {
  if (-not $env:RUN_ID) { $env:RUN_ID = Get-Date -Format "yyyy-MM-dd_HH-mm-ss" }
  if (-not $env:RUN_LOG_DIR) { $env:RUN_LOG_DIR = Join-Path $projectRoot (Join-Path "logs" $env:RUN_ID) }
  if (-not $env:RUN_OUTPUT_DIR) { $env:RUN_OUTPUT_DIR = Join-Path $projectRoot (Join-Path "outputs" $env:RUN_ID) }
  New-Item -ItemType Directory -Force -Path $env:RUN_LOG_DIR | Out-Null
  New-Item -ItemType Directory -Force -Path $env:RUN_OUTPUT_DIR | Out-Null
  $logFile = Join-Path $env:RUN_LOG_DIR "05_gpt2_answers.log"
}

$dataset = "tiage"
$name = "TAKE_tiage_all_feats"
$split = "test"
$epoch = "all"
$gpt2Model = "gpt2"

$baseOutputPath = $null
if ($env:RUN_OUTPUT_DIR) {
  $baseOutputPath = Join-Path $env:RUN_OUTPUT_DIR "knowSelect\output\"
}

$args = @(
  "main.py","generate-shift-answers",
  "--dataset",$dataset,
  "--name",$name,
  "--split",$split,
  "--epoch",$epoch,
  "--gpt2-model",$gpt2Model
)
if ($baseOutputPath) { $args += @("--base-output-path",$baseOutputPath) }

if ($logFile) {
  uv run python @args 2>&1 | Tee-Object -FilePath $logFile -Append
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
  uv run python @args
}

if ($logFile) {
  Write-Host ("[OK] logs:    {0}" -f $env:RUN_LOG_DIR)
  Write-Host ("[OK] outputs: {0}" -f $env:RUN_OUTPUT_DIR)
}

Write-Host "完成：已生成 shift 回答文字檔。"


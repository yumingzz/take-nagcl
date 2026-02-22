<#
Tiage 完整訓練 + 測試流程（uv 版）
  1) uv sync
  2) 生成 tiage.split
  3) DGCN3 匯出中心性（所有 slices）
  4) TAKE 訓練（train）
  5) TAKE 測試推論（test）
  6) GPT-2 生成 shift 事件回答
  7) Smoke Check
#>
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $projectRoot

$runId = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$runLogDir = Join-Path $projectRoot (Join-Path "logs" $runId)
$runOutputDir = Join-Path $projectRoot (Join-Path "outputs" $runId)
New-Item -ItemType Directory -Force -Path $runLogDir | Out-Null
New-Item -ItemType Directory -Force -Path $runOutputDir | Out-Null

$env:RUN_ID = $runId
$env:RUN_LOG_DIR = $runLogDir
$env:RUN_OUTPUT_DIR = $runOutputDir
$env:ORCHESTRATED = "1"

function Invoke-Step {
  param(
    [Parameter(Mandatory=$true)][string]$StepId,
    [Parameter(Mandatory=$true)][string]$Title,
    [Parameter(Mandatory=$true)][scriptblock]$Action
  )
  $logFile = Join-Path $runLogDir ("{0}_{1}.log" -f $StepId, $Title)
  $start = Get-Date
  ("=== [{0}] START {1} {2} ===" -f $start.ToString("yyyy-MM-dd HH:mm:ss"), $StepId, $Title) | Tee-Object -FilePath $logFile -Append
  try {
    $elapsed = Measure-Command { & $Action 2>&1 | Tee-Object -FilePath $logFile -Append }
    $end = Get-Date
    ("=== [{0}] END {1} {2} rc=0 duration_sec={3} ===" -f $end.ToString("yyyy-MM-dd HH:mm:ss"), $StepId, $Title, [int]$elapsed.TotalSeconds) | Tee-Object -FilePath $logFile -Append
  } catch {
    $end = Get-Date
    ("=== [{0}] END {1} {2} rc=1 ===" -f $end.ToString("yyyy-MM-dd HH:mm:ss"), $StepId, $Title) | Tee-Object -FilePath $logFile -Append
    throw
  }
}

Invoke-Step -StepId "00" -Title "uv_setup" -Action { & (Join-Path $PSScriptRoot "uv_setup.ps1") }
Invoke-Step -StepId "01" -Title "generate_split" -Action { & (Join-Path $PSScriptRoot "uv_generate_tiage_split.ps1") }
Invoke-Step -StepId "02" -Title "export_centrality" -Action { & (Join-Path $PSScriptRoot "uv_run_export_centrality_tiage.ps1") }
Invoke-Step -StepId "03" -Title "take_train" -Action { & (Join-Path $PSScriptRoot "uv_run_take_tiage_train.ps1") }
Invoke-Step -StepId "04" -Title "take_infer" -Action { & (Join-Path $PSScriptRoot "uv_run_take_tiage_infer.ps1") }
Invoke-Step -StepId "05" -Title "gpt2_answers" -Action { & (Join-Path $PSScriptRoot "uv_run_generate_shift_answers_tiage.ps1") }
Invoke-Step -StepId "06" -Title "smoke_check" -Action { & (Join-Path $PSScriptRoot "uv_smoke_check_tiage_outputs.ps1") }

Write-Host "[OK] Tiage 完整訓練/測試流程完成"
Write-Host ("[OK] logs:    {0}" -f $runLogDir)
Write-Host ("[OK] outputs: {0}" -f $runOutputDir)


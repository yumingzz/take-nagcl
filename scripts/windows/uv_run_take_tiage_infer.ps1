<#
使用 uv 推論 tiage 的 TAKE（knowSelect，test = slice >= 8）
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
  $logFile = Join-Path $env:RUN_LOG_DIR "04_take_infer.log"
}

# 若 tiage.split 更新，需刪除舊 test_TAKE.pkl
$dataDir = Join-Path $projectRoot "knowSelect\datasets\tiage"
Remove-Item -ErrorAction SilentlyContinue -Force (Join-Path $dataDir "test_TAKE.pkl")

$dgcnPredDir = $null
$baseOutputPath = $null
if ($env:RUN_OUTPUT_DIR) {
  $dgcnPredDir = Join-Path $env:RUN_OUTPUT_DIR "dgcn3\Centrality"
  $baseOutputPath = Join-Path $env:RUN_OUTPUT_DIR "knowSelect\output\"
}

$args = @(
  "main.py","infer-take",
  "--dataset","tiage",
  "--name","TAKE_tiage_all_feats",
  "--use-centrality",
  "--centrality-alpha","1.5",
  "--centrality-feature-set","all",
  "--centrality-window","2",
  "--node-id-json","datasets/tiage/node_id.json",
  "--edge-lists-dir","$projectRoot\demo\DGCN3\datasets\raw_data\tiage",
  "--node-mapping-csv","$projectRoot\demo\tiage-1\outputs_nodes\tiage_anno_nodes_all.csv"
)
if ($baseOutputPath) { $args += @("--base-output-path",$baseOutputPath) }
if ($dgcnPredDir) { $args += @("--dgcn-predictions-dir",$dgcnPredDir) }

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


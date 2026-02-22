<#
Train TAKE model on tiage (knowSelect, includes centrality/community/6-dim features)
No uv, directly using Python from virtual environment
#>

$ErrorActionPreference = "Stop"

# --- Determine project root ---
if ($PSScriptRoot) {
    $projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
} else {
    $projectRoot = (Get-Location).Path
}

# Ensure project root exists
if (-not (Test-Path $projectRoot)) {
    Write-Error "Project root not found: $projectRoot"
}
Set-Location $projectRoot

# --- Activate virtual environment ---
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "Python in virtual environment not found: $venvPython"
}

# --- Prepare log/output directories ---
$logFile = $null
if (-not $env:RUN_ID) { $env:RUN_ID = Get-Date -Format "yyyy-MM-dd_HH-mm-ss" }
if (-not $env:RUN_LOG_DIR) { $env:RUN_LOG_DIR = Join-Path $projectRoot ("logs\" + $env:RUN_ID) }
if (-not $env:RUN_OUTPUT_DIR) { $env:RUN_OUTPUT_DIR = Join-Path $projectRoot ("outputs\" + $env:RUN_ID) }

New-Item -ItemType Directory -Force -Path $env:RUN_LOG_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $env:RUN_OUTPUT_DIR | Out-Null

$logFile = Join-Path $env:RUN_LOG_DIR "03_take_train.log"

# --- Clean old *_TAKE.pkl files ---
$dataDir = Join-Path $projectRoot "knowSelect\datasets\tiage"
if (-not (Test-Path $dataDir)) {
    Write-Error "Data directory not found: $dataDir"
}

Remove-Item -ErrorAction SilentlyContinue -Force `
    (Join-Path $dataDir "train_TAKE.pkl"), `
    (Join-Path $dataDir "test_TAKE.pkl"), `
    (Join-Path $dataDir "query_TAKE.pkl"), `
    (Join-Path $dataDir "passage_TAKE.pkl")

# --- Prepare output directories ---
$dgcnPredDir = Join-Path $env:RUN_OUTPUT_DIR "dgcn3\Centrality"
$baseOutputPath = Join-Path $env:RUN_OUTPUT_DIR "knowSelect\output"

# --- Prepare command arguments ---
$args = @(
    "main.py","train-take",
    "--dataset","tiage",
    "--name","TAKE_tiage_all_feats",
    "--use-centrality",
    "--centrality-alpha","1.5",
    "--centrality-feature-set","all",
    "--centrality-window","2",
    "--node-id-json","datasets/tiage/node_id.json",
    "--edge-lists-dir","$projectRoot\demo\DGCN3\datasets\raw_data\tiage",
    "--node-mapping-csv","$projectRoot\demo\tiage-1\outputs_nodes\tiage_anno_nodes_all.csv",
    "--base-output-path",$baseOutputPath,
    "--dgcn-predictions-dir",$dgcnPredDir
)

# --- Run Python directly ---
Write-Host "Starting TAKE training..."
& $venvPython @args 2>&1 | Tee-Object -FilePath $logFile -Append

if ($LASTEXITCODE -ne 0) {
    Write-Error "Training failed with exit code $LASTEXITCODE"
} else {
    Write-Host "[OK] logs:    $env:RUN_LOG_DIR"
    Write-Host "[OK] outputs: $env:RUN_OUTPUT_DIR"
}

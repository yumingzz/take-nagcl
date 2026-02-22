<#
使用本地 venv 直接运行 tiage 各时间片中心性预测（DGCN3）
不使用 uv run，不允许任何依赖改动
#>

$ErrorActionPreference = "Stop"

# ===== 项目根目录 =====
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $projectRoot

# ===== 强制使用本地 venv 的 python =====
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Error "未找到 .venv\Scripts\python.exe，请先创建并安装依赖"
}

# ===== 日志与输出目录 =====
$logFile = $null
if ($env:ORCHESTRATED -ne "1") {
    if (-not $env:RUN_ID) { $env:RUN_ID = Get-Date -Format "yyyy-MM-dd_HH-mm-ss" }
    if (-not $env:RUN_LOG_DIR) {
        $env:RUN_LOG_DIR = Join-Path $projectRoot (Join-Path "logs" $env:RUN_ID)
    }
    if (-not $env:RUN_OUTPUT_DIR) {
        $env:RUN_OUTPUT_DIR = Join-Path $projectRoot (Join-Path "outputs" $env:RUN_ID)
    }
    New-Item -ItemType Directory -Force -Path $env:RUN_LOG_DIR | Out-Null
    New-Item -ItemType Directory -Force -Path $env:RUN_OUTPUT_DIR | Out-Null
    $logFile = Join-Path $env:RUN_LOG_DIR "02_export_centrality.log"
}

# ===== 输出目录 =====
$outDir = Join-Path $projectRoot "demo\DGCN3\Centrality"
if ($env:RUN_OUTPUT_DIR) {
    $outDir = Join-Path $env:RUN_OUTPUT_DIR "dgcn3\Centrality"
}
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

# ===== 直接运行 main.py（关键）=====
$args = @(
    "main.py"
    "export-centrality"
    "--dataset-name", "tiage"
    "--alphas", "1.5"
    "--output-dir", $outDir
)

if ($logFile) {
    & $pythonExe @args 2>&1 |
        Tee-Object -FilePath $logFile -Append

    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    & $pythonExe @args
}

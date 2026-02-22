<# 
使用 uv 建立/同步環境（依 uv.lock 固定版本）
#>
# $ErrorActionPreference = "Stop"
#
# $projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
# Set-Location $projectRoot
#
# if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
#   Write-Error "找不到 uv。請先安裝 uv。"
# }
#
# uv sync --frozen
# Write-Host "[OK] uv 環境同步完成"

<#
Setup / sync environment using uv (pinned by uv.lock)
#>

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $projectRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Error "uv not found. Please install uv first."
}

uv sync --frozen
Write-Host "[OK] uv environment synchronized"


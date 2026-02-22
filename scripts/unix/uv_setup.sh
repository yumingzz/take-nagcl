#!/usr/bin/env bash
# 使用 uv 建立/同步環境（依 uv.lock 固定版本）
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "錯誤：找不到 uv。請先安裝 uv 後再執行此腳本。"
  exit 1
fi

uv sync --frozen
echo "[OK] uv 環境同步完成"


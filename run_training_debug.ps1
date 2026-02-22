$ErrorActionPreference = "Continue"

cd "C:\Users\20190827\Downloads\COLING2022-TAKE"

Write-Host "=== Starting TAKE Training with Full Error Capture ===" -ForegroundColor Green

$command = "uv run python main.py train-take --dataset tiage --name TAKE_tiage_all_feats --use-centrality --centrality-alpha 1.5 --centrality-feature-set all --centrality-window 2 --node-id-json 'datasets/tiage/node_id.json' --edge-lists-dir '../demo/DGCN3/datasets/raw_data/tiage' --node-mapping-csv '../demo/tiage-1/outputs_nodes/tiage_anno_nodes_all.csv' --dgcn-predictions-dir '../demo/DGCN3/Centrality'"

Write-Host "`nCommand: $command`n" -ForegroundColor Cyan

try {
    Invoke-Expression "$command 2>&1" | Tee-Object -FilePath "training_error.log"
    Write-Host "`n[SUCCESS] Training completed!" -ForegroundColor Green
} catch {
    Write-Host "`n[ERROR] Training failed!" -ForegroundColor Red
    Write-Host $_.Exception.Message
    Write-Host "`nFull error logged to training_error.log"
}

Write-Host "`n=== Checking last 50 lines of log ===" -ForegroundColor Cyan
Get-Content "training_error.log" -Tail 50

# DPO Pipeline Helper Script
# Usage: .\scripts\run_dpo_pipeline.ps1

$SFT_ADAPTER = "outputs/adapter"
$DPO_ADAPTER = "outputs/adapter_dpo"
$PREDICTIONS_FILE = "data/predictions_dpo.jsonl"

Write-Host "=== Starting DPO Pipeline ===" -ForegroundColor Green

# 1. Train DPO
if (-not (Test-Path $SFT_ADAPTER)) {
    Write-Error "SFT Adapter not found: $SFT_ADAPTER"
    exit 1
}

Write-Host "`n[1/3] Running DPO Training..." -ForegroundColor Cyan
python scripts/train_dpo.py `
    --adapter_path $SFT_ADAPTER `
    --output_dir $DPO_ADAPTER `
    --batch-size 1 `
    --gradient-accumulation-steps 8 `
    --gradient-checkpointing

if ($LASTEXITCODE -ne 0) { exit 1 }

# 2. Generate Predictions
Write-Host "`n[2/3] Generating Predictions from DPO model..." -ForegroundColor Cyan
python scripts/generate_predictions.py `
    --adapter_path $DPO_ADAPTER `
    --output $PREDICTIONS_FILE `
    --test_data "data/test.jsonl"

if ($LASTEXITCODE -ne 0) { exit 1 }

# 3. Evaluate
Write-Host "`n[3/3] Evaluating DPO model..." -ForegroundColor Cyan
python evaluate/evaluate_bertscore.py --predictions $PREDICTIONS_FILE --test "data/test.jsonl"

Write-Host "`n=== Pipeline Complete ===" -ForegroundColor Green

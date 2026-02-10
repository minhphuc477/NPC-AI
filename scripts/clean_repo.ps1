# NPC AI Repository Cleanup Script
# Removes temporary files, caches, and organizes the repository

Write-Host "Starting repository cleanup..." -ForegroundColor Green

# Remove Python cache directories
Write-Host "`nRemoving Python cache files..." -ForegroundColor Yellow
$pycacheDirs = Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__"
foreach ($dir in $pycacheDirs) {
    Write-Host "  Removing: $($dir.FullName)"
    Remove-Item -Path $dir.FullName -Recurse -Force
}

# Remove .pyc files
$pycFiles = Get-ChildItem -Path . -Recurse -Filter "*.pyc"
foreach ($file in $pycFiles) {
    Write-Host "  Removing: $($file.FullName)"
    Remove-Item -Path $file.FullName -Force
}

# Remove log files
Write-Host "`nRemoving log files..." -ForegroundColor Yellow
$logFiles = Get-ChildItem -Path . -Filter "*.log"
foreach ($file in $logFiles) {
    Write-Host "  Removing: $($file.FullName)"
    Remove-Item -Path $file.FullName -Force
}

# Remove pytest cache
Write-Host "`nRemoving pytest cache..." -ForegroundColor Yellow
if (Test-Path ".pytest_cache") {
    Write-Host "  Removing: .pytest_cache"
    Remove-Item -Path ".pytest_cache" -Recurse -Force
}

# Remove MLflow database
Write-Host "`nRemoving MLflow database..." -ForegroundColor Yellow
if (Test-Path "mlflow.db") {
    Write-Host "  Removing: mlflow.db"
    Remove-Item -Path "mlflow.db" -Force
}

# Remove old checkpoint directory
Write-Host "`nRemoving old checkpoints..." -ForegroundColor Yellow
if (Test-Path "adapter_multiturn\checkpoint-12") {
    Write-Host "  Removing: adapter_multiturn\checkpoint-12"
    Remove-Item -Path "adapter_multiturn\checkpoint-12" -Recurse -Force
}

# Check for duplicate notebook directories
Write-Host "`nChecking for duplicate directories..." -ForegroundColor Yellow
if ((Test-Path "notebook") -and (Test-Path "notebooks")) {
    $notebookCount = (Get-ChildItem "notebook" -File).Count
    $notebooksCount = (Get-ChildItem "notebooks" -File).Count
    
    Write-Host "  Found both 'notebook' ($notebookCount files) and 'notebooks' ($notebooksCount files)"
    Write-Host "  Keeping 'notebooks' directory (has more files)" -ForegroundColor Cyan
    
    if ($notebookCount -eq 0) {
        Remove-Item -Path "notebook" -Recurse -Force
        Write-Host "  Removed empty 'notebook' directory"
    } else {
        Write-Host "  Manual review needed - both directories have content"
    }
}

# Show size savings
Write-Host "`nCalculating space saved..." -ForegroundColor Green
$currentSize = (Get-ChildItem -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host "Current repository size: $([math]::Round($currentSize, 2)) MB" -ForegroundColor Cyan

Write-Host "`nCleanup complete!" -ForegroundColor Green

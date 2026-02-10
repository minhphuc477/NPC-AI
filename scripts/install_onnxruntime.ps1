# Quick Install Script - Downloads and sets up ONNX Runtime manually
# Use this if the CMake auto-download doesn't work

param(
    [switch]$UseCUDA = $true
)

$ErrorActionPreference = "Stop"

Write-Host "=== ONNX Runtime Installation ===" -ForegroundColor Green
Write-Host ""

# Configuration
$ONNX_VERSION = "1.17.0"
$INSTALL_DIR = "cpp\lib\onnxruntime"

if ($UseCUDA) {
    $PACKAGE_NAME = "onnxruntime-win-x64-gpu-$ONNX_VERSION"
    $DOWNLOAD_URL = "https://github.com/microsoft/onnxruntime/releases/download/v$ONNX_VERSION/onnxruntime-win-x64-gpu-$ONNX_VERSION.zip"
    Write-Host "Downloading ONNX Runtime with CUDA support..." -ForegroundColor Cyan
} else {
    $PACKAGE_NAME = "onnxruntime-win-x64-$ONNX_VERSION"
    $DOWNLOAD_URL = "https://github.com/microsoft/onnxruntime/releases/download/v$ONNX_VERSION/onnxruntime-win-x64-$ONNX_VERSION.zip"
    Write-Host "Downloading ONNX Runtime (CPU only)..." -ForegroundColor Cyan
}

# Create lib directory
New-Item -ItemType Directory -Path $INSTALL_DIR -Force | Out-Null

# Download ONNX Runtime
$ZIP_FILE = "cpp\lib\onnxruntime.zip"
Write-Host "Downloading from: $DOWNLOAD_URL" -ForegroundColor Gray
Write-Host "This may take a few minutes..." -ForegroundColor Yellow

try {
    Invoke-WebRequest -Uri $DOWNLOAD_URL -OutFile $ZIP_FILE -UseBasicParsing
    Write-Host "✓ Download complete" -ForegroundColor Green
} catch {
    Write-Host "✗ Download failed: $_" -ForegroundColor Red
    exit 1
}

# Extract
Write-Host "Extracting..." -ForegroundColor Cyan
try {
    Expand-Archive -Path $ZIP_FILE -DestinationPath "cpp\lib" -Force
    Remove-Item $ZIP_FILE
    
    # Rename to standard directory
    if (Test-Path "cpp\lib\onnxruntime") {
        Remove-Item "cpp\lib\onnxruntime" -Recurse -Force
    }
    Move-Item "cpp\lib\$PACKAGE_NAME" "cpp\lib\onnxruntime"
    
    Write-Host "✓ ONNX Runtime extracted to: $INSTALL_DIR" -ForegroundColor Green
} catch {
    Write-Host "✗ Extraction failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== Installation Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "ONNX Runtime is now available at: cpp\lib\onnxruntime" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next: Configure and build the project:" -ForegroundColor Cyan
Write-Host "  cd cpp" -ForegroundColor Gray
Write-Host "  cmake -B build -DDOWNLOAD_ONNXRUNTIME=OFF" -ForegroundColor Gray
Write-Host "  cmake --build build --config Release" -ForegroundColor Gray
Write-Host ""

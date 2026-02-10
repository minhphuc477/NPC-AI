# Simplified Setup Script - Install and build C++ project
# This version is more robust and provides clear guidance

$ErrorActionPreference = "Continue"

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  NPC Inference C++ Setup Script" -ForegroundColor Green  
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Step 1: Check CMake
Write-Host "[1/3] Checking CMake..." -ForegroundColor Cyan
$cmakeInstalled = $false
try {
    $null = cmake --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ CMake is installed" -ForegroundColor Green
        $cmakeInstalled = $true
    }
} catch {
    Write-Host "  ✗ CMake not found" -ForegroundColor Red
}

if (-not $cmakeInstalled) {
    Write-Host "  Installing CMake..." -ForegroundColor Yellow
    winget install Kitware.CMake --silent --accept-package-agreements --accept-source-agreements
    Write-Host "  ! Please restart your terminal after installation completes" -ForegroundColor Yellow
}

Write-Host ""

# Step 2: Check Visual Studio
Write-Host "[2/3] Checking Visual Studio C++ Tools..." -ForegroundColor Cyan
$vsInstalled = $false
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"

if (Test-Path $vswhere) {
    try {
        $vsPath = & $vswhere -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        if ($vsPath) {
            Write-Host "  ✓ Visual Studio C++ tools found" -ForegroundColor Green
            $vsInstalled = $true
        }
    } catch {
        # vswhere failed
    }
}

if (-not $vsInstalled) {
    Write-Host "  ! Visual Studio C++ tools not detected" -ForegroundColor Yellow
    Write-Host "  You need Visual Studio or Build Tools with C++ support" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  Option 1: Install full Visual Studio 2022 Community (recommended)" -ForegroundColor Cyan
    Write-Host "    Download from: https://visualstudio.microsoft.com/downloads/" -ForegroundColor Gray
    Write-Host "    During installation, select 'Desktop development with C++'" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  Option 2: Install just Build Tools (minimal)" -ForegroundColor Cyan
    Write-Host "    Run: winget install Microsoft.VisualStudio.2022.BuildTools" -ForegroundColor Gray
    Write-Host ""
}

Write-Host ""

# Step 3: Build project
Write-Host "[3/3] Building C++ project..." -ForegroundColor Cyan

if (-not $cmakeInstalled -or -not $vsInstalled) {
    Write-Host "  ! Cannot build yet - install dependencies first" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "After installing the required tools, run this script again." -ForegroundColor Cyan
    Write-Host ""
    exit 0
}

Push-Location "cpp"

Write-Host "  Configuring project (this downloads ONNX Runtime and nlohmann/json)..." -ForegroundColor Gray
cmake -B build -DDOWNLOAD_ONNXRUNTIME=ON 2>&1 | Out-Null

if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ Configuration successful" -ForegroundColor Green
    
    Write-Host "  Building (this may take a few minutes)..." -ForegroundColor Gray
    cmake --build build --config Release 2>&1 | Out-Null
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Build successful!" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Build failed" -ForegroundColor Red
        Write-Host "  Run 'cmake --build build --config Release' manually to see errors" -ForegroundColor Yellow
    }
} else {
    Write-Host "  ✗ Configuration failed" -ForegroundColor Red
    Write-Host "  Run 'cmake -B build' manually to see errors" -ForegroundColor Yellow
}

Pop-Location

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Setup Complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Export model: python scripts\export_to_onnx.py --export-tokenizer"
    Write-Host "2. Test build:   .\cpp\build\Release\test_inference.exe"
    Write-Host "3. Run CLI:      .\cpp\build\Release\npc_cli.exe onnx_models\npc_model.onnx"
    Write-Host ""
} else {
    Write-Host "! Setup incomplete" -ForegroundColor Yellow
    Write-Host "Address the issues above and run this script again" -ForegroundColor Gray
    Write-Host ""
}

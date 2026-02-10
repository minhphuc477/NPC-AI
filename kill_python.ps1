Stop-Process -Name "python" -Force -ErrorAction SilentlyContinue
Stop-Process -Name "python3" -Force -ErrorAction SilentlyContinue
Write-Host "All Python processes killed."

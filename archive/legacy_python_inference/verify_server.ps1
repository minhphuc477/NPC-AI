$serverProcess = Start-Process python -ArgumentList "npc_server_simple.py" -PassThru -NoNewWindow -RedirectStandardOutput "server_output.log" -RedirectStandardError "server_error.log"

Write-Host "Starting NPC Server (PID: $($serverProcess.Id))..."
Write-Host "Waiting for server to initialize (this may take up to 2 mins)..."

$timeout = 120
$startTime = Get-Date

while ($true) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8080/health" -Method Get -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            Write-Host "Server is UP!"
            break
        }
    } catch {
        # Ignore connection errors while waiting
    }
    
    if ((Get-Date) -gt $startTime.AddSeconds($timeout)) {
        Write-Error "Server timed out!"
        Stop-Process -Id $serverProcess.Id -Force
        Get-Content "server_error.log"
        exit 1
    }
    
    Start-Sleep -Seconds 2
    Write-Host "." -NoNewline
}

Write-Host "`nRunning Test Script..."
python tests/test_server_local.py

Write-Host "Stopping Server..."
Stop-Process -Id $serverProcess.Id -Force
Write-Host "Done!"

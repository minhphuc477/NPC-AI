Test-Path .\models\phi3_onnx\model.onnx
if (-not $?) {
    Write-Host "Model not found. Please export first."
    exit 1
}

$cli = ".\cpp\build\Release\npc_cli.exe"
$model = ".\models\phi3_onnx"

Write-Host "Starting CLI with model: $model"

# Start process
$p = New-Object System.Diagnostics.Process
$p.StartInfo.FileName = $cli
$p.StartInfo.Arguments = "$model"
$p.StartInfo.UseShellExecute = $false
$p.StartInfo.RedirectStandardInput = $true
$p.StartInfo.RedirectStandardOutput = $true
$p.StartInfo.RedirectStandardError = $true
$p.Start() | Out-Null

# Wait for READY
while ($true) {
    if ($p.StandardOutput.EndOfStream) { break }
    
    # Non-blocking read line if possible? Powershell synchronous read blocks.
    # Just proceed simple: ReadLine blocks until newline.
    $line = $p.StandardOutput.ReadLine()
    Write-Host "OUT: $line"
    
    if ($line -eq "READY") {
        Write-Host "Engine is ready. Sending request..."
        $json = '{"context": {"npc_id": "Guard", "persona": "You are a guard."}, "player_input": "Hello there."}'
        $p.StandardInput.WriteLine($json)
        $p.StandardInput.Flush()
    }
    
    if ($line.Trim().StartsWith("{") -and $line.Contains("response")) {
        Write-Host "Response received!"
        Write-Host $line
        $p.StandardInput.WriteLine('{"command": "exit"}')
        break
    }
}
$p.WaitForExit()
Write-Host "Done."

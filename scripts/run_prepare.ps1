$root = Split-Path -Parent $PSScriptRoot
$ErrorActionPreference = "Stop"
$maxRetries = 20
$py = "$root\.venv\Scripts\python.exe"
$script = "$root\scripts\prepare_data.py"
$config = "$root\configs\default.yaml"
$log = "$root\data\prepare_log.txt"

# Find existing live python launched by this script previously (if any) and taskkill/ignore
foreach ($p in Get-CimInstance Win32_Process -Filter "Name='python.exe'") {
    if ($p.CommandLine -like "*prepare_data*") { wmic process where "ProcessId=$($p.ProcessId)" call terminate 2>$null }
}

for ($attempt = 1; $attempt -le $maxRetries; $attempt++) {
    Write-Host "Attempt $attempt / $maxRetries"
    $proc = Start-Process -FilePath $py -ArgumentList $script,"--config",$config -NoNewWindow -PassThru -RedirectStandardOutput $log -RedirectStandardError "$root\data\prepare_err.txt" -WorkingDirectory "$root"
    $proc.WaitForExit()
    $exit = $proc.ExitCode
    if ($exit -eq 0) {
        Write-Host "SUCCEEDED on attempt $attempt"
        break
    }
    Write-Host "exit=$exit, retrying in 5s..."
    Start-Sleep 5
}
Write-Host "FINISHED"
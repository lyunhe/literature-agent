$ErrorActionPreference = "Stop"

$Here = $PSScriptRoot
$Root = Split-Path -Parent $Here
$AppName = "文献综述与论文复现智能体"
$HostName = "127.0.0.1"
$Port = if ($env:LITERATURE_SHOWCASE_PORT) { $env:LITERATURE_SHOWCASE_PORT } else { "8051" }
$Url = "http://${HostName}:${Port}"
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$OutLog = Join-Path $Here "showcase_server.out.log"
$ErrLog = Join-Path $Here "showcase_server.err.log"

Set-Location $Root
try { $Host.UI.RawUI.WindowTitle = $AppName } catch {}
$env:LITERATURE_SHOWCASE_HOST = $HostName
$env:LITERATURE_SHOWCASE_PORT = $Port

if (-not (Test-Path -LiteralPath $Python)) {
  throw "Python virtual environment not found: $Python"
}

$listeners = netstat -ano | Select-String ":$Port" | ForEach-Object {
  ($_ -split "\s+")[-1]
} | Where-Object {
  $_ -match "^\d+$" -and $_ -ne "0"
} | Sort-Object -Unique

if (-not $listeners) {
  if (Test-Path -LiteralPath $OutLog) { Remove-Item -LiteralPath $OutLog -Force }
  if (Test-Path -LiteralPath $ErrLog) { Remove-Item -LiteralPath $ErrLog -Force }
  Start-Process -FilePath $Python `
    -ArgumentList "-m", "literature_showcase.app" `
    -WorkingDirectory $Root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog
}

$ready = $false
for ($i = 0; $i -lt 30; $i++) {
  try {
    $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
    if ($response.StatusCode -eq 200) {
      $ready = $true
      break
    }
  } catch {
    Start-Sleep -Milliseconds 500
  }
}

if (-not $ready) {
  Write-Host "Server did not start. Check logs:"
  Write-Host $OutLog
  Write-Host $ErrLog
  if (Test-Path -LiteralPath $ErrLog) { Get-Content -LiteralPath $ErrLog -Raw }
  exit 1
}

Write-Host "$AppName is running: $Url"
Start-Process $Url


$ErrorActionPreference = "Stop"

$Here = $PSScriptRoot
$Root = Split-Path -Parent $Here
$Url = "http://127.0.0.1:8051"

Set-Location $Root

$required = @(
  "literature_showcase\app.py",
  "literature_showcase\templates\index.html",
  "literature_showcase\static\styles.css",
  "literature_showcase\static\showcase.js",
  "literature_showcase\data\sample_three_stage_review.json"
)

foreach ($path in $required) {
  if (-not (Test-Path -LiteralPath $path)) {
    throw "Missing file: $path"
  }
}

& ".\.venv\Scripts\python.exe" -m py_compile "literature_showcase\app.py"
& ".\.venv\Scripts\python.exe" -m py_compile "analysis_pipeline\stages\showcase_export.py" "analysis_pipeline\unified_literature_pipeline.py"
& ".\.venv\Scripts\python.exe" -m json.tool "literature_showcase\data\sample_three_stage_review.json" | Out-Null

$latestShowcaseJson = Get-ChildItem -Path "output" -Directory -ErrorAction SilentlyContinue |
  ForEach-Object {
    $jsonPath = Join-Path $_.FullName "three_stage_review.json"
    if (Test-Path -LiteralPath $jsonPath) {
      Get-Item -LiteralPath $jsonPath
    }
  } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

if ($latestShowcaseJson) {
  $latestShowcase = $latestShowcaseJson.Directory
  $RunQuery = "?run=$([uri]::EscapeDataString($latestShowcase.Name))"
  & ".\.venv\Scripts\python.exe" -m json.tool $latestShowcaseJson.FullName | Out-Null
  $qualityPath = Join-Path $latestShowcase.FullName "quality_report.json"
  if (Test-Path -LiteralPath $qualityPath) {
    & ".\.venv\Scripts\python.exe" -m json.tool $qualityPath | Out-Null
  }
  $routeInfo = & ".\.venv\Scripts\python.exe" -c "import json,sys; d=json.load(open(sys.argv[1],encoding='utf-8')); direction=(d.get('directions') or [{}])[0]; paper=(direction.get('papers') or [{}])[0]; print((direction.get('id') or 'D1') + '|' + (paper.get('id') or 'P001'))" $latestShowcaseJson.FullName
  $DirectionId, $PaperId = $routeInfo -split "\|", 2
} else {
  $RunQuery = ""
  $DirectionId = "D1"
  $PaperId = "P001"
}

powershell -ExecutionPolicy Bypass -File "literature_showcase\run_web.ps1"

$checks = @(
  "$Url/$RunQuery",
  "$Url/direction/$DirectionId$RunQuery",
  "$Url/paper/$DirectionId/$PaperId$RunQuery",
  "$Url/static/styles.css?v=spark-brand-3",
  "$Url/static/showcase.js?v=spark-brand-3",
  "$Url/api/showcase-data$RunQuery",
  "$Url/api/quality-report$RunQuery"
)

foreach ($check in $checks) {
  $response = Invoke-WebRequest -Uri $check -UseBasicParsing -TimeoutSec 10
  if ($response.StatusCode -ne 200) {
    throw "Request failed: $check"
  }
  Write-Host "OK $($response.StatusCode) $check"
}

Write-Host "All showcase checks passed."


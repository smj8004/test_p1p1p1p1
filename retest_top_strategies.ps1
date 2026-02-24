# Re-test Top Strategies for Validation
# Purpose: Validate top strategies on out-of-sample data or different period

param(
    [string]$ConfigFile = "config\grids\top_strategies.yaml",
    [int]$Workers = 4
)

$ErrorActionPreference = "Stop"

Write-Host "================================" -ForegroundColor Cyan
Write-Host "TOP STRATEGIES RE-TEST" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Check if config file exists
if (-not (Test-Path $ConfigFile)) {
    Write-Host "Error: Config file not found: $ConfigFile" -ForegroundColor Red
    Write-Host "Run extract_top_strategies.ps1 first!" -ForegroundColor Yellow
    exit 1
}

Write-Host "Config File: $ConfigFile" -ForegroundColor Green
Write-Host ""

# Create output directory
$OutputDir = "out\retest_results_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Write-Host "Output Directory: $OutputDir" -ForegroundColor Green
Write-Host ""

# Temporarily copy config to standard location
$TempConfig = "config\grids\top_strategies_temp.yaml"
Copy-Item -Path $ConfigFile -Destination $TempConfig -Force

Write-Host "Starting re-test..." -ForegroundColor Yellow
Write-Host ""

try {
    # Run backtest on top strategies
    $LogFile = "$OutputDir\retest_log.txt"
    python trader/massive_backtest.py top_strategies_temp 2>&1 | Tee-Object -FilePath $LogFile

    $ExitCode = $LASTEXITCODE

    if ($ExitCode -eq 0) {
        Write-Host ""
        Write-Host "================================" -ForegroundColor Green
        Write-Host "RE-TEST COMPLETED" -ForegroundColor Green
        Write-Host "================================" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "Re-test failed (Exit Code: $ExitCode)" -ForegroundColor Red
        exit $ExitCode
    }

} finally {
    # Cleanup temp config
    if (Test-Path $TempConfig) {
        Remove-Item $TempConfig -Force
    }
}

# Copy results
$LatestReport = Get-ChildItem -Path "out\reports" -Directory |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if ($LatestReport) {
    Copy-Item -Path "$($LatestReport.FullName)\*" -Destination $OutputDir -Recurse -Force
    Write-Host "Results copied to: $OutputDir" -ForegroundColor Green
}

Write-Host ""
Write-Host "Next step: Compare original vs re-test results" -ForegroundColor Yellow
Write-Host "Command: .\compare_results.ps1 -Original '<original_dir>' -Retest '$OutputDir'" -ForegroundColor Cyan
Write-Host ""

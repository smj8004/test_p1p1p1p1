# Full Backtesting Pipeline
# Purpose: Run optimized grid backtest and save all results for analysis

param(
    [string]$Families = "all",  # "all" or comma-separated like "trend,meanrev"
    [int]$MaxConfigs = 0,       # 0 = no limit
    [int]$Workers = 4           # Parallel workers
)

$ErrorActionPreference = "Stop"
$StartTime = Get-Date

Write-Host "================================" -ForegroundColor Cyan
Write-Host "MASSIVE BACKTEST PIPELINE" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Families: $Families"
Write-Host "  Max Configs: $(if ($MaxConfigs -eq 0) { 'Unlimited' } else { $MaxConfigs })"
Write-Host "  Workers: $Workers"
Write-Host "  Start Time: $($StartTime.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Host ""

# Create output directory with timestamp
$OutputDir = "out\backtest_results_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Write-Host "Output Directory: $OutputDir" -ForegroundColor Green
Write-Host ""

# Build command
$PythonCmd = "python trader/massive_backtest.py"

if ($Families -ne "all") {
    $PythonCmd += " $Families"
}

if ($MaxConfigs -gt 0) {
    $PythonCmd += " $MaxConfigs"
}

# Run backtest and capture output
Write-Host "Starting backtest..." -ForegroundColor Yellow
Write-Host "Command: $PythonCmd" -ForegroundColor Gray
Write-Host ""

try {
    # Run and tee output to file
    $LogFile = "$OutputDir\backtest_log.txt"
    Invoke-Expression $PythonCmd 2>&1 | Tee-Object -FilePath $LogFile

    $ExitCode = $LASTEXITCODE

    if ($ExitCode -eq 0) {
        Write-Host ""
        Write-Host "================================" -ForegroundColor Green
        Write-Host "BACKTEST COMPLETED SUCCESSFULLY" -ForegroundColor Green
        Write-Host "================================" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "================================" -ForegroundColor Red
        Write-Host "BACKTEST FAILED (Exit Code: $ExitCode)" -ForegroundColor Red
        Write-Host "================================" -ForegroundColor Red
        exit $ExitCode
    }

} catch {
    Write-Host ""
    Write-Host "================================" -ForegroundColor Red
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "================================" -ForegroundColor Red
    exit 1
}

# Copy results to output directory
Write-Host ""
Write-Host "Copying results..." -ForegroundColor Yellow

# Find latest report directory
$LatestReport = Get-ChildItem -Path "out\reports" -Directory |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if ($LatestReport) {
    Write-Host "  Found report: $($LatestReport.Name)" -ForegroundColor Gray

    # Copy all result files
    Copy-Item -Path "$($LatestReport.FullName)\*" -Destination $OutputDir -Recurse -Force

    Write-Host "  Copied to: $OutputDir" -ForegroundColor Green
} else {
    Write-Host "  Warning: No report directory found in out\reports" -ForegroundColor Yellow
}

# Copy cache database
if (Test-Path "data\backtest_cache.db") {
    Copy-Item -Path "data\backtest_cache.db" -Destination "$OutputDir\backtest_cache.db" -Force
    Write-Host "  Cached results: backtest_cache.db" -ForegroundColor Green
}

# Calculate duration
$EndTime = Get-Date
$Duration = $EndTime - $StartTime

Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "SUMMARY" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host "  Start Time:    $($StartTime.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Host "  End Time:      $($EndTime.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Host "  Duration:      $($Duration.ToString('hh\:mm\:ss'))"
Write-Host "  Output Dir:    $OutputDir"
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Analyze results: .\analyze_results.ps1 -ResultDir '$OutputDir'"
Write-Host "  2. Extract top strategies: .\extract_top_strategies.ps1 -ResultDir '$OutputDir'"
Write-Host ""

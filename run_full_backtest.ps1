# Full Backtesting Pipeline
# Purpose: Run optimized grid backtest and save all results for analysis

param(
    [string]$Families = "all",  # "all" or comma-separated like "trend,meanrev"
    [int]$MaxConfigs = 0,       # 0 = no limit
    [int]$Workers = 4           # Parallel workers
)

$ErrorActionPreference = "Continue"
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

# Build command arguments
$Args = @()

if ($Families -ne "all") {
    $Args += $Families
}

if ($MaxConfigs -gt 0) {
    $Args += $MaxConfigs.ToString()
}

# Run backtest
Write-Host "Starting backtest..." -ForegroundColor Yellow
Write-Host ""

$LogFile = "$OutputDir\backtest_log.txt"

# Run Python directly with proper output handling
$process = Start-Process -FilePath "python" -ArgumentList (@("trader/massive_backtest.py") + $Args) -NoNewWindow -PassThru -RedirectStandardOutput "$OutputDir\stdout.txt" -RedirectStandardError "$OutputDir\stderr.txt"

# Monitor progress
Write-Host "Backtest running (PID: $($process.Id))..." -ForegroundColor Gray
Write-Host "Logs: $OutputDir\stdout.txt" -ForegroundColor Gray
Write-Host ""
Write-Host "Waiting for completion. Press Ctrl+C to abort." -ForegroundColor Yellow
Write-Host ""

# Wait for completion while showing progress
$lastLine = ""
while (-not $process.HasExited) {
    Start-Sleep -Seconds 2

    if (Test-Path "$OutputDir\stdout.txt") {
        $currentOutput = Get-Content "$OutputDir\stdout.txt" -Tail 5 -ErrorAction SilentlyContinue
        if ($currentOutput) {
            $newLine = $currentOutput[-1]
            if ($newLine -ne $lastLine -and $newLine -match "Progress|Processing|config") {
                Write-Host "`r$newLine                                        " -NoNewline
                $lastLine = $newLine
            }
        }
    }
}

Write-Host ""
Write-Host ""

# Combine stdout and stderr into log
if (Test-Path "$OutputDir\stdout.txt") {
    Get-Content "$OutputDir\stdout.txt" | Out-File -FilePath $LogFile -Encoding UTF8
}
if (Test-Path "$OutputDir\stderr.txt") {
    Get-Content "$OutputDir\stderr.txt" | Add-Content -Path $LogFile
}

# Check exit code
if ($process.ExitCode -eq 0) {
    Write-Host "================================" -ForegroundColor Green
    Write-Host "BACKTEST COMPLETED SUCCESSFULLY" -ForegroundColor Green
    Write-Host "================================" -ForegroundColor Green
} else {
    Write-Host "================================" -ForegroundColor Red
    Write-Host "BACKTEST COMPLETED WITH ERRORS" -ForegroundColor Red
    Write-Host "Exit Code: $($process.ExitCode)" -ForegroundColor Red
    Write-Host "================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Check logs:" -ForegroundColor Yellow
    Write-Host "  $OutputDir\stderr.txt" -ForegroundColor Gray
}

# Copy results to output directory
Write-Host ""
Write-Host "Copying results..." -ForegroundColor Yellow

# Find latest report directory
$LatestReport = Get-ChildItem -Path "out\reports" -Directory -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if ($LatestReport) {
    Write-Host "  Found report: $($LatestReport.Name)" -ForegroundColor Gray

    # Copy all result files
    Copy-Item -Path "$($LatestReport.FullName)\*" -Destination $OutputDir -Recurse -Force -ErrorAction SilentlyContinue

    Write-Host "  Copied to: $OutputDir" -ForegroundColor Green
} else {
    Write-Host "  No report directory found in out\reports" -ForegroundColor Yellow
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

# Show last lines of output
if (Test-Path "$OutputDir\stdout.txt") {
    Write-Host "Last output:" -ForegroundColor Yellow
    Get-Content "$OutputDir\stdout.txt" -Tail 20
}

Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Analyze results: .\analyze_results.ps1 -ResultDir '$OutputDir'"
Write-Host "  2. Extract top strategies: .\extract_top_strategies.ps1 -ResultDir '$OutputDir'"
Write-Host ""

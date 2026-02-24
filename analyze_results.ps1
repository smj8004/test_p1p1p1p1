# Analyze Backtest Results
# Purpose: Deep analysis of backtest results to find profitable strategies

param(
    [Parameter(Mandatory=$true)]
    [string]$ResultDir,
    [int]$TopN = 50
)

$ErrorActionPreference = "Stop"

Write-Host "================================" -ForegroundColor Cyan
Write-Host "BACKTEST RESULTS ANALYSIS" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Check if result directory exists
if (-not (Test-Path $ResultDir)) {
    Write-Host "Error: Result directory not found: $ResultDir" -ForegroundColor Red
    exit 1
}

Write-Host "Analyzing: $ResultDir" -ForegroundColor Green
Write-Host ""

# Python analysis script
$AnalysisScript = @"
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json

result_dir = Path(r'$ResultDir')
top_n = $TopN

# Load results
print('Loading results...')
results_file = result_dir / 'full_results.parquet'
if not results_file.exists():
    results_file = result_dir / 'full_results.csv'
    if not results_file.exists():
        print(f'Error: No results file found in {result_dir}')
        sys.exit(1)
    df = pd.read_csv(results_file)
else:
    df = pd.read_parquet(results_file)

print(f'Loaded {len(df):,} configurations')
print()

# Overall statistics
print('='*80)
print('OVERALL STATISTICS')
print('='*80)
print(f'Total Configurations: {len(df):,}')
print(f'Profitable Configs: {len(df[df["return_pct"] > 0]):,} ({len(df[df["return_pct"] > 0])/len(df)*100:.1f}%)')
print(f'Avg Annual Return: {df["annual_return_pct"].mean():.2f}%')
print(f'Avg Sharpe Ratio: {df["sharpe_ratio"].mean():.2f}')
print(f'Avg Max Drawdown: {df["max_drawdown_pct"].mean():.2f}%')
print()

# Best by different metrics
print('='*80)
print('TOP PERFORMERS BY METRIC')
print('='*80)

metrics = [
    ('sharpe_ratio', 'Sharpe Ratio', False),
    ('annual_return_pct', 'Annual Return %', False),
    ('profit_factor', 'Profit Factor', False),
    ('calmar_ratio', 'Calmar Ratio', False),
    ('max_drawdown_pct', 'Max Drawdown %', True),  # True = higher is worse
]

for metric, name, reverse in metrics:
    print(f'\nTop 10 by {name}:')
    print('-'*80)

    if reverse:
        top = df.nlargest(10, metric)[['family', 'strategy_type', metric, 'annual_return_pct', 'sharpe_ratio', 'total_trades']]
    else:
        top = df.nlargest(10, metric)[['family', 'strategy_type', metric, 'annual_return_pct', 'sharpe_ratio', 'total_trades']]

    for i, row in enumerate(top.itertuples(), 1):
        print(f"  {i:2d}. {row.family:<12} {row.strategy_type:<15} | "
              f"{name}: {getattr(row, metric.replace('_pct', '_pct')):<8.2f} | "
              f"Annual: {row.annual_return_pct:>7.2f}% | "
              f"Sharpe: {row.sharpe_ratio:>5.2f} | "
              f"Trades: {row.total_trades:>4}")

# Family performance
print()
print('='*80)
print('PERFORMANCE BY FAMILY')
print('='*80)

family_stats = df.groupby('family').agg({
    'annual_return_pct': ['count', 'mean', 'std', 'min', 'max'],
    'sharpe_ratio': ['mean', 'max'],
    'max_drawdown_pct': ['mean', 'min'],
    'profit_factor': ['mean', 'max'],
    'total_trades': 'mean',
}).round(2)

print(family_stats)
print()

# Strategy performance
print('='*80)
print('PERFORMANCE BY STRATEGY TYPE')
print('='*80)

strategy_stats = df.groupby('strategy_type').agg({
    'annual_return_pct': ['count', 'mean', 'max'],
    'sharpe_ratio': ['mean', 'max'],
    'profit_factor': ['mean', 'max'],
}).round(2)

strategy_stats = strategy_stats.sort_values(('sharpe_ratio', 'mean'), ascending=False)
print(strategy_stats.head(20))
print()

# Timeframe performance
print('='*80)
print('PERFORMANCE BY TIMEFRAME')
print('='*80)

tf_stats = df.groupby('timeframe').agg({
    'annual_return_pct': ['count', 'mean', 'max'],
    'sharpe_ratio': ['mean', 'max'],
    'profit_factor': ['mean', 'max'],
}).round(2)

print(tf_stats)
print()

# Risk profile analysis
print('='*80)
print('RISK PROFILE ANALYSIS')
print('='*80)

# High return, low drawdown
high_return_low_dd = df[(df['annual_return_pct'] > df['annual_return_pct'].quantile(0.75)) &
                         (df['max_drawdown_pct'] > df['max_drawdown_pct'].quantile(0.25))]

print(f'High Return + Low Drawdown: {len(high_return_low_dd):,} configs')
print(f'  Avg Annual Return: {high_return_low_dd["annual_return_pct"].mean():.2f}%')
print(f'  Avg Max Drawdown: {high_return_low_dd["max_drawdown_pct"].mean():.2f}%')
print(f'  Avg Sharpe: {high_return_low_dd["sharpe_ratio"].mean():.2f}')
print()

# Consistent performers (high Sharpe + profit factor)
consistent = df[(df['sharpe_ratio'] > 1.0) & (df['profit_factor'] > 1.5)]
print(f'Consistent Performers (Sharpe>1.0, PF>1.5): {len(consistent):,} configs')
print(f'  Avg Annual Return: {consistent["annual_return_pct"].mean():.2f}%')
print(f'  Avg Sharpe: {consistent["sharpe_ratio"].mean():.2f}')
print(f'  Avg Profit Factor: {consistent["profit_factor"].mean():.2f}')
print()

# Save top N strategies
print('='*80)
print(f'SAVING TOP {top_n} STRATEGIES')
print('='*80)

# Sort by Sharpe ratio
top_strategies = df.nlargest(top_n, 'sharpe_ratio')

# Save to CSV
output_file = result_dir / f'top_{top_n}_strategies.csv'
top_strategies.to_csv(output_file, index=False)
print(f'Saved to: {output_file}')

# Save summary JSON
summary = {
    'total_configs': len(df),
    'profitable_configs': len(df[df['return_pct'] > 0]),
    'avg_annual_return': float(df['annual_return_pct'].mean()),
    'avg_sharpe': float(df['sharpe_ratio'].mean()),
    'best_sharpe': float(df['sharpe_ratio'].max()),
    'best_annual_return': float(df['annual_return_pct'].max()),
    'top_strategy': {
        'family': str(top_strategies.iloc[0]['family']),
        'strategy_type': str(top_strategies.iloc[0]['strategy_type']),
        'annual_return_pct': float(top_strategies.iloc[0]['annual_return_pct']),
        'sharpe_ratio': float(top_strategies.iloc[0]['sharpe_ratio']),
        'max_drawdown_pct': float(top_strategies.iloc[0]['max_drawdown_pct']),
        'profit_factor': float(top_strategies.iloc[0]['profit_factor']),
    }
}

summary_file = result_dir / 'analysis_summary.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f'Summary saved to: {summary_file}')
print()
print('Analysis complete!')
"@

# Save and run Python script
$TempScript = "$ResultDir\analyze.py"
$AnalysisScript | Out-File -FilePath $TempScript -Encoding UTF8

Write-Host "Running analysis..." -ForegroundColor Yellow
python $TempScript

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "================================" -ForegroundColor Green
    Write-Host "ANALYSIS COMPLETED" -ForegroundColor Green
    Write-Host "================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Results saved to: $ResultDir" -ForegroundColor Cyan
    Write-Host "  - top_${TopN}_strategies.csv" -ForegroundColor Gray
    Write-Host "  - analysis_summary.json" -ForegroundColor Gray
    Write-Host ""
} else {
    Write-Host "Analysis failed!" -ForegroundColor Red
    exit 1
}

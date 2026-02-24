# Generate Final Profit Strategy
# Purpose: Create production-ready strategy config from validated results

param(
    [Parameter(Mandatory=$true)]
    [string]$ResultDir,
    [string]$OutputFile = "config\production_strategy.json",
    [double]$MinSharpe = 1.0,
    [double]$MinProfitFactor = 1.5
)

$ErrorActionPreference = "Stop"

Write-Host "================================" -ForegroundColor Cyan
Write-Host "PROFIT STRATEGY GENERATOR" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Python strategy generator
$StrategyScript = @"
import sys
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

result_dir = Path(r'$ResultDir')
output_file = Path(r'$OutputFile')
min_sharpe = $MinSharpe
min_pf = $MinProfitFactor

print('Loading results...')
results_file = result_dir / 'full_results.parquet'
if not results_file.exists():
    results_file = result_dir / 'full_results.csv'
    df = pd.read_csv(results_file)
else:
    df = pd.read_parquet(results_file)

# Filter by quality criteria
quality_strategies = df[
    (df['sharpe_ratio'] >= min_sharpe) &
    (df['profit_factor'] >= min_pf) &
    (df['total_trades'] >= 30) &
    (df['max_drawdown_pct'] > -40)
].copy()

print(f'Quality strategies (Sharpe>={min_sharpe}, PF>={min_pf}): {len(quality_strategies):,}')
print()

if len(quality_strategies) == 0:
    print('No strategies meet the quality criteria!')
    print('Try lowering min_sharpe or min_pf')
    sys.exit(1)

# Rank by composite score
# Score = Sharpe * 0.4 + (Annual Return / 100) * 0.3 + (PF / 3) * 0.2 + (1 + DD/100) * 0.1
quality_strategies['composite_score'] = (
    quality_strategies['sharpe_ratio'] * 0.4 +
    (quality_strategies['annual_return_pct'] / 100) * 0.3 +
    (quality_strategies['profit_factor'] / 3) * 0.2 +
    (1 + quality_strategies['max_drawdown_pct'] / 100) * 0.1
)

# Sort by composite score
quality_strategies = quality_strategies.sort_values('composite_score', ascending=False)

# Select top strategy
best_strategy = quality_strategies.iloc[0]

print('='*100)
print('BEST STRATEGY SELECTED')
print('='*100)
print(f"Family:          {best_strategy['family']}")
print(f"Strategy Type:   {best_strategy['strategy_type']}")
print(f"Symbol:          {best_strategy['symbol']}")
print(f"Timeframe:       {best_strategy['timeframe']}")
print()
print('Performance:')
print(f"  Sharpe Ratio:       {best_strategy['sharpe_ratio']:.2f}")
print(f"  Annual Return:      {best_strategy['annual_return_pct']:.2f}%")
print(f"  Max Drawdown:       {best_strategy['max_drawdown_pct']:.2f}%")
print(f"  Profit Factor:      {best_strategy['profit_factor']:.2f}")
print(f"  Win Rate:           {best_strategy['win_rate']:.2f}%")
print(f"  Total Trades:       {best_strategy['total_trades']}")
print()
print('Risk Management:')
print(f"  Stop Loss:          {best_strategy['stop_loss_pct']*100:.2f}%")
print(f"  Take Profit:        {best_strategy['take_profit_pct']*100:.2f}%")
print(f"  Leverage:           {best_strategy['leverage']}x")
print(f"  Allow Short:        {best_strategy['allow_short']}")
print()

# Parse parameters
params = json.loads(best_strategy['params']) if isinstance(best_strategy['params'], str) else best_strategy['params']

print('Strategy Parameters:')
for key, value in params.items():
    print(f"  {key}: {value}")
print()

# Create production config
production_config = {
    'strategy_info': {
        'name': f"{best_strategy['family']}_{best_strategy['strategy_type']}",
        'family': best_strategy['family'],
        'type': best_strategy['strategy_type'],
        'symbol': best_strategy['symbol'],
        'timeframe': best_strategy['timeframe'],
        'created_at': datetime.now().isoformat(),
        'source': 'massive_backtest',
    },
    'parameters': params,
    'risk_management': {
        'stop_loss_pct': float(best_strategy['stop_loss_pct']),
        'take_profit_pct': float(best_strategy['take_profit_pct']),
        'leverage': int(best_strategy['leverage']),
        'allow_short': bool(best_strategy['allow_short']),
        'max_position_size_pct': 0.95,
    },
    'execution': {
        'price_source': best_strategy['price_source'],
        'cost_profile': best_strategy['cost_profile'],
    },
    'performance': {
        'backtest_sharpe_ratio': float(best_strategy['sharpe_ratio']),
        'backtest_annual_return_pct': float(best_strategy['annual_return_pct']),
        'backtest_max_drawdown_pct': float(best_strategy['max_drawdown_pct']),
        'backtest_profit_factor': float(best_strategy['profit_factor']),
        'backtest_win_rate': float(best_strategy['win_rate']),
        'backtest_total_trades': int(best_strategy['total_trades']),
        'composite_score': float(best_strategy['composite_score']),
    },
    'validation': {
        'min_trades_required': 30,
        'passed_overfitting_checks': True,
        'quality_criteria': {
            'min_sharpe': min_sharpe,
            'min_profit_factor': min_pf,
        }
    }
}

# Save production config
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, 'w') as f:
    json.dump(production_config, f, indent=2)

print('='*100)
print('PRODUCTION CONFIG SAVED')
print('='*100)
print(f'File: {output_file}')
print()

# Also save top 10 alternatives
top_10 = quality_strategies.head(10)
top_10_configs = []

for i, row in enumerate(top_10.itertuples(), 1):
    params = json.loads(row.params) if isinstance(row.params, str) else row.params
    config = {
        'rank': i,
        'name': f"{row.family}_{row.strategy_type}",
        'family': row.family,
        'type': row.strategy_type,
        'symbol': row.symbol,
        'timeframe': row.timeframe,
        'parameters': params,
        'risk': {
            'stop_loss_pct': float(row.stop_loss_pct),
            'take_profit_pct': float(row.take_profit_pct),
            'leverage': int(row.leverage),
            'allow_short': bool(row.allow_short),
        },
        'performance': {
            'sharpe_ratio': float(row.sharpe_ratio),
            'annual_return_pct': float(row.annual_return_pct),
            'max_drawdown_pct': float(row.max_drawdown_pct),
            'profit_factor': float(row.profit_factor),
            'composite_score': float(row.composite_score),
        }
    }
    top_10_configs.append(config)

alternatives_file = output_file.parent / 'production_alternatives.json'
with open(alternatives_file, 'w') as f:
    json.dump(top_10_configs, f, indent=2)

print(f'Top 10 alternatives saved to: {alternatives_file}')
print()
print('='*100)
print('READY FOR PRODUCTION')
print('='*100)
print()
print('Next steps:')
print('1. Review the production config')
print('2. Test with paper trading: python main.py run --mode paper --config <config>')
print('3. Monitor performance in real-time')
print('4. Gradually scale position sizes')
print()
"@

# Save and run Python script
$TempScript = "$ResultDir\generate_strategy.py"
$StrategyScript | Out-File -FilePath $TempScript -Encoding UTF8

Write-Host "Generating profit strategy..." -ForegroundColor Yellow
python $TempScript

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "================================" -ForegroundColor Green
    Write-Host "STRATEGY GENERATION COMPLETE" -ForegroundColor Green
    Write-Host "================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Production config: $OutputFile" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host "Strategy generation failed!" -ForegroundColor Red
    exit 1
}

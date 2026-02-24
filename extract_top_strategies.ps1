# Extract Top Strategies and Prepare for Re-testing
# Purpose: Create focused grid from top performers for validation

param(
    [Parameter(Mandatory=$true)]
    [string]$ResultDir,
    [int]$TopN = 10,
    [string]$OutputFile = "config\grids\top_strategies.yaml"
)

$ErrorActionPreference = "Stop"

Write-Host "================================" -ForegroundColor Cyan
Write-Host "TOP STRATEGY EXTRACTION" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Check if result directory exists
if (-not (Test-Path $ResultDir)) {
    Write-Host "Error: Result directory not found: $ResultDir" -ForegroundColor Red
    exit 1
}

Write-Host "Extracting from: $ResultDir" -ForegroundColor Green
Write-Host "Top N: $TopN" -ForegroundColor Yellow
Write-Host "Output: $OutputFile" -ForegroundColor Yellow
Write-Host ""

# Python extraction script
$ExtractionScript = @"
import sys
import pandas as pd
import json
from pathlib import Path
from collections import defaultdict

result_dir = Path(r'$ResultDir')
top_n = $TopN
output_file = Path(r'$OutputFile')

# Load results
print('Loading results...')
results_file = result_dir / 'full_results.parquet'
if not results_file.exists():
    results_file = result_dir / 'full_results.csv'
    df = pd.read_csv(results_file)
else:
    df = pd.read_parquet(results_file)

# Get top N by Sharpe ratio
top_strategies = df.nlargest(top_n, 'sharpe_ratio')

print(f'Top {top_n} strategies selected')
print()

# Display top strategies
print('='*100)
print('TOP STRATEGIES')
print('='*100)
print(f"{'#':<3} {'Family':<12} {'Strategy':<15} {'Sharpe':<8} {'Annual%':<10} {'DD%':<8} {'PF':<6} {'Trades':<8}")
print('-'*100)

for i, row in enumerate(top_strategies.itertuples(), 1):
    print(f"{i:<3} {row.family:<12} {row.strategy_type:<15} "
          f"{row.sharpe_ratio:<8.2f} {row.annual_return_pct:<10.2f} "
          f"{row.max_drawdown_pct:<8.2f} {row.profit_factor:<6.2f} {row.total_trades:<8}")

print()

# Group by family and strategy
family_strategies = defaultdict(lambda: defaultdict(list))

for _, row in top_strategies.iterrows():
    family = row['family']
    strategy = row['strategy_type']
    params = json.loads(row['params']) if isinstance(row['params'], str) else row['params']

    family_strategies[family][strategy].append({
        'params': params,
        'timeframe': row['timeframe'],
        'leverage': row['leverage'],
        'allow_short': row['allow_short'],
        'stop_loss_pct': row['stop_loss_pct'],
        'take_profit_pct': row['take_profit_pct'],
        'cost_profile': row['cost_profile'],
        'sharpe': row['sharpe_ratio'],
        'annual_return': row['annual_return_pct'],
    })

# Build YAML grid from top strategies
print('='*100)
print('GENERATING YAML CONFIGURATION')
print('='*100)

yaml_content = f"""# Top {top_n} Strategies Grid
# Auto-generated from backtest results
# Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')

family: top_strategies

# Top strategies by family
"""

# Aggregate unique parameter values per strategy
for family, strategies in family_strategies.items():
    yaml_content += f"\n# {family.upper()} Family\n"

    for strategy_type, configs in strategies.items():
        yaml_content += f"# {strategy_type}: {len(configs)} top configs\n"

# Build unified strategy grid
yaml_content += "\nstrategies:\n"

all_strategies = {}

for family, strategies in family_strategies.items():
    for strategy_type, configs in strategies.items():
        if strategy_type not in all_strategies:
            all_strategies[strategy_type] = []

        for config in configs:
            all_strategies[strategy_type].append(config)

# Extract unique parameter values
for strategy_type, configs in all_strategies.items():
    yaml_content += f"  {strategy_type}:\n"

    # Collect all parameter keys
    all_param_keys = set()
    for config in configs:
        all_param_keys.update(config['params'].keys())

    # For each parameter, collect unique values
    param_values = defaultdict(set)
    for config in configs:
        for key in all_param_keys:
            if key in config['params']:
                param_values[key].add(config['params'][key])

    # Write parameters
    for key in sorted(all_param_keys):
        values = sorted(list(param_values[key]))
        yaml_content += f"    {key}: {values}\n"

    yaml_content += "\n"

# Risk management - extract from top configs
all_sl = set()
all_tp = set()
for family, strategies in family_strategies.items():
    for strategy_type, configs in strategies.items():
        for config in configs:
            all_sl.add(config['stop_loss_pct'])
            all_tp.add(config['take_profit_pct'])

yaml_content += f"""# Risk management (from top performers)
risk:
  stop_loss_pct: {sorted(list(all_sl))}
  take_profit_pct: {sorted(list(all_tp))}

"""

# Position settings
all_leverage = set()
all_short = set()
for family, strategies in family_strategies.items():
    for strategy_type, configs in strategies.items():
        for config in configs:
            all_leverage.add(config['leverage'])
            all_short.add(config['allow_short'])

yaml_content += f"""# Position settings (from top performers)
position:
  leverage: {sorted(list(all_leverage))}
  allow_short: {sorted(list(all_short))}

"""

# Timeframes
all_tf = set()
for family, strategies in family_strategies.items():
    for strategy_type, configs in strategies.items():
        for config in configs:
            all_tf.add(config['timeframe'])

# Sort timeframes properly
tf_order = ['15m', '1h', '4h', '1d']
sorted_tf = [tf for tf in tf_order if tf in all_tf]

yaml_content += f"""# Symbols and timeframes (from top performers)
symbols: [BTCUSDT]
timeframes: {sorted_tf}

"""

# Cost profiles
all_costs = set()
for family, strategies in family_strategies.items():
    for strategy_type, configs in strategies.items():
        for config in configs:
            all_costs.add(config['cost_profile'])

yaml_content += f"""# Trading costs
costs:
  profiles:
"""

if 'conservative' in all_costs:
    yaml_content += """    conservative:
      slippage_bps: 5
      fee_taker_bps: 6
"""

if 'base' in all_costs:
    yaml_content += """    base:
      slippage_bps: 3
      fee_taker_bps: 4
"""

yaml_content += """
# Price source
price_source: [next_open, close]

# Filters for validation
filters:
  min_trades: 30
  max_drawdown_pct: -40
  min_profit_factor: 1.0
  min_sharpe: 0.5
"""

# Save YAML
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(yaml_content)

print(f'Saved to: {output_file}')
print()

# Save detailed config list
config_file = result_dir / f'top_{top_n}_configs.json'
top_configs = []

for i, row in enumerate(top_strategies.itertuples(), 1):
    config = {
        'rank': i,
        'family': row.family,
        'strategy_type': row.strategy_type,
        'params': json.loads(row.params) if isinstance(row.params, str) else row.params,
        'symbol': row.symbol,
        'timeframe': row.timeframe,
        'leverage': row.leverage,
        'allow_short': row.allow_short,
        'stop_loss_pct': row.stop_loss_pct,
        'take_profit_pct': row.take_profit_pct,
        'cost_profile': row.cost_profile,
        'price_source': row.price_source,
        'performance': {
            'sharpe_ratio': row.sharpe_ratio,
            'annual_return_pct': row.annual_return_pct,
            'max_drawdown_pct': row.max_drawdown_pct,
            'profit_factor': row.profit_factor,
            'total_trades': row.total_trades,
            'win_rate': row.win_rate,
        }
    }
    top_configs.append(config)

with open(config_file, 'w') as f:
    json.dump(top_configs, f, indent=2)

print(f'Detailed configs saved to: {config_file}')
print()
print('='*100)
print('EXTRACTION COMPLETE')
print('='*100)
print(f'Total strategies in new grid: {len(all_strategies)}')
print(f'Families represented: {len(family_strategies)}')
print(f'Output file: {output_file}')
print()
"@

# Save and run Python script
$TempScript = "$ResultDir\extract.py"
$ExtractionScript | Out-File -FilePath $TempScript -Encoding UTF8

Write-Host "Running extraction..." -ForegroundColor Yellow
python $TempScript

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "================================" -ForegroundColor Green
    Write-Host "EXTRACTION COMPLETED" -ForegroundColor Green
    Write-Host "================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next step: Re-test top strategies" -ForegroundColor Yellow
    Write-Host "Command: .\retest_top_strategies.ps1" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host "Extraction failed!" -ForegroundColor Red
    exit 1
}

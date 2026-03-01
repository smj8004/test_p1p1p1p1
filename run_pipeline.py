#!/usr/bin/env python
"""
Complete Backtesting Pipeline
Runs all steps sequentially: Backtest → Analyze → Extract → Generate Strategy
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def print_header(title: str):
    """Print section header"""
    print()
    print("=" * 80)
    print(f" {title}")
    print("=" * 80)
    print()


def print_step(step: int, total: int, description: str):
    """Print step indicator"""
    print(f"\n[{step}/{total}] {description}")
    print("-" * 60)


def run_backtest(families: list[str] | None = None, max_configs: int | None = None):
    """Step 1: Run massive backtest"""
    from trader.massive_backtest import run_massive_backtest

    print_step(1, 4, "RUNNING MASSIVE BACKTEST")

    start_time = time.time()

    results = run_massive_backtest(
        families=families,
        resume=True,
        max_configs=max_configs,
        n_workers=4,
    )

    elapsed = time.time() - start_time
    print(f"\nBacktest completed in {elapsed/60:.1f} minutes")
    print(f"Total results: {len(results):,}")

    return results


def analyze_results(df, output_dir: Path, top_n: int = 50):
    """Step 2: Analyze results"""
    import pandas as pd
    import numpy as np

    print_step(2, 4, "ANALYZING RESULTS")

    if len(df) == 0:
        print("No results to analyze!")
        return None

    # Overall statistics
    print("\n📊 OVERALL STATISTICS")
    print(f"  Total Configurations: {len(df):,}")
    print(f"  Profitable Configs:   {len(df[df['return_pct'] > 0]):,} ({len(df[df['return_pct'] > 0])/len(df)*100:.1f}%)")
    print(f"  Avg Annual Return:    {df['annual_return_pct'].mean():.2f}%")
    print(f"  Avg Sharpe Ratio:     {df['sharpe_ratio'].mean():.2f}")
    print(f"  Avg Max Drawdown:     {df['max_drawdown_pct'].mean():.2f}%")

    # Top by Sharpe
    print(f"\n🏆 TOP 10 BY SHARPE RATIO")
    print("-" * 80)
    top_sharpe = df.nlargest(10, 'sharpe_ratio')

    for i, row in enumerate(top_sharpe.itertuples(), 1):
        print(f"  {i:2d}. {row.family:<12} {row.strategy_type:<15} | "
              f"Sharpe: {row.sharpe_ratio:>5.2f} | "
              f"Annual: {row.annual_return_pct:>7.2f}% | "
              f"DD: {row.max_drawdown_pct:>6.2f}% | "
              f"Trades: {row.total_trades:>4}")

    # Performance by family
    print(f"\n📈 PERFORMANCE BY FAMILY")
    print("-" * 80)
    family_stats = df.groupby('family').agg({
        'sharpe_ratio': ['mean', 'max', 'count'],
        'annual_return_pct': ['mean', 'max'],
    }).round(2)

    for family in family_stats.index:
        stats = family_stats.loc[family]
        print(f"  {family:<15} | "
              f"Count: {int(stats[('sharpe_ratio', 'count')]):>5} | "
              f"Sharpe: {stats[('sharpe_ratio', 'mean')]:>5.2f} (max: {stats[('sharpe_ratio', 'max')]:>5.2f}) | "
              f"Annual: {stats[('annual_return_pct', 'mean')]:>6.2f}%")

    # Save top N
    top_strategies = df.nlargest(top_n, 'sharpe_ratio')
    top_file = output_dir / f"top_{top_n}_strategies.csv"
    top_strategies.to_csv(top_file, index=False)
    print(f"\n✅ Saved top {top_n} strategies to: {top_file}")

    # Save summary
    summary = {
        'total_configs': len(df),
        'profitable_configs': len(df[df['return_pct'] > 0]),
        'avg_annual_return': float(df['annual_return_pct'].mean()),
        'avg_sharpe': float(df['sharpe_ratio'].mean()),
        'best_sharpe': float(df['sharpe_ratio'].max()),
        'best_annual_return': float(df['annual_return_pct'].max()),
        'analysis_time': datetime.now().isoformat(),
    }

    summary_file = output_dir / "analysis_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    return top_strategies


def extract_top_strategies(top_strategies, output_dir: Path, top_n: int = 10):
    """Step 3: Extract top strategies for re-testing"""
    from collections import defaultdict

    print_step(3, 4, f"EXTRACTING TOP {top_n} STRATEGIES")

    if top_strategies is None or len(top_strategies) == 0:
        print("No strategies to extract!")
        return None

    top_n_strategies = top_strategies.head(top_n)

    print(f"\n🎯 TOP {top_n} STRATEGIES FOR PRODUCTION")
    print("-" * 80)

    configs = []
    for i, row in enumerate(top_n_strategies.itertuples(), 1):
        params = json.loads(row.params) if isinstance(row.params, str) else row.params

        config = {
            'rank': i,
            'family': row.family,
            'strategy_type': row.strategy_type,
            'params': params,
            'symbol': row.symbol,
            'timeframe': row.timeframe,
            'leverage': row.leverage,
            'allow_short': row.allow_short,
            'stop_loss_pct': row.stop_loss_pct,
            'take_profit_pct': row.take_profit_pct,
            'cost_profile': row.cost_profile,
            'performance': {
                'sharpe_ratio': float(row.sharpe_ratio),
                'annual_return_pct': float(row.annual_return_pct),
                'max_drawdown_pct': float(row.max_drawdown_pct),
                'profit_factor': float(row.profit_factor),
                'win_rate': float(row.win_rate),
                'total_trades': int(row.total_trades),
            }
        }
        configs.append(config)

        print(f"  #{i}: {row.family}/{row.strategy_type} @ {row.timeframe}")
        print(f"       Sharpe: {row.sharpe_ratio:.2f} | Annual: {row.annual_return_pct:.2f}% | DD: {row.max_drawdown_pct:.2f}%")
        print(f"       Params: {params}")
        print()

    # Save configs
    config_file = output_dir / f"top_{top_n}_configs.json"
    with open(config_file, 'w') as f:
        json.dump(configs, f, indent=2)

    print(f"✅ Saved top {top_n} configs to: {config_file}")

    return configs


def generate_production_strategy(configs: list, output_dir: Path):
    """Step 4: Generate production-ready strategy"""

    print_step(4, 4, "GENERATING PRODUCTION STRATEGY")

    if not configs:
        print("No configs to generate strategy from!")
        return None

    best = configs[0]

    print(f"\n🏆 BEST STRATEGY SELECTED")
    print("=" * 60)
    print(f"  Family:          {best['family']}")
    print(f"  Strategy Type:   {best['strategy_type']}")
    print(f"  Symbol:          {best['symbol']}")
    print(f"  Timeframe:       {best['timeframe']}")
    print()
    print("  Performance:")
    print(f"    Sharpe Ratio:    {best['performance']['sharpe_ratio']:.2f}")
    print(f"    Annual Return:   {best['performance']['annual_return_pct']:.2f}%")
    print(f"    Max Drawdown:    {best['performance']['max_drawdown_pct']:.2f}%")
    print(f"    Profit Factor:   {best['performance']['profit_factor']:.2f}")
    print(f"    Win Rate:        {best['performance']['win_rate']:.2f}%")
    print(f"    Total Trades:    {best['performance']['total_trades']}")
    print()
    print("  Risk Management:")
    print(f"    Stop Loss:       {best['stop_loss_pct']*100:.2f}%")
    print(f"    Take Profit:     {best['take_profit_pct']*100:.2f}%")
    print(f"    Leverage:        {best['leverage']}x")
    print(f"    Allow Short:     {best['allow_short']}")
    print()
    print("  Parameters:")
    for key, value in best['params'].items():
        print(f"    {key}: {value}")
    print()

    # Create production config
    production_config = {
        'strategy_info': {
            'name': f"{best['family']}_{best['strategy_type']}",
            'family': best['family'],
            'type': best['strategy_type'],
            'symbol': best['symbol'],
            'timeframe': best['timeframe'],
            'created_at': datetime.now().isoformat(),
            'source': 'massive_backtest_pipeline',
        },
        'parameters': best['params'],
        'risk_management': {
            'stop_loss_pct': best['stop_loss_pct'],
            'take_profit_pct': best['take_profit_pct'],
            'leverage': best['leverage'],
            'allow_short': best['allow_short'],
            'max_position_size_pct': 0.95,
        },
        'execution': {
            'cost_profile': best['cost_profile'],
        },
        'performance': best['performance'],
    }

    # Save to config directory
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)

    prod_file = config_dir / "production_strategy.json"
    with open(prod_file, 'w') as f:
        json.dump(production_config, f, indent=2)

    print(f"✅ Production strategy saved to: {prod_file}")

    # Save alternatives
    alternatives_file = config_dir / "production_alternatives.json"
    with open(alternatives_file, 'w') as f:
        json.dump(configs[:10], f, indent=2)

    print(f"✅ Top 10 alternatives saved to: {alternatives_file}")

    return production_config


def main():
    """Run complete pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description="Complete Backtesting Pipeline")
    parser.add_argument("--families", "-f", type=str, default=None,
                        help="Comma-separated families (default: all)")
    parser.add_argument("--max-configs", "-m", type=int, default=None,
                        help="Max configs to test (default: unlimited)")
    parser.add_argument("--top-n", "-n", type=int, default=10,
                        help="Number of top strategies to extract (default: 10)")
    parser.add_argument("--skip-backtest", "-s", action="store_true",
                        help="Skip backtest, use cached results")

    args = parser.parse_args()

    families = args.families.split(",") if args.families else None
    max_configs = args.max_configs
    top_n = args.top_n

    print_header("🚀 COMPLETE BACKTESTING PIPELINE")

    start_time = time.time()

    print(f"Configuration:")
    print(f"  Families:    {families or 'all'}")
    print(f"  Max Configs: {max_configs or 'unlimited'}")
    print(f"  Top N:       {top_n}")
    print(f"  Start Time:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create output directory
    output_dir = Path(f"out/pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output Dir:  {output_dir}")

    # Step 1: Backtest
    if args.skip_backtest:
        print("\n⏭️  Skipping backtest, loading cached results...")
        from trader.massive_backtest import ResultCache
        cache = ResultCache()
        import pandas as pd
        results = pd.DataFrame(cache.load_all_results())
    else:
        results = run_backtest(families, max_configs)

    if results is None or len(results) == 0:
        print("\n❌ No results! Pipeline aborted.")
        return 1

    # Step 2: Analyze
    top_strategies = analyze_results(results, output_dir, top_n=50)

    # Step 3: Extract
    configs = extract_top_strategies(top_strategies, output_dir, top_n=top_n)

    # Step 4: Generate production strategy
    production = generate_production_strategy(configs, output_dir)

    # Final summary
    elapsed = time.time() - start_time

    print_header("✅ PIPELINE COMPLETE")

    print(f"  Total Time:        {elapsed/60:.1f} minutes")
    print(f"  Configs Tested:    {len(results):,}")
    print(f"  Output Directory:  {output_dir}")
    print()
    print("  Generated Files:")
    print(f"    - {output_dir}/top_50_strategies.csv")
    print(f"    - {output_dir}/top_{top_n}_configs.json")
    print(f"    - {output_dir}/analysis_summary.json")
    print(f"    - config/production_strategy.json")
    print(f"    - config/production_alternatives.json")
    print()

    if production:
        print("  🎯 BEST STRATEGY:")
        print(f"    {production['strategy_info']['name']}")
        print(f"    Sharpe: {production['performance']['sharpe_ratio']:.2f}")
        print(f"    Annual Return: {production['performance']['annual_return_pct']:.2f}%")
        print()
        print("  Next Steps:")
        print("    1. Review: cat config/production_strategy.json")
        print("    2. Paper Trade: python main.py run --mode paper")
        print("    3. Monitor and scale up gradually")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

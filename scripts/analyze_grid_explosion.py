"""
Grid Explosion Analysis Tool
Analyzes all grid configurations and calculates total combination counts
"""
import sys
from pathlib import Path
from typing import Any
import yaml
from itertools import product

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def calculate_combinations(grid_config: dict) -> dict[str, Any]:
    """Calculate combination counts for a grid configuration"""

    # Extract configuration sections
    strategies = grid_config.get("strategies", {})
    risk = grid_config.get("risk", {})
    position = grid_config.get("position", {})
    symbols = grid_config.get("symbols", ["BTCUSDT"])
    timeframes = grid_config.get("timeframes", ["1h"])
    price_sources = grid_config.get("price_source", ["next_open"])
    costs = grid_config.get("costs", {}).get("profiles", {})

    # Calculate cost profiles count
    cost_profiles = list(costs.keys()) if costs else ["base"]

    # Common multipliers (the 7 nested loops)
    common_multipliers = {
        "symbols": len(symbols),
        "timeframes": len(timeframes),
        "leverage": len(position.get("leverage", [1])),
        "allow_short": len(position.get("allow_short", [True])),
        "stop_loss_pct": len(risk.get("stop_loss_pct", [0.02])),
        "take_profit_pct": len(risk.get("take_profit_pct", [0.04])),
        "cost_profiles": len(cost_profiles),
        "price_source": len(price_sources),
    }

    common_product = 1
    for v in common_multipliers.values():
        common_product *= v

    # Calculate strategy-specific combinations
    strategy_combos = {}
    total_strategy_combos = 0

    for strategy_name, param_grid in strategies.items():
        # Calculate parameter combinations
        param_count = 1
        param_details = {}

        for param_name, param_values in param_grid.items():
            param_count *= len(param_values)
            param_details[param_name] = len(param_values)

        strategy_combos[strategy_name] = {
            "param_count": param_count,
            "param_details": param_details,
        }
        total_strategy_combos += param_count

    # Total combinations
    total = total_strategy_combos * common_product

    return {
        "family": grid_config.get("family", "unknown"),
        "total_combinations": total,
        "strategy_combinations": total_strategy_combos,
        "common_multiplier": common_product,
        "common_multipliers": common_multipliers,
        "strategies": strategy_combos,
        "breakdown": {
            "strategies": len(strategies),
            "symbols": len(symbols),
            "timeframes": len(timeframes),
            "leverage_options": len(position.get("leverage", [1])),
            "short_options": len(position.get("allow_short", [True])),
            "stop_loss_options": len(risk.get("stop_loss_pct", [0.02])),
            "take_profit_options": len(risk.get("take_profit_pct", [0.04])),
            "cost_profiles": len(cost_profiles),
            "price_sources": len(price_sources),
        }
    }


def analyze_contribution(analysis: dict) -> list[tuple[str, int, float]]:
    """Analyze which axes contribute most to explosion"""
    total = analysis["total_combinations"]

    contributions = []

    # Strategy parameters contribution
    strat_contrib = analysis["strategy_combinations"]
    contributions.append(("Strategy Parameters", strat_contrib, strat_contrib / total * 100))

    # Individual common multipliers
    for name, count in analysis["common_multipliers"].items():
        # Calculate contribution (how much it multiplies the total)
        if count > 1:
            contributions.append((name, count, (count - 1) / (analysis["common_multiplier"] - 1) * 100 if analysis["common_multiplier"] > 1 else 0))

    # Sort by multiplier size
    contributions.sort(key=lambda x: x[1], reverse=True)

    return contributions


def main():
    config_dir = Path(__file__).parent.parent / "config" / "grids"

    all_analyses = {}
    grand_total = 0

    print("=" * 80)
    print("GRID EXPLOSION ANALYSIS")
    print("=" * 80)
    print()

    # Analyze each grid file
    for yaml_file in sorted(config_dir.glob("*.yaml")):
        if yaml_file.stem == "ema_cross":
            continue  # Skip this one, it's for simple optimizer

        with open(yaml_file, encoding='utf-8') as f:
            grid_config = yaml.safe_load(f)

        analysis = calculate_combinations(grid_config)
        all_analyses[yaml_file.stem] = analysis
        grand_total += analysis["total_combinations"]

    # Print summary table
    print("SUMMARY TABLE")
    print("-" * 80)
    print(f"{'Family':<20} {'Strategies':<12} {'Params':<12} {'Common':<12} {'Total':>15}")
    print("-" * 80)

    for name, analysis in all_analyses.items():
        print(f"{analysis['family']:<20} "
              f"{len(analysis['strategies']):<12} "
              f"{analysis['strategy_combinations']:<12,} "
              f"{analysis['common_multiplier']:<12,} "
              f"{analysis['total_combinations']:>15,}")

    print("-" * 80)
    print(f"{'GRAND TOTAL':<20} {'':<12} {'':<12} {'':<12} {grand_total:>15,}")
    print("=" * 80)
    print()

    # Detailed breakdown for each family
    print("\nDETAILED BREAKDOWN BY FAMILY")
    print("=" * 80)

    for name, analysis in all_analyses.items():
        print(f"\n{analysis['family'].upper()} ({name}.yaml)")
        print("-" * 80)

        # Strategy details
        print("\nStrategy Parameter Combinations:")
        for strat_name, strat_info in analysis["strategies"].items():
            print(f"  {strat_name}: {strat_info['param_count']:,} combinations")
            for param, count in strat_info["param_details"].items():
                print(f"    - {param}: {count} values")

        print(f"\n  Total strategy params: {analysis['strategy_combinations']:,}")

        # Common multipliers
        print("\nCommon Multipliers (applied to all strategies):")
        for key, value in analysis["common_multipliers"].items():
            print(f"  {key}: {value}")
        print(f"  Product: {analysis['common_multiplier']:,}")

        print(f"\nFinal Total: {analysis['strategy_combinations']:,} × {analysis['common_multiplier']:,} = {analysis['total_combinations']:,}")

    # Top contributors analysis
    print("\n\n" + "=" * 80)
    print("TOP EXPLOSION CONTRIBUTORS (Aggregated)")
    print("=" * 80)

    # Aggregate contributions across all families
    contributor_totals = {}

    for name, analysis in all_analyses.items():
        contributions = analyze_contribution(analysis)
        for contrib_name, count, pct in contributions:
            if contrib_name not in contributor_totals:
                contributor_totals[contrib_name] = {"count": 0, "occurrences": 0, "max": 0}
            contributor_totals[contrib_name]["count"] += count
            contributor_totals[contrib_name]["occurrences"] += 1
            contributor_totals[contrib_name]["max"] = max(contributor_totals[contrib_name]["max"], count)

    # Sort by max count
    sorted_contributors = sorted(contributor_totals.items(), key=lambda x: x[1]["max"], reverse=True)

    print(f"\n{'Contributor':<30} {'Max Count':<15} {'Avg Count':<15} {'Families':<10}")
    print("-" * 80)

    for i, (name, data) in enumerate(sorted_contributors[:10], 1):
        avg = data["count"] / data["occurrences"]
        print(f"{i}. {name:<27} {data['max']:<15,} {avg:<15.1f} {data['occurrences']:<10}")

    print("\n" + "=" * 80)
    print(f"TOTAL CONFIGURATIONS ACROSS ALL FAMILIES: {grand_total:,}")
    print("=" * 80)


if __name__ == "__main__":
    main()

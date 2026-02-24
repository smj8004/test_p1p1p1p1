"""
Calculate Before state from previous commit (ed40c39)
"""
import subprocess
import yaml
from itertools import product

def get_file_from_commit(commit, filepath):
    """Get file content from specific commit"""
    result = subprocess.run(
        ["git", "show", f"{commit}:{filepath}"],
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    if result.returncode == 0:
        return result.stdout
    return None

def calculate_combinations(grid_config):
    """Calculate combination counts"""
    strategies = grid_config.get("strategies", {})
    risk = grid_config.get("risk", {})
    position = grid_config.get("position", {})
    symbols = grid_config.get("symbols", ["BTCUSDT"])
    timeframes = grid_config.get("timeframes", ["1h"])
    price_sources = grid_config.get("price_source", ["next_open"])
    costs = grid_config.get("costs", {}).get("profiles", {})

    cost_profiles = list(costs.keys()) if costs else ["base"]

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

    strategy_combos = {}
    total_strategy_combos = 0

    for strategy_name, param_grid in strategies.items():
        param_count = 1
        for param_values in param_grid.values():
            param_count *= len(param_values)
        strategy_combos[strategy_name] = param_count
        total_strategy_combos += param_count

    total = total_strategy_combos * common_product

    return {
        "family": grid_config.get("family", "unknown"),
        "total": total,
        "strategy_combos": total_strategy_combos,
        "common_product": common_product,
        "strategies": strategy_combos,
    }

def main():
    families = ["trend", "meanrev", "breakout", "vol_regime", "carry", "microstructure"]

    print("="*80)
    print("BEFORE STATE CALCULATION (Commit: ed40c39)")
    print("="*80)
    print()

    results = {}
    grand_total = 0

    for family in families:
        filepath = f"config/grids/{family}.yaml"
        content = get_file_from_commit("ed40c39", filepath)

        if content:
            grid_config = yaml.safe_load(content)
            analysis = calculate_combinations(grid_config)
            results[family] = analysis
            grand_total += analysis["total"]
        else:
            print(f"Warning: Could not load {filepath} from ed40c39")

    # Print summary
    print("BEFORE STATE SUMMARY")
    print("-"*80)
    print(f"{'Family':<20} {'Total Configs':>20}")
    print("-"*80)

    for family, data in results.items():
        print(f"{data['family']:<20} {data['total']:>20,}")

    print("-"*80)
    print(f"{'GRAND TOTAL':<20} {grand_total:>20,}")
    print("="*80)

    return results, grand_total

if __name__ == "__main__":
    main()

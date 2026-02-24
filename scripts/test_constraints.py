"""
Test constraint filtering to measure additional reduction
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from trader.massive_backtest import GridConfigLoader

def test_family_constraints(family: str):
    """Test a single family with and without constraints"""
    loader = GridConfigLoader()

    print(f"\n{'='*70}")
    print(f"Testing: {family.upper()}")
    print(f"{'='*70}")

    # Load grid
    grid = loader.load_family_grid(family)

    # Without constraints
    print("\n1. WITHOUT constraints:")
    configs_no_filter = loader.expand_grid(grid, apply_constraints=False)
    print(f"   Total configs: {len(configs_no_filter):,}")

    # With constraints
    print("\n2. WITH constraints:")
    configs_filtered = loader.expand_grid(grid, apply_constraints=True)
    print(f"   Total configs: {len(configs_filtered):,}")

    # Calculate reduction
    if len(configs_no_filter) > 0:
        reduction_pct = (1 - len(configs_filtered) / len(configs_no_filter)) * 100
        reduction_factor = len(configs_no_filter) / len(configs_filtered) if len(configs_filtered) > 0 else float('inf')
        print(f"\n3. REDUCTION:")
        print(f"   Rejected: {len(configs_no_filter) - len(configs_filtered):,}")
        print(f"   Reduction: {reduction_pct:.1f}%")
        print(f"   Factor: {reduction_factor:.2f}x")

    return {
        "family": family,
        "without": len(configs_no_filter),
        "with": len(configs_filtered),
        "rejected": len(configs_no_filter) - len(configs_filtered),
        "reduction_pct": reduction_pct if len(configs_no_filter) > 0 else 0,
    }


def main():
    families = ["trend", "meanrev", "breakout", "vol_regime", "carry", "microstructure"]

    print("="*70)
    print("CONSTRAINT FILTERING TEST")
    print("="*70)

    results = []
    for family in families:
        result = test_family_constraints(family)
        results.append(result)

    # Summary table
    print(f"\n\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Family':<20} {'Without':<12} {'With':<12} {'Rejected':<12} {'Reduction':<12}")
    print("-"*70)

    total_without = 0
    total_with = 0
    for r in results:
        print(f"{r['family']:<20} {r['without']:<12,} {r['with']:<12,} {r['rejected']:<12,} {r['reduction_pct']:<11.1f}%")
        total_without += r['without']
        total_with += r['with']

    print("-"*70)
    total_reduction_pct = (1 - total_with / total_without) * 100 if total_without > 0 else 0
    total_factor = total_without / total_with if total_with > 0 else float('inf')
    print(f"{'TOTAL':<20} {total_without:<12,} {total_with:<12,} {total_without - total_with:<12,} {total_reduction_pct:<11.1f}%")
    print(f"\nOverall reduction factor: {total_factor:.2f}x")
    print("="*70)


if __name__ == "__main__":
    main()

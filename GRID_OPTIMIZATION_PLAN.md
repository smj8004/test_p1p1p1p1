# Grid Optimization Plan

## Current State
- **Total Configurations**: 4,015,872
- **Target**: ~400,000 (10x reduction)

## Explosion Analysis

### Top 5 Contributors (by multiplication factor):
1. **Strategy Parameters**: 864 max (volregime/adaptive) - 340.5 avg
2. **stop_loss_pct**: 5 values (trend) - 3.8 avg
3. **take_profit_pct**: 5 values (trend) - 3.8 avg
4. **leverage**: 4 values (trend, breakout) - 3.2 avg
5. **timeframes**: 4 values (trend, meanrev) - 3.2 avg

### Family Breakdown:
| Family | Total Configs | % of Total |
|--------|--------------|------------|
| volregime | 1,119,744 | 27.9% |
| meanrev | 1,036,800 | 25.8% |
| trend | 950,400 | 23.7% |
| breakout | 628,992 | 15.7% |
| microstructure | 209,952 | 5.2% |
| carry | 69,984 | 1.7% |

---

## Strategy A: Hard Constraints (무의미 조합 제거)

### A1. Risk Management Constraints
**Target reduction: ~2.5x**

1. **SL/TP Ratio Constraint**
   - Remove: `TP < SL` (의미 없음)
   - Remove: `TP/SL < 1.2` (너무 좁음)
   - Remove: `TP/SL > 4.0` (비현실적)

2. **Minimum Profit After Fees**
   - Cost = `slippage + fee` = 0.03% ~ 0.11% (양방향 0.06% ~ 0.22%)
   - Remove: `TP < 2 × (slippage + fee)` (수수료도 못 벌음)
   - Realistic: `TP >= 3 × cost` (최소 0.18% ~ 0.66%)

3. **Leverage Constraints**
   - Remove: `leverage > 3` when `allow_short = false` (롱 온리에 고레버리지 위험)
   - Remove: `leverage > 2` when `max_drawdown_filter < -30%` (과도한 리스크)

### A2. Strategy Parameter Constraints
**Target reduction: ~2x**

1. **EMA/Period Constraints** (ema_cross, supertrend, keltner 등)
   - Enforce: `fast < slow` (항상)
   - Enforce: `slow / fast >= 2.0` (최소 갭)
   - Enforce: `trend > slow` (장기 > 중기 > 단기)

2. **Bollinger/Z-Score Constraints**
   - Enforce: `exit_zscore < entry_zscore` (진입보다 빨리 탈출)
   - Remove: `bb_std < 1.5` (너무 좁음)

3. **Lookback Period Constraints**
   - Remove combinations where `short_period > long_period / 2`
   - Keep ratios: 1:3, 1:4, 1:5, 1:10

### A3. Execution Constraints
**Target reduction: ~1.3x**

1. **Price Source Simplification**
   - Remove: `mark_close` (현실적으로 next_open이나 close만 사용)
   - Keep: `next_open` (conservative), `close` (optimistic)
   - Reduction: 3 → 2 values

2. **Cost Profile Simplification**
   - Remove: `aggressive` (1bps slippage는 비현실적)
   - Keep: `base` (realistic), `conservative` (safe)
   - Reduction: 3 → 2 values

---

## Strategy B: Representative Sampling (대표점 선택)

### B1. Parameter Value Reduction
**Target reduction: ~3x**

#### Fibonacci/Logarithmic Scaling
Instead of linear steps, use geometric progression:

**Period/Lookback Parameters** (현재: 균일 간격 → 변경: 로그 스케일)
- Old: [10, 20, 30, 40, 50] → 5 values
- New: [10, 21, 55] → 3 values (Fibonacci)
- Reduction: 5 → 3 (1.67x)

**Multiplier/Ratio Parameters** (현재: 0.5 step → 변경: 대표 3-4점)
- Old: [1.0, 1.5, 2.0, 2.5, 3.0, 3.5] → 6 values
- New: [1.5, 2.5, 3.5] → 3 values
- Reduction: 6 → 3 (2x)

**Percentage Parameters** (SL/TP)
- Old SL: [0.01, 0.015, 0.02, 0.025, 0.03] → 5 values
- New SL: [0.01, 0.02, 0.03] → 3 values (1%, 2%, 3%)
- Old TP: [0.02, 0.03, 0.04, 0.05, 0.06] → 5 values
- New TP: [0.02, 0.04, 0.06] → 3 values (2%, 4%, 6%)
- Reduction: 5×5=25 → 3×3=9 combos (2.78x after constraints)

**Threshold Parameters** (RSI, ADX, etc.)
- Old: [15, 20, 25, 30] → 4 values
- New: [20, 30] → 2 values (extreme/moderate)
- Reduction: 4 → 2 (2x)

### B2. Leverage Reduction
- Old: [1, 2, 3, 5] → 4 values
- New: [1, 3] → 2 values (conservative/aggressive)
- Reduction: 4 → 2 (2x)

### B3. Timeframe Reduction
**Keep most informative timeframes**
- Old: [15m, 1h, 4h, 1d] → 4 values
- New: [1h, 1d] → 2 values (intraday/daily)
- Alternative: [1h, 4h, 1d] → 3 values if more granularity needed
- Reduction: 4 → 2 (2x)

---

## Strategy C: Two-Stage Search (Coarse → Fine)

### C1. Stage 1: Coarse Grid (1/20th of full grid)
**Execution**: Mandatory first pass

Parameters:
- Use only 1-2 values per axis (extremes or median)
- Keep all strategy types
- Use only `base` cost profile
- Use only `next_open` price source
- Use only `leverage=1`
- Use only 1 timeframe (`1h` or `1d`)

Expected: ~200,000 configs → Run all

### C2. Stage 2: Fine Grid (Top N neighborhoods)
**Execution**: After Stage 1 completes

Selection:
- Sort Stage 1 results by `sharpe_ratio` (or custom metric)
- Select top 50 configs (or top 1% if > 5000 configs)
- For each top config:
  - Expand parameters to ±1 step in each direction
  - Create local grid around winning config

Expected: 50 configs × ~27 neighbors = ~1,350 configs

---

## Combined Reduction Estimate

### Before Optimization:
- **Total**: 4,015,872 configs

### After Strategy A (Hard Constraints):
- SL/TP ratio filter: ÷2.5
- Strategy constraints: ÷2.0
- Execution simplification: ÷1.3
- **Subtotal**: 4,015,872 ÷ (2.5 × 2.0 × 1.3) ≈ **617,827**

### After Strategy B (Representative Sampling):
- Period reduction: ÷1.67
- Multiplier reduction: ÷2.0
- SL/TP value reduction: ÷2.78
- Threshold reduction: ÷2.0
- Leverage reduction: ÷2.0
- Timeframe reduction: ÷2.0
- **Subtotal**: 617,827 ÷ (1.67 × 2.0 × 2.78 × 2.0 × 2.0 × 2.0) ≈ **3,307**

Wait, that's too aggressive! Let's recalibrate:

### Realistic Reduction (A + B):
- Apply A only: 4,015,872 → **617,827** (6.5x)
- Apply A + B (selective):
  - Keep 4 timeframes (don't reduce)
  - Reduce periods: ÷1.5 (not ÷1.67)
  - Reduce SL/TP: ÷2.0 (with constraints)
  - Reduce leverage: ÷2.0
  - Reduce thresholds: ÷1.5
  - **Result**: 617,827 ÷ (1.5 × 2.0 × 2.0 × 1.5) ≈ **68,647** (58x total)

### With Strategy C (Two-Stage):
- Stage 1 (coarse): ~**50,000** configs (aggressive sampling)
- Stage 2 (fine): ~**1,500** configs (top 50 × 30 neighbors)
- **Total**: ~**51,500** configs (78x reduction)

---

## Implementation Approach

### Phase 1: Modify Grid Generator
**File**: `trader/massive_backtest.py`

Add constraint filtering in `GridConfigLoader.expand_grid()`:
```python
def expand_grid(self, grid: dict, mode: str = "full") -> list[BacktestConfig]:
    """
    mode: "full" | "coarse" | "fine"
    """
    # Generate combinations as before
    # Then apply constraint filters
    configs = []
    for config in raw_configs:
        if self._passes_constraints(config):
            configs.append(config)
    return configs

def _passes_constraints(self, config: BacktestConfig) -> bool:
    # A1: SL/TP ratio
    if config.take_profit_pct <= config.stop_loss_pct * 1.2:
        return False
    if config.take_profit_pct >= config.stop_loss_pct * 4.0:
        return False

    # A2: Min profit after fees
    cost = COST_PROFILES[config.cost_profile]
    total_cost = (cost.slippage_pct + cost.fee_pct) * 2
    if config.take_profit_pct < total_cost * 3:
        return False

    # A3: Leverage constraints
    if not config.allow_short and config.leverage > 3:
        return False

    # Strategy-specific constraints
    if "fast" in config.params and "slow" in config.params:
        if config.params["fast"] >= config.params["slow"]:
            return False
        if config.params["slow"] / config.params["fast"] < 2.0:
            return False

    return True
```

### Phase 2: Reduce Grid YAML Values
**Files**: `config/grids/*.yaml`

Apply Strategy B (representative sampling) directly in YAML:
- Reduce value counts per parameter
- Keep representative points

### Phase 3: Add Two-Stage Mode
**File**: `trader/massive_backtest.py`

Add `--mode coarse|fine` CLI option:
```python
def run_massive_backtest(
    families: list[str] | None = None,
    mode: str = "full",  # NEW
    base_results: pd.DataFrame | None = None,  # NEW for fine mode
    top_n: int = 50,  # NEW
    ...
):
    if mode == "coarse":
        # Use only representative values
        configs = loader.expand_grid(grid, mode="coarse")
    elif mode == "fine":
        # Expand around top configs from base_results
        configs = loader.expand_fine_grid(base_results, top_n)
    else:
        # Full grid with constraints
        configs = loader.expand_grid(grid, mode="full")
```

---

## Validation Plan

1. **Sanity Check**: Run 1 family (carry, smallest) with old vs new grid
   - Verify constraint logic doesn't break execution
   - Compare top 10 configs from both runs

2. **Performance Check**: Measure runtime reduction
   - Expected: ~10x fewer configs → ~10x faster (if I/O bound)
   - Actual: May be less due to overhead

3. **Quality Check**: Compare Sharpe ratios
   - Top 10 Sharpe from old grid
   - Top 10 Sharpe from new grid
   - Acceptable if: new top 10 avg within 10% of old top 10 avg

---

## Deliverables

1. Modified `trader/massive_backtest.py` with constraint filtering
2. Modified `config/grids/*.yaml` with reduced values
3. New script `scripts/compare_grids.py` for validation
4. Updated `analyze_grid_explosion.py` to show constraint impact
5. README section documenting new options
6. Before/After comparison table


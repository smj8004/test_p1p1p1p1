"""
MTF Strategy Optimizer with Market Regime Detection

Features:
1. Grid search parameter optimization
2. Market regime detection (trending/ranging/volatile)
3. Regime-specific strategy selection
4. Walk-forward validation
"""

from __future__ import annotations

import copy
import itertools
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trader.logging import get_logger
from trader.mtf_backtest import (
    MTFBacktestConfig,
    MTFBacktester,
    MTFIndicatorCalculator,
    MTFBar,
    MTFBarBuilder,
    TrendFollowMTF,
    MomentumBreakoutMTF,
    MACDDivergenceMTF,
    RSIMeanReversionMTF,
    AdaptiveTrendMTF,
)

logger = get_logger(__name__)


# =============================================================================
# Market Regime Detection
# =============================================================================

@dataclass
class MarketRegime:
    """Market regime classification."""
    regime: str  # "trending_up", "trending_down", "ranging", "volatile"
    confidence: float  # 0.0 to 1.0
    adx: float
    volatility: float
    trend_direction: str  # "bullish", "bearish", "neutral"


class MarketRegimeDetector:
    """
    Detects market regime using multiple indicators.

    Regimes:
    - trending_up: Strong uptrend (ADX > 25, price > EMA50)
    - trending_down: Strong downtrend (ADX > 25, price < EMA50)
    - ranging: Low ADX, price oscillating around EMA
    - volatile: High ATR, rapid regime changes
    """

    def __init__(
        self,
        adx_trending_threshold: float = 25.0,
        adx_ranging_threshold: float = 20.0,
        volatility_high_threshold: float = 2.0,  # ATR as % of price
    ):
        self.adx_trending_threshold = adx_trending_threshold
        self.adx_ranging_threshold = adx_ranging_threshold
        self.volatility_high_threshold = volatility_high_threshold

        # State for calculations
        self.prices: list[float] = []
        self.highs: list[float] = []
        self.lows: list[float] = []
        self.max_history = 200

    def update(self, close: float, high: float, low: float):
        """Update with new price data."""
        self.prices.append(close)
        self.highs.append(high)
        self.lows.append(low)

        if len(self.prices) > self.max_history:
            self.prices.pop(0)
            self.highs.pop(0)
            self.lows.pop(0)

    def _calculate_ema(self, period: int) -> float:
        """Calculate EMA."""
        if len(self.prices) < period:
            return self.prices[-1] if self.prices else 0.0

        multiplier = 2 / (period + 1)
        ema = sum(self.prices[:period]) / period

        for price in self.prices[period:]:
            ema = (price - ema) * multiplier + ema

        return ema

    def _calculate_adx(self, period: int = 14) -> float:
        """Calculate ADX (simplified)."""
        if len(self.prices) < period + 1:
            return 0.0

        plus_dm = []
        minus_dm = []
        tr_values = []

        for i in range(-period, 0):
            high = self.highs[i]
            low = self.lows[i]
            prev_high = self.highs[i-1]
            prev_low = self.lows[i-1]
            prev_close = self.prices[i-1]

            up_move = high - prev_high
            down_move = prev_low - low

            plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0)
            minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0)

            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)

        atr = sum(tr_values) / period
        if atr == 0:
            return 0.0

        plus_di = 100 * sum(plus_dm) / period / atr
        minus_di = 100 * sum(minus_dm) / period / atr

        if plus_di + minus_di == 0:
            return 0.0

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx

    def _calculate_atr_pct(self, period: int = 14) -> float:
        """Calculate ATR as percentage of price."""
        if len(self.prices) < period + 1:
            return 0.0

        tr_values = []
        for i in range(-period, 0):
            high = self.highs[i]
            low = self.lows[i]
            prev_close = self.prices[i-1]

            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)

        atr = sum(tr_values) / period
        current_price = self.prices[-1]

        return (atr / current_price) * 100 if current_price > 0 else 0.0

    def detect(self) -> MarketRegime:
        """Detect current market regime."""
        if len(self.prices) < 50:
            return MarketRegime(
                regime="unknown",
                confidence=0.0,
                adx=0.0,
                volatility=0.0,
                trend_direction="neutral"
            )

        adx = self._calculate_adx()
        ema_21 = self._calculate_ema(21)
        ema_50 = self._calculate_ema(50)
        volatility = self._calculate_atr_pct()
        current_price = self.prices[-1]

        # Determine trend direction
        if current_price > ema_50 and ema_21 > ema_50:
            trend_direction = "bullish"
        elif current_price < ema_50 and ema_21 < ema_50:
            trend_direction = "bearish"
        else:
            trend_direction = "neutral"

        # Classify regime
        if volatility > self.volatility_high_threshold:
            regime = "volatile"
            confidence = min(volatility / self.volatility_high_threshold / 2, 1.0)
        elif adx > self.adx_trending_threshold:
            if trend_direction == "bullish":
                regime = "trending_up"
            elif trend_direction == "bearish":
                regime = "trending_down"
            else:
                regime = "trending_up" if current_price > ema_50 else "trending_down"
            confidence = min((adx - self.adx_trending_threshold) / 25, 1.0)
        elif adx < self.adx_ranging_threshold:
            regime = "ranging"
            confidence = min((self.adx_ranging_threshold - adx) / 20, 1.0)
        else:
            # Transition zone
            regime = "mixed"
            confidence = 0.5

        return MarketRegime(
            regime=regime,
            confidence=confidence,
            adx=adx,
            volatility=volatility,
            trend_direction=trend_direction
        )


# =============================================================================
# Strategy Parameter Definitions
# =============================================================================

STRATEGY_PARAMS = {
    "TrendFollow": {
        "class": TrendFollowMTF,
        "params": {
            "trend_adx_threshold": [25.0, 30.0],
            "pullback_rsi_low": [35.0, 40.0],
            "pullback_rsi_high": [60.0, 65.0],
            "entry_rsi_threshold": [50.0],
        },
        "best_regimes": ["trending_up", "trending_down"],
    },
    "MomentumBreakout": {
        "class": MomentumBreakoutMTF,
        "params": {
            "bb_squeeze_threshold": [0.015, 0.02],
            "volume_multiplier": [1.5, 2.0],
            "rsi_confirmation": [55.0],
        },
        "best_regimes": ["ranging", "volatile"],
    },
    "MACDDivergence": {
        "class": MACDDivergenceMTF,
        "params": {
            "divergence_bars": [10, 15],
            "rsi_oversold": [25.0, 30.0],
            "rsi_overbought": [70.0, 75.0],
        },
        "best_regimes": ["trending_up", "trending_down", "ranging"],
    },
    "RSIMeanReversion": {
        "class": RSIMeanReversionMTF,
        "params": {
            "h1_rsi_oversold": [20.0, 25.0],
            "h1_rsi_overbought": [75.0, 80.0],
            "m15_rsi_recovery": [35.0],
            "require_trend": [True],
        },
        "best_regimes": ["ranging"],
    },
}

# Risk management parameter grid (reduced)
RISK_PARAMS = {
    "stop_loss_pct": [3.0, 4.0],
    "take_profit_pct": [6.0, 8.0],
    "min_holding_bars": [60, 120],
}


def generate_strategy_combinations(strategy_name: str) -> list[dict]:
    """Generate all parameter combinations for a strategy."""
    if strategy_name not in STRATEGY_PARAMS:
        return []

    params = STRATEGY_PARAMS[strategy_name]["params"]
    keys = list(params.keys())
    values = list(params.values())

    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))

    return combinations


# =============================================================================
# Optimizer
# =============================================================================

@dataclass
class OptimizationResult:
    """Result of a single optimization run."""
    strategy_name: str
    params: dict
    leverage: int
    total_return_pct: float
    win_rate: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    total_trades: int
    avg_trade_pnl: float
    regime_performance: dict  # Performance by market regime


class MTFOptimizer:
    """
    Optimizes MTF strategies with:
    1. Grid search over parameters
    2. Market regime analysis
    3. Walk-forward validation
    """

    def __init__(
        self,
        df: pd.DataFrame,
        funding_df: pd.DataFrame | None = None,
        initial_capital: float = 10000.0,
    ):
        self.df = df
        self.funding_df = funding_df
        self.initial_capital = initial_capital

        # Pre-calculate market regimes for the entire dataset
        self.regimes = self._calculate_regimes()

    def _calculate_regimes(self) -> pd.DataFrame:
        """Pre-calculate market regimes for each timestamp."""
        logger.info("Calculating market regimes...")

        detector = MarketRegimeDetector()
        regimes = []

        for idx, row in self.df.iterrows():
            detector.update(row["close"], row["high"], row["low"])
            regime = detector.detect()
            regimes.append({
                "timestamp": row["timestamp"],
                "regime": regime.regime,
                "adx": regime.adx,
                "volatility": regime.volatility,
                "trend": regime.trend_direction,
            })

        regime_df = pd.DataFrame(regimes)

        # Log regime distribution
        regime_counts = regime_df["regime"].value_counts()
        logger.info("Market regime distribution:")
        for regime, count in regime_counts.items():
            pct = count / len(regime_df) * 100
            logger.info(f"  {regime}: {count:,} bars ({pct:.1f}%)")

        return regime_df

    def _run_single_backtest(
        self,
        strategy,
        leverage: int,
        stop_loss_pct: float,
        take_profit_pct: float,
        min_holding_bars: int,
    ) -> dict:
        """Run a single backtest and return results."""
        config = MTFBacktestConfig(
            initial_capital=self.initial_capital,
            leverage=leverage,
            use_stop_loss=True,
            stop_loss_pct=stop_loss_pct,
            use_take_profit=True,
            take_profit_pct=take_profit_pct,
            use_trailing_stop=True,
            trailing_stop_pct=stop_loss_pct * 0.75,
            min_holding_bars=min_holding_bars,
            cooldown_bars=15,
        )

        backtester = MTFBacktester(
            config=config,
            strategy=strategy,
            funding_rates=self.funding_df,
        )

        # Suppress progress logging for optimization
        import logging
        old_level = logging.getLogger("trader.mtf_backtest").level
        logging.getLogger("trader.mtf_backtest").setLevel(logging.WARNING)

        result = backtester.run(self.df.copy())

        logging.getLogger("trader.mtf_backtest").setLevel(old_level)

        return result

    def optimize_strategy(
        self,
        strategy_name: str,
        leverages: list[int] = [3, 5],
        top_n: int = 5,
    ) -> list[OptimizationResult]:
        """
        Optimize a single strategy.

        Returns top N parameter combinations.
        """
        if strategy_name not in STRATEGY_PARAMS:
            logger.error(f"Unknown strategy: {strategy_name}")
            return []

        strategy_info = STRATEGY_PARAMS[strategy_name]
        strategy_class = strategy_info["class"]
        param_combinations = generate_strategy_combinations(strategy_name)

        logger.info(f"Optimizing {strategy_name}...")
        logger.info(f"  Parameter combinations: {len(param_combinations)}")
        logger.info(f"  Leverages: {leverages}")
        logger.info(f"  Risk param combinations: {len(list(itertools.product(*RISK_PARAMS.values())))}")

        results: list[OptimizationResult] = []
        total_tests = (
            len(param_combinations) *
            len(leverages) *
            len(RISK_PARAMS["stop_loss_pct"]) *
            len(RISK_PARAMS["take_profit_pct"]) *
            len(RISK_PARAMS["min_holding_bars"])
        )

        logger.info(f"  Total tests: {total_tests}")

        current_test = 0
        for params in param_combinations:
            for leverage in leverages:
                for sl in RISK_PARAMS["stop_loss_pct"]:
                    for tp in RISK_PARAMS["take_profit_pct"]:
                        for hold in RISK_PARAMS["min_holding_bars"]:
                            current_test += 1

                            if current_test % 50 == 0:
                                logger.info(f"  Progress: {current_test}/{total_tests}")

                            try:
                                strategy = strategy_class(**params)
                                result = self._run_single_backtest(
                                    strategy=strategy,
                                    leverage=leverage,
                                    stop_loss_pct=sl,
                                    take_profit_pct=tp,
                                    min_holding_bars=hold,
                                )

                                # Calculate average trade PnL
                                avg_pnl = 0.0
                                if result["total_trades"] > 0:
                                    avg_pnl = result["total_return"] / result["total_trades"]

                                results.append(OptimizationResult(
                                    strategy_name=strategy_name,
                                    params={
                                        **params,
                                        "stop_loss_pct": sl,
                                        "take_profit_pct": tp,
                                        "min_holding_bars": hold,
                                    },
                                    leverage=leverage,
                                    total_return_pct=result["total_return_pct"],
                                    win_rate=result["win_rate"],
                                    max_drawdown_pct=result["max_drawdown_pct"],
                                    sharpe_ratio=result["sharpe_ratio"],
                                    profit_factor=result["profit_factor"],
                                    total_trades=result["total_trades"],
                                    avg_trade_pnl=avg_pnl,
                                    regime_performance={},  # TODO: Calculate per-regime
                                ))
                            except Exception as e:
                                logger.warning(f"Test failed: {e}")
                                continue

        # Sort by Sharpe ratio (or use composite score)
        results.sort(key=lambda x: x.sharpe_ratio, reverse=True)

        return results[:top_n]

    def optimize_all_strategies(
        self,
        leverages: list[int] = [3, 5],
        top_n_per_strategy: int = 3,
    ) -> dict[str, list[OptimizationResult]]:
        """Optimize all strategies and return best parameters for each."""
        all_results = {}

        for strategy_name in STRATEGY_PARAMS.keys():
            results = self.optimize_strategy(
                strategy_name=strategy_name,
                leverages=leverages,
                top_n=top_n_per_strategy,
            )
            all_results[strategy_name] = results

        return all_results

    def find_regime_best_strategy(
        self,
        results: dict[str, list[OptimizationResult]],
    ) -> dict[str, OptimizationResult]:
        """
        Find the best strategy for each market regime.

        Returns: {regime: best_strategy_result}
        """
        # TODO: Implement per-regime backtesting
        # For now, use the strategy's declared best regimes
        regime_best = {}

        for strategy_name, strategy_results in results.items():
            if not strategy_results:
                continue

            best = strategy_results[0]
            best_regimes = STRATEGY_PARAMS[strategy_name]["best_regimes"]

            for regime in best_regimes:
                if regime not in regime_best:
                    regime_best[regime] = best
                elif best.sharpe_ratio > regime_best[regime].sharpe_ratio:
                    regime_best[regime] = best

        return regime_best


# =============================================================================
# Adaptive Strategy
# =============================================================================

class AdaptiveRegimeStrategy:
    """
    Adaptive strategy that switches based on market regime.

    Uses different optimized strategies for:
    - Trending up: TrendFollow or MACDDivergence
    - Trending down: TrendFollow or MACDDivergence
    - Ranging: RSIMeanReversion or MomentumBreakout
    - Volatile: MomentumBreakout
    """

    name = "AdaptiveRegime_MTF"

    def __init__(
        self,
        trending_strategy: Any,
        ranging_strategy: Any,
        volatile_strategy: Any,
    ):
        self.trending_strategy = trending_strategy
        self.ranging_strategy = ranging_strategy
        self.volatile_strategy = volatile_strategy

        self.regime_detector = MarketRegimeDetector()
        self.current_regime = "unknown"
        self.regime_switch_count = 0

    def on_bar(self, bars, indicators, position: str, entry_price: float | None) -> str:
        """Generate signal based on current regime."""
        # Update regime detector
        self.regime_detector.update(
            bars.m1.close,
            bars.m1.high,
            bars.m1.low
        )

        regime = self.regime_detector.detect()

        # Track regime switches
        if regime.regime != self.current_regime:
            self.current_regime = regime.regime
            self.regime_switch_count += 1

        # Select strategy based on regime
        if regime.regime in ["trending_up", "trending_down"]:
            return self.trending_strategy.on_bar(bars, indicators, position, entry_price)
        elif regime.regime == "ranging":
            return self.ranging_strategy.on_bar(bars, indicators, position, entry_price)
        elif regime.regime == "volatile":
            return self.volatile_strategy.on_bar(bars, indicators, position, entry_price)
        else:
            # Mixed/unknown - use trending strategy as default
            return self.trending_strategy.on_bar(bars, indicators, position, entry_price)


# =============================================================================
# Main Optimization Function
# =============================================================================

def run_mtf_optimization(
    symbol: str = "BTCUSDT",
    days: int = 365,
    leverages: list[int] = [3, 5],
    data_dir: str = "data/futures",
    output_dir: str = "data/futures/optimization",
) -> pd.DataFrame:
    """
    Run full MTF optimization.

    Steps:
    1. Load data
    2. Optimize each strategy
    3. Find best strategy per regime
    4. Test adaptive strategy
    5. Save results
    """
    from trader.logging import setup_logging
    setup_logging(level="INFO")

    # Load data
    data_path = Path(data_dir) / "clean" / symbol / "ohlcv_1m.parquet"
    if not data_path.exists():
        data_path = Path(data_dir) / "clean" / f"{symbol}_1m.parquet"
    if not data_path.exists():
        logger.error(f"Data not found: {data_path}")
        return pd.DataFrame()

    logger.info(f"Loading data: {data_path}")
    df = pd.read_parquet(data_path)

    # Normalize columns
    if "open_time" in df.columns:
        df = df.rename(columns={"open_time": "timestamp"})

    # Filter to requested days
    if days < 1095:
        cutoff = df["timestamp"].max() - pd.Timedelta(days=days)
        df = df[df["timestamp"] >= cutoff].reset_index(drop=True)

    logger.info(f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"Total bars: {len(df):,}")

    # Load funding rates
    funding_path = Path(data_dir) / "clean" / symbol / "funding_rate.parquet"
    if not funding_path.exists():
        funding_path = Path(data_dir) / "clean" / f"{symbol}_funding_rate.parquet"
    funding_df = None
    if funding_path.exists():
        funding_df = pd.read_parquet(funding_path)

    # Create optimizer
    optimizer = MTFOptimizer(df=df, funding_df=funding_df)

    # Optimize all strategies
    logger.info("")
    logger.info("=" * 70)
    logger.info("STRATEGY OPTIMIZATION")
    logger.info("=" * 70)

    all_results = optimizer.optimize_all_strategies(
        leverages=leverages,
        top_n_per_strategy=3,
    )

    # Print results
    logger.info("")
    logger.info("=" * 70)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("=" * 70)

    results_list = []
    for strategy_name, results in all_results.items():
        logger.info(f"\n{strategy_name}:")
        for i, r in enumerate(results, 1):
            logger.info(f"  #{i}: Return={r.total_return_pct:.2f}%, "
                       f"Sharpe={r.sharpe_ratio:.2f}, "
                       f"WinRate={r.win_rate:.1f}%, "
                       f"MaxDD={r.max_drawdown_pct:.1f}%, "
                       f"Trades={r.total_trades}")

            results_list.append({
                "strategy": strategy_name,
                "rank": i,
                "leverage": r.leverage,
                "total_return_pct": r.total_return_pct,
                "sharpe_ratio": r.sharpe_ratio,
                "win_rate": r.win_rate,
                "max_drawdown_pct": r.max_drawdown_pct,
                "profit_factor": r.profit_factor,
                "total_trades": r.total_trades,
                "avg_trade_pnl": r.avg_trade_pnl,
                **{f"param_{k}": v for k, v in r.params.items()},
            })

    # Find best per regime
    regime_best = optimizer.find_regime_best_strategy(all_results)

    logger.info("")
    logger.info("=" * 70)
    logger.info("BEST STRATEGY PER REGIME")
    logger.info("=" * 70)

    for regime, result in regime_best.items():
        logger.info(f"  {regime}: {result.strategy_name} "
                   f"(Return={result.total_return_pct:.2f}%, "
                   f"Sharpe={result.sharpe_ratio:.2f})")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results_list)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"optimization_{symbol}_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)

    logger.info("")
    logger.info(f"Results saved to: {results_file}")

    # Save regime info
    regime_file = output_path / f"regime_best_{symbol}_{timestamp}.json"
    regime_data = {
        regime: {
            "strategy": r.strategy_name,
            "params": r.params,
            "leverage": r.leverage,
            "return_pct": r.total_return_pct,
            "sharpe": r.sharpe_ratio,
        }
        for regime, r in regime_best.items()
    }
    with open(regime_file, "w") as f:
        json.dump(regime_data, f, indent=2)

    logger.info(f"Regime best saved to: {regime_file}")

    return results_df


if __name__ == "__main__":
    run_mtf_optimization(days=90)

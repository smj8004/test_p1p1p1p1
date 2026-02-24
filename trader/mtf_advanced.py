"""
Advanced MTF Trading System

Features:
1. Smart Filters (Volume, Time, Volatility, Momentum)
2. Machine Learning Optimization (Optuna-style Bayesian)
3. Walk-Forward Validation (Anti-overfitting)
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from trader.logger_utils import get_logger
from trader.mtf_backtest import (
    MTFBacktestConfig,
    MTFBacktester,
    MTFBar,
    MTFBars,
    MTFIndicators,
    TrendFollowMTF,
    MomentumBreakoutMTF,
    MACDDivergenceMTF,
    RSIMeanReversionMTF,
)

logger = get_logger(__name__)


# =============================================================================
# 1. SMART FILTERS
# =============================================================================

@dataclass
class FilterConfig:
    """Configuration for trade filters."""
    # Volume filter
    use_volume_filter: bool = True
    volume_ma_period: int = 20
    min_volume_ratio: float = 0.8  # Min ratio vs MA
    max_volume_ratio: float = 3.0  # Max ratio vs MA (avoid spikes)

    # Time filter (UTC hours)
    use_time_filter: bool = True
    avoid_hours: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])  # Asian low vol
    prefer_hours: list[int] = field(default_factory=lambda: [13, 14, 15, 16, 17, 18, 19, 20])  # US session

    # Volatility filter
    use_volatility_filter: bool = True
    atr_period: int = 14
    min_atr_pct: float = 0.3  # Min volatility for entry
    max_atr_pct: float = 3.0  # Max volatility (too risky)

    # Momentum filter
    use_momentum_filter: bool = True
    momentum_period: int = 10
    min_momentum_strength: float = 0.5  # % move required

    # Spread/liquidity filter (simulated)
    use_spread_filter: bool = True
    max_spread_pct: float = 0.1  # Max acceptable spread

    # Consecutive loss filter
    use_loss_filter: bool = True
    max_consecutive_losses: int = 3


class SmartFilter:
    """
    Applies multiple filters to trading signals.

    Filters can:
    - Block entries in unfavorable conditions
    - Force exits in dangerous conditions
    - Adjust position sizing based on conditions
    """

    def __init__(self, config: FilterConfig):
        self.config = config

        # State tracking
        self.volume_history: list[float] = []
        self.atr_history: list[float] = []
        self.price_history: list[float] = []
        self.high_history: list[float] = []
        self.low_history: list[float] = []
        self.consecutive_losses: int = 0

        # Filter statistics
        self.filter_stats = {
            "volume_blocked": 0,
            "time_blocked": 0,
            "volatility_blocked": 0,
            "momentum_blocked": 0,
            "loss_blocked": 0,
            "total_signals": 0,
            "passed_signals": 0,
        }

    def update(self, bar: MTFBar):
        """Update filter state with new bar."""
        self.volume_history.append(bar.volume)
        self.price_history.append(bar.close)
        self.high_history.append(bar.high)
        self.low_history.append(bar.low)

        # Trim history
        max_len = max(
            self.config.volume_ma_period,
            self.config.atr_period,
            self.config.momentum_period,
        ) + 10

        if len(self.volume_history) > max_len:
            self.volume_history = self.volume_history[-max_len:]
            self.price_history = self.price_history[-max_len:]
            self.high_history = self.high_history[-max_len:]
            self.low_history = self.low_history[-max_len:]

    def record_trade_result(self, is_win: bool):
        """Track consecutive losses."""
        if is_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

    def _check_volume(self) -> tuple[bool, str]:
        """Check volume conditions."""
        if not self.config.use_volume_filter:
            return True, ""

        if len(self.volume_history) < self.config.volume_ma_period:
            return True, ""

        recent = self.volume_history[-self.config.volume_ma_period:]
        avg_volume = sum(recent) / len(recent)
        current_volume = self.volume_history[-1]

        if avg_volume == 0:
            return True, ""

        ratio = current_volume / avg_volume

        if ratio < self.config.min_volume_ratio:
            return False, f"Low volume ({ratio:.2f}x avg)"
        if ratio > self.config.max_volume_ratio:
            return False, f"Volume spike ({ratio:.2f}x avg)"

        return True, ""

    def _check_time(self, timestamp: datetime) -> tuple[bool, str]:
        """Check time conditions."""
        if not self.config.use_time_filter:
            return True, ""

        hour = timestamp.hour

        if hour in self.config.avoid_hours:
            return False, f"Avoid hour ({hour}:00 UTC)"

        return True, ""

    def _check_volatility(self) -> tuple[bool, str]:
        """Check volatility conditions."""
        if not self.config.use_volatility_filter:
            return True, ""

        if len(self.price_history) < self.config.atr_period + 1:
            return True, ""

        # Calculate ATR
        tr_values = []
        for i in range(-self.config.atr_period, 0):
            high = self.high_history[i]
            low = self.low_history[i]
            prev_close = self.price_history[i - 1]

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            tr_values.append(tr)

        atr = sum(tr_values) / len(tr_values)
        current_price = self.price_history[-1]
        atr_pct = (atr / current_price) * 100 if current_price > 0 else 0

        if atr_pct < self.config.min_atr_pct:
            return False, f"Low volatility ({atr_pct:.2f}%)"
        if atr_pct > self.config.max_atr_pct:
            return False, f"High volatility ({atr_pct:.2f}%)"

        return True, ""

    def _check_momentum(self, direction: str) -> tuple[bool, str]:
        """Check momentum alignment."""
        if not self.config.use_momentum_filter:
            return True, ""

        if len(self.price_history) < self.config.momentum_period:
            return True, ""

        start_price = self.price_history[-self.config.momentum_period]
        current_price = self.price_history[-1]
        momentum_pct = ((current_price - start_price) / start_price) * 100

        if direction == "long":
            if momentum_pct < -self.config.min_momentum_strength:
                return False, f"Negative momentum ({momentum_pct:.2f}%)"
        elif direction == "short":
            if momentum_pct > self.config.min_momentum_strength:
                return False, f"Positive momentum ({momentum_pct:.2f}%)"

        return True, ""

    def _check_consecutive_losses(self) -> tuple[bool, str]:
        """Check consecutive loss limit."""
        if not self.config.use_loss_filter:
            return True, ""

        if self.consecutive_losses >= self.config.max_consecutive_losses:
            return False, f"Consecutive losses ({self.consecutive_losses})"

        return True, ""

    def should_enter(
        self,
        signal: str,
        timestamp: datetime,
    ) -> tuple[bool, str, float]:
        """
        Check if entry should be allowed.

        Returns:
            (allowed, reason, position_size_multiplier)
        """
        if signal not in ["long", "short"]:
            return True, "", 1.0

        self.filter_stats["total_signals"] += 1

        # Check all filters
        checks = [
            ("volume", self._check_volume()),
            ("time", self._check_time(timestamp)),
            ("volatility", self._check_volatility()),
            ("momentum", self._check_momentum(signal)),
            ("loss", self._check_consecutive_losses()),
        ]

        for name, (passed, reason) in checks:
            if not passed:
                self.filter_stats[f"{name}_blocked"] += 1
                return False, reason, 0.0

        self.filter_stats["passed_signals"] += 1

        # Calculate position size multiplier based on conditions
        multiplier = 1.0

        # Reduce size if in less preferred hours
        hour = timestamp.hour
        if hour not in self.config.prefer_hours:
            multiplier *= 0.7

        # Reduce size after losses
        if self.consecutive_losses > 0:
            multiplier *= (1.0 - 0.2 * self.consecutive_losses)
            multiplier = max(multiplier, 0.3)

        return True, "", multiplier

    def should_exit_early(self, timestamp: datetime) -> tuple[bool, str]:
        """Check if position should be exited early due to conditions."""
        # Exit if entering avoid hours with position
        if self.config.use_time_filter:
            hour = timestamp.hour
            if hour in self.config.avoid_hours:
                return True, "Entering low-volume hours"

        return False, ""

    def get_stats(self) -> dict:
        """Get filter statistics."""
        total = self.filter_stats["total_signals"]
        if total == 0:
            return self.filter_stats

        stats = dict(self.filter_stats)
        stats["pass_rate"] = (self.filter_stats["passed_signals"] / total) * 100
        return stats


# =============================================================================
# 2. MACHINE LEARNING OPTIMIZATION (Bayesian-style)
# =============================================================================

@dataclass
class TrialResult:
    """Result of a single optimization trial."""
    trial_id: int
    params: dict
    score: float  # Objective value (higher is better)
    metrics: dict  # Additional metrics


class BayesianOptimizer:
    """
    Bayesian-style optimization for strategy parameters.

    Uses a simple acquisition function based on:
    - Exploitation: Focus on best known parameters
    - Exploration: Try new parameter combinations
    """

    def __init__(
        self,
        param_space: dict[str, tuple],
        objective_fn: Callable[[dict], float],
        n_initial: int = 10,
        n_iterations: int = 50,
        exploration_weight: float = 0.3,
    ):
        """
        Args:
            param_space: {param_name: (min, max, type)} where type is 'int' or 'float'
            objective_fn: Function that takes params dict and returns score
            n_initial: Number of random initial trials
            n_iterations: Total number of iterations
            exploration_weight: Balance between exploration and exploitation
        """
        self.param_space = param_space
        self.objective_fn = objective_fn
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.exploration_weight = exploration_weight

        self.trials: list[TrialResult] = []
        self.best_params: dict = {}
        self.best_score: float = float("-inf")

    def _sample_random(self) -> dict:
        """Sample random parameters from the space."""
        params = {}
        for name, (low, high, ptype) in self.param_space.items():
            if ptype == "int":
                params[name] = random.randint(int(low), int(high))
            elif ptype == "float":
                params[name] = random.uniform(low, high)
            elif ptype == "choice":
                params[name] = random.choice(low)  # low contains choices
            else:
                params[name] = random.uniform(low, high)
        return params

    def _sample_near_best(self) -> dict:
        """Sample parameters near the best known."""
        if not self.trials:
            return self._sample_random()

        # Get top 3 trials
        sorted_trials = sorted(self.trials, key=lambda t: t.score, reverse=True)
        top_trials = sorted_trials[:3]

        # Pick one randomly
        base = random.choice(top_trials).params.copy()

        # Perturb some parameters
        params = {}
        for name, (low, high, ptype) in self.param_space.items():
            if random.random() < 0.5:  # 50% chance to perturb
                if ptype == "int":
                    delta = int((high - low) * 0.1) + 1
                    new_val = base.get(name, (low + high) // 2) + random.randint(-delta, delta)
                    params[name] = max(int(low), min(int(high), new_val))
                elif ptype == "float":
                    delta = (high - low) * 0.15
                    new_val = base.get(name, (low + high) / 2) + random.uniform(-delta, delta)
                    params[name] = max(low, min(high, new_val))
                elif ptype == "choice":
                    params[name] = random.choice(low)
                else:
                    params[name] = base.get(name, (low + high) / 2)
            else:
                params[name] = base.get(name, (low + high) / 2)

        return params

    def _suggest(self) -> dict:
        """Suggest next parameters to try."""
        # Initial random exploration
        if len(self.trials) < self.n_initial:
            return self._sample_random()

        # Balance exploration and exploitation
        if random.random() < self.exploration_weight:
            return self._sample_random()
        else:
            return self._sample_near_best()

    def optimize(self) -> tuple[dict, float]:
        """Run optimization and return best parameters."""
        logger.info(f"Starting Bayesian optimization ({self.n_iterations} iterations)")

        for i in range(self.n_iterations):
            params = self._suggest()

            try:
                score = self.objective_fn(params)
            except Exception as e:
                logger.warning(f"Trial {i} failed: {e}")
                score = float("-inf")

            self.trials.append(TrialResult(
                trial_id=i,
                params=params,
                score=score,
                metrics={}
            ))

            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                logger.info(f"  Trial {i}: New best score = {score:.4f}")

            if (i + 1) % 10 == 0:
                logger.info(f"  Progress: {i + 1}/{self.n_iterations}, Best: {self.best_score:.4f}")

        return self.best_params, self.best_score


def create_ml_objective(
    df: pd.DataFrame,
    strategy_class: type,
    funding_df: pd.DataFrame | None = None,
    leverage: int = 3,
) -> Callable[[dict], float]:
    """Create objective function for ML optimization."""

    def objective(params: dict) -> float:
        # Separate strategy params from risk params
        strategy_params = {
            k: v for k, v in params.items()
            if k not in ["stop_loss_pct", "take_profit_pct", "min_holding_bars"]
        }

        try:
            strategy = strategy_class(**strategy_params)
        except Exception:
            return float("-inf")

        config = MTFBacktestConfig(
            leverage=leverage,
            use_stop_loss=True,
            stop_loss_pct=params.get("stop_loss_pct", 3.0),
            use_take_profit=True,
            take_profit_pct=params.get("take_profit_pct", 6.0),
            min_holding_bars=int(params.get("min_holding_bars", 60)),
        )

        backtester = MTFBacktester(
            config=config,
            strategy=strategy,
            funding_rates=funding_df,
        )

        # Suppress logging
        import logging
        old_level = logging.getLogger("trader.mtf_backtest").level
        logging.getLogger("trader.mtf_backtest").setLevel(logging.ERROR)

        result = backtester.run(df.copy())

        logging.getLogger("trader.mtf_backtest").setLevel(old_level)

        # Composite score: Sharpe ratio + win rate bonus - drawdown penalty
        sharpe = result.get("sharpe_ratio", 0)
        win_rate = result.get("win_rate", 0) / 100
        max_dd = result.get("max_drawdown_pct", 100) / 100
        trades = result.get("total_trades", 0)

        # Penalize too few or too many trades
        trade_penalty = 0
        if trades < 10:
            trade_penalty = -2
        elif trades > 200:
            trade_penalty = -1

        score = sharpe + win_rate - max_dd + trade_penalty

        return score

    return objective


# =============================================================================
# 3. WALK-FORWARD VALIDATION
# =============================================================================

@dataclass
class WalkForwardWindow:
    """A single walk-forward window."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_result: dict = field(default_factory=dict)
    test_result: dict = field(default_factory=dict)
    best_params: dict = field(default_factory=dict)


class WalkForwardValidator:
    """
    Walk-forward validation to prevent overfitting.

    Process:
    1. Split data into rolling train/test windows
    2. Optimize on train window
    3. Validate on test window (out-of-sample)
    4. Roll forward and repeat
    5. Aggregate results across all windows
    """

    def __init__(
        self,
        df: pd.DataFrame,
        train_days: int = 60,
        test_days: int = 30,
        step_days: int = 30,
        funding_df: pd.DataFrame | None = None,
    ):
        self.df = df
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
        self.funding_df = funding_df

        self.windows: list[WalkForwardWindow] = []
        self._create_windows()

    def _create_windows(self):
        """Create walk-forward windows."""
        df = self.df

        start_date = df["timestamp"].min()
        end_date = df["timestamp"].max()
        total_days = (end_date - start_date).days

        current_start = start_date
        window_id = 0

        while True:
            train_end = current_start + timedelta(days=self.train_days)
            test_start = train_end
            test_end = test_start + timedelta(days=self.test_days)

            if test_end > end_date:
                break

            self.windows.append(WalkForwardWindow(
                train_start=current_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            ))

            current_start += timedelta(days=self.step_days)
            window_id += 1

        logger.info(f"Created {len(self.windows)} walk-forward windows")

    def _get_window_data(self, window: WalkForwardWindow, is_train: bool) -> pd.DataFrame:
        """Get data for a specific window."""
        if is_train:
            start, end = window.train_start, window.train_end
        else:
            start, end = window.test_start, window.test_end

        mask = (self.df["timestamp"] >= start) & (self.df["timestamp"] < end)
        return self.df[mask].reset_index(drop=True)

    def run_validation(
        self,
        strategy_class: type,
        param_space: dict[str, tuple],
        n_optimization_trials: int = 30,
        leverage: int = 3,
    ) -> dict:
        """
        Run walk-forward validation.

        Returns aggregated results across all windows.
        """
        logger.info("=" * 70)
        logger.info("WALK-FORWARD VALIDATION")
        logger.info("=" * 70)
        logger.info(f"  Windows: {len(self.windows)}")
        logger.info(f"  Train: {self.train_days} days, Test: {self.test_days} days")

        all_test_returns = []
        all_test_sharpe = []
        all_test_win_rates = []

        for i, window in enumerate(self.windows):
            logger.info(f"\nWindow {i + 1}/{len(self.windows)}")
            logger.info(f"  Train: {window.train_start.date()} to {window.train_end.date()}")
            logger.info(f"  Test: {window.test_start.date()} to {window.test_end.date()}")

            # Get train data
            train_df = self._get_window_data(window, is_train=True)
            if len(train_df) < 1000:
                logger.warning("  Insufficient train data, skipping")
                continue

            # Optimize on train
            objective = create_ml_objective(
                df=train_df,
                strategy_class=strategy_class,
                funding_df=self.funding_df,
                leverage=leverage,
            )

            optimizer = BayesianOptimizer(
                param_space=param_space,
                objective_fn=objective,
                n_initial=5,
                n_iterations=n_optimization_trials,
            )

            best_params, train_score = optimizer.optimize()
            window.best_params = best_params

            # Test on out-of-sample data
            test_df = self._get_window_data(window, is_train=False)
            if len(test_df) < 500:
                logger.warning("  Insufficient test data, skipping")
                continue

            # Create strategy with best params
            strategy_params = {
                k: v for k, v in best_params.items()
                if k not in ["stop_loss_pct", "take_profit_pct", "min_holding_bars"]
            }
            strategy = strategy_class(**strategy_params)

            config = MTFBacktestConfig(
                leverage=leverage,
                use_stop_loss=True,
                stop_loss_pct=best_params.get("stop_loss_pct", 3.0),
                use_take_profit=True,
                take_profit_pct=best_params.get("take_profit_pct", 6.0),
                min_holding_bars=int(best_params.get("min_holding_bars", 60)),
            )

            # Suppress logging
            import logging
            old_level = logging.getLogger("trader.mtf_backtest").level
            logging.getLogger("trader.mtf_backtest").setLevel(logging.ERROR)

            backtester = MTFBacktester(
                config=config,
                strategy=strategy,
                funding_rates=self.funding_df,
            )
            test_result = backtester.run(test_df.copy())

            logging.getLogger("trader.mtf_backtest").setLevel(old_level)

            window.test_result = test_result

            logger.info(f"  Train score: {train_score:.4f}")
            logger.info(f"  Test return: {test_result['total_return_pct']:.2f}%")
            logger.info(f"  Test Sharpe: {test_result['sharpe_ratio']:.2f}")
            logger.info(f"  Test trades: {test_result['total_trades']}")

            all_test_returns.append(test_result["total_return_pct"])
            all_test_sharpe.append(test_result["sharpe_ratio"])
            all_test_win_rates.append(test_result["win_rate"])

        # Aggregate results
        if not all_test_returns:
            return {"error": "No valid windows"}

        results = {
            "n_windows": len(self.windows),
            "avg_test_return": np.mean(all_test_returns),
            "std_test_return": np.std(all_test_returns),
            "avg_test_sharpe": np.mean(all_test_sharpe),
            "avg_test_win_rate": np.mean(all_test_win_rates),
            "positive_windows": sum(1 for r in all_test_returns if r > 0),
            "negative_windows": sum(1 for r in all_test_returns if r <= 0),
            "best_window_return": max(all_test_returns),
            "worst_window_return": min(all_test_returns),
            "per_window_returns": all_test_returns,
        }

        logger.info("\n" + "=" * 70)
        logger.info("WALK-FORWARD RESULTS")
        logger.info("=" * 70)
        logger.info(f"  Average test return: {results['avg_test_return']:.2f}%")
        logger.info(f"  Std dev: {results['std_test_return']:.2f}%")
        logger.info(f"  Average Sharpe: {results['avg_test_sharpe']:.2f}")
        logger.info(f"  Positive/Negative windows: {results['positive_windows']}/{results['negative_windows']}")

        return results


# =============================================================================
# INTEGRATED STRATEGY WITH FILTERS
# =============================================================================

class FilteredMTFStrategy:
    """Wrapper that adds smart filters to any MTF strategy."""

    def __init__(
        self,
        base_strategy: Any,
        filter_config: FilterConfig | None = None,
    ):
        self.base_strategy = base_strategy
        self.filter = SmartFilter(filter_config or FilterConfig())
        self.name = f"Filtered_{base_strategy.name}"

        self._last_signal: str = "hold"
        self._pending_exit: bool = False

    def on_bar(
        self,
        bars: MTFBars,
        indicators: MTFIndicators,
        position: str,
        entry_price: float | None,
    ) -> str:
        # Update filter
        self.filter.update(bars.m1)

        # Check for early exit
        if position != "flat":
            should_exit, reason = self.filter.should_exit_early(bars.m1.timestamp)
            if should_exit:
                return "exit"

        # Get base strategy signal
        signal = self.base_strategy.on_bar(bars, indicators, position, entry_price)

        # Filter entry signals
        if signal in ["long", "short"]:
            allowed, reason, size_mult = self.filter.should_enter(
                signal, bars.m1.timestamp
            )
            if not allowed:
                return "hold"

        return signal


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def run_ml_optimization(
    symbol: str = "BTCUSDT",
    days: int = 90,
    strategy_name: str = "TrendFollow",
    n_trials: int = 50,
    leverage: int = 3,
    data_dir: str = "data/futures",
    output_dir: str = "data/futures/ml_optimization",
) -> dict:
    """Run ML-based optimization for a strategy."""
    from trader.logger_utils import setup_logging
    setup_logging(level="INFO")

    # Load data
    data_path = Path(data_dir) / "clean" / symbol / "ohlcv_1m.parquet"
    if not data_path.exists():
        data_path = Path(data_dir) / "clean" / f"{symbol}_1m.parquet"
    if not data_path.exists():
        logger.error(f"Data not found: {data_path}")
        return {}

    logger.info(f"Loading data: {data_path}")
    df = pd.read_parquet(data_path)

    if "open_time" in df.columns:
        df = df.rename(columns={"open_time": "timestamp"})

    if days < 1095:
        cutoff = df["timestamp"].max() - pd.Timedelta(days=days)
        df = df[df["timestamp"] >= cutoff].reset_index(drop=True)

    logger.info(f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Define parameter spaces for each strategy
    param_spaces = {
        "TrendFollow": {
            "trend_adx_threshold": (15.0, 35.0, "float"),
            "pullback_rsi_low": (25.0, 45.0, "float"),
            "pullback_rsi_high": (55.0, 75.0, "float"),
            "entry_rsi_threshold": (40.0, 60.0, "float"),
            "stop_loss_pct": (2.0, 5.0, "float"),
            "take_profit_pct": (4.0, 10.0, "float"),
            "min_holding_bars": (30, 180, "int"),
        },
        "MACDDivergence": {
            "divergence_bars": (5, 25, "int"),
            "rsi_oversold": (20.0, 40.0, "float"),
            "rsi_overbought": (60.0, 80.0, "float"),
            "stop_loss_pct": (2.0, 5.0, "float"),
            "take_profit_pct": (4.0, 10.0, "float"),
            "min_holding_bars": (30, 180, "int"),
        },
        "MomentumBreakout": {
            "bb_squeeze_threshold": (0.005, 0.03, "float"),
            "volume_multiplier": (1.2, 3.0, "float"),
            "rsi_confirmation": (45.0, 65.0, "float"),
            "stop_loss_pct": (2.0, 5.0, "float"),
            "take_profit_pct": (4.0, 10.0, "float"),
            "min_holding_bars": (30, 180, "int"),
        },
        "RSIMeanReversion": {
            "h1_rsi_oversold": (15.0, 35.0, "float"),
            "h1_rsi_overbought": (65.0, 85.0, "float"),
            "m15_rsi_recovery": (25.0, 45.0, "float"),
            "stop_loss_pct": (2.0, 5.0, "float"),
            "take_profit_pct": (4.0, 10.0, "float"),
            "min_holding_bars": (30, 180, "int"),
        },
    }

    strategy_classes = {
        "TrendFollow": TrendFollowMTF,
        "MACDDivergence": MACDDivergenceMTF,
        "MomentumBreakout": MomentumBreakoutMTF,
        "RSIMeanReversion": RSIMeanReversionMTF,
    }

    if strategy_name not in strategy_classes:
        logger.error(f"Unknown strategy: {strategy_name}")
        return {}

    param_space = param_spaces[strategy_name]
    strategy_class = strategy_classes[strategy_name]

    # Load funding rates
    funding_path = Path(data_dir) / "clean" / symbol / "funding_rate.parquet"
    funding_df = None
    if funding_path.exists():
        funding_df = pd.read_parquet(funding_path)

    # Create objective and run optimization
    objective = create_ml_objective(
        df=df,
        strategy_class=strategy_class,
        funding_df=funding_df,
        leverage=leverage,
    )

    optimizer = BayesianOptimizer(
        param_space=param_space,
        objective_fn=objective,
        n_initial=10,
        n_iterations=n_trials,
    )

    best_params, best_score = optimizer.optimize()

    # Run final backtest with best params
    strategy_params = {
        k: v for k, v in best_params.items()
        if k not in ["stop_loss_pct", "take_profit_pct", "min_holding_bars"]
    }
    strategy = strategy_class(**strategy_params)

    config = MTFBacktestConfig(
        leverage=leverage,
        use_stop_loss=True,
        stop_loss_pct=best_params.get("stop_loss_pct", 3.0),
        use_take_profit=True,
        take_profit_pct=best_params.get("take_profit_pct", 6.0),
        min_holding_bars=int(best_params.get("min_holding_bars", 60)),
    )

    backtester = MTFBacktester(
        config=config,
        strategy=strategy,
        funding_rates=funding_df,
    )
    final_result = backtester.run(df.copy())

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_path / f"ml_opt_{strategy_name}_{timestamp}.json"

    results = {
        "strategy": strategy_name,
        "best_params": best_params,
        "best_score": best_score,
        "final_result": final_result,
        "n_trials": n_trials,
        "leverage": leverage,
    }

    with open(result_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {result_file}")
    logger.info(f"Best params: {best_params}")
    logger.info(f"Final return: {final_result['total_return_pct']:.2f}%")

    return results


def run_walk_forward(
    symbol: str = "BTCUSDT",
    strategy_name: str = "TrendFollow",
    train_days: int = 60,
    test_days: int = 30,
    n_trials: int = 30,
    leverage: int = 3,
    data_dir: str = "data/futures",
    output_dir: str = "data/futures/walk_forward",
) -> dict:
    """Run walk-forward validation."""
    from trader.logger_utils import setup_logging
    setup_logging(level="INFO")

    # Load data
    data_path = Path(data_dir) / "clean" / symbol / "ohlcv_1m.parquet"
    if not data_path.exists():
        data_path = Path(data_dir) / "clean" / f"{symbol}_1m.parquet"
    if not data_path.exists():
        logger.error(f"Data not found: {data_path}")
        return {}

    df = pd.read_parquet(data_path)
    if "open_time" in df.columns:
        df = df.rename(columns={"open_time": "timestamp"})

    logger.info(f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Load funding
    funding_path = Path(data_dir) / "clean" / symbol / "funding_rate.parquet"
    funding_df = None
    if funding_path.exists():
        funding_df = pd.read_parquet(funding_path)

    # Strategy setup
    strategy_classes = {
        "TrendFollow": TrendFollowMTF,
        "MACDDivergence": MACDDivergenceMTF,
        "MomentumBreakout": MomentumBreakoutMTF,
        "RSIMeanReversion": RSIMeanReversionMTF,
    }

    param_spaces = {
        "TrendFollow": {
            "trend_adx_threshold": (15.0, 35.0, "float"),
            "pullback_rsi_low": (25.0, 45.0, "float"),
            "pullback_rsi_high": (55.0, 75.0, "float"),
            "entry_rsi_threshold": (40.0, 60.0, "float"),
            "stop_loss_pct": (2.0, 5.0, "float"),
            "take_profit_pct": (4.0, 10.0, "float"),
            "min_holding_bars": (30, 180, "int"),
        },
        "MACDDivergence": {
            "divergence_bars": (5, 25, "int"),
            "rsi_oversold": (20.0, 40.0, "float"),
            "rsi_overbought": (60.0, 80.0, "float"),
            "stop_loss_pct": (2.0, 5.0, "float"),
            "take_profit_pct": (4.0, 10.0, "float"),
            "min_holding_bars": (30, 180, "int"),
        },
        "MomentumBreakout": {
            "bb_squeeze_threshold": (0.005, 0.03, "float"),
            "volume_multiplier": (1.2, 3.0, "float"),
            "rsi_confirmation": (45.0, 65.0, "float"),
            "stop_loss_pct": (2.0, 5.0, "float"),
            "take_profit_pct": (4.0, 10.0, "float"),
            "min_holding_bars": (30, 180, "int"),
        },
        "RSIMeanReversion": {
            "h1_rsi_oversold": (15.0, 35.0, "float"),
            "h1_rsi_overbought": (65.0, 85.0, "float"),
            "m15_rsi_recovery": (25.0, 45.0, "float"),
            "stop_loss_pct": (2.0, 5.0, "float"),
            "take_profit_pct": (4.0, 10.0, "float"),
            "min_holding_bars": (30, 180, "int"),
        },
    }

    validator = WalkForwardValidator(
        df=df,
        train_days=train_days,
        test_days=test_days,
        step_days=test_days,
        funding_df=funding_df,
    )

    results = validator.run_validation(
        strategy_class=strategy_classes[strategy_name],
        param_space=param_spaces[strategy_name],
        n_optimization_trials=n_trials,
        leverage=leverage,
    )

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_path / f"wf_{strategy_name}_{timestamp}.json"

    with open(result_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {result_file}")

    return results


if __name__ == "__main__":
    # Example: run ML optimization
    run_ml_optimization(
        days=90,
        strategy_name="TrendFollow",
        n_trials=50,
    )

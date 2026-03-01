"""
Walk-Forward Validation Experiment

Tests strategy robustness through time-based cross-validation:
- Rolling train/test splits
- Parameter optimization on training data
- Validation on out-of-sample test data
- Parameter stability analysis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from trader.experiments.core import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentType,
    ScenarioResult,
)
from trader.strategy import STRATEGY_FACTORIES
from trader.strategy.base import Bar, Strategy

logger = logging.getLogger(__name__)


@dataclass
class WFOSplit:
    """Single walk-forward split"""
    split_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_df: pd.DataFrame
    test_df: pd.DataFrame


class WalkForwardExperiment:
    """
    Walk-Forward Validation Experiment

    Validates strategy through time-based cross-validation.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        df: pd.DataFrame,
        param_grid: dict[str, list[Any]] | None = None,
    ):
        self.config = config
        self.df = df
        self.param_grid = param_grid or self._default_param_grid()

    def _default_param_grid(self) -> dict[str, list[Any]]:
        """Default parameter grid for common strategies"""
        if self.config.strategy_name == "ema_cross":
            return {
                "short_window": [8, 12, 20],
                "long_window": [20, 26, 50],
            }
        elif self.config.strategy_name == "rsi":
            return {
                "period": [10, 14, 21],
                "oversold": [20, 30, 40],
                "overbought": [60, 70, 80],
            }
        else:
            return {}

    def create_splits(self) -> list[WFOSplit]:
        """Create walk-forward splits"""
        type_spec = self.config.type_specific
        train_days = type_spec.get("train_days", 180)
        test_days = type_spec.get("test_days", 60)
        n_splits = type_spec.get("n_splits", 5)

        # Infer timeframe hours
        tf = self.config.timeframe.lower()
        if "h" in tf:
            tf_hours = int(tf.replace("h", ""))
        elif "m" in tf:
            tf_hours = int(tf.replace("m", "")) / 60
        elif "d" in tf:
            tf_hours = int(tf.replace("d", "")) * 24
        else:
            tf_hours = 1

        train_bars = int(train_days * 24 / tf_hours)
        test_bars = int(test_days * 24 / tf_hours)
        step_bars = test_bars  # Non-overlapping windows

        splits = []
        total_bars = len(self.df)

        for i in range(n_splits):
            train_start_idx = i * step_bars
            train_end_idx = train_start_idx + train_bars
            test_end_idx = train_end_idx + test_bars

            if test_end_idx > total_bars:
                break

            train_df = self.df.iloc[train_start_idx:train_end_idx]
            test_df = self.df.iloc[train_end_idx:test_end_idx]

            if len(train_df) < 100 or len(test_df) < 20:
                continue

            splits.append(
                WFOSplit(
                    split_id=i,
                    train_start=str(train_df.index[0]),
                    train_end=str(train_df.index[-1]),
                    test_start=str(test_df.index[0]),
                    test_end=str(test_df.index[-1]),
                    train_df=train_df,
                    test_df=test_df,
                )
            )

        return splits

    def run(self) -> ExperimentResult:
        """Execute walk-forward validation"""
        logger.info(f"Starting Walk-Forward Validation: {self.config.experiment_id}")

        splits = self.create_splits()
        logger.info(f"Created {len(splits)} walk-forward splits")

        if not splits:
            logger.warning("No valid splits created, insufficient data")
            return self._empty_result()

        results = []
        optimal_params_per_split = []

        for split in splits:
            logger.info(
                f"Processing split {split.split_id}: "
                f"train={split.train_start} to {split.train_end}, "
                f"test={split.test_start} to {split.test_end}"
            )

            # Optimize on training data
            best_params = self._optimize_on_train(split)
            optimal_params_per_split.append(best_params)

            # Validate on test data
            test_result = self._validate_on_test(split, best_params)
            results.append(test_result)

        # Calculate robustness score
        robustness_score = self._calculate_robustness(results, optimal_params_per_split)

        # Summary statistics
        oos_sharpe = [r.sharpe_ratio for r in results]
        oos_pnl = [r.net_pnl for r in results]
        oos_positive = sum(1 for p in oos_pnl if p > 0)

        summary = {
            "oos_positive_ratio": oos_positive / len(results) if results else 0.0,
            "oos_mean_sharpe": np.mean(oos_sharpe) if oos_sharpe else 0.0,
            "oos_median_sharpe": np.median(oos_sharpe) if oos_sharpe else 0.0,
            "oos_mean_pnl": np.mean(oos_pnl) if oos_pnl else 0.0,
            "param_stability_score": self._param_stability(optimal_params_per_split),
            "split_count": len(results),
        }

        verdict = ExperimentResult.calculate_verdict(robustness_score)

        return ExperimentResult(
            config=self.config,
            summary=summary,
            scenarios=results,
            robustness_score=robustness_score,
            verdict=verdict,
        )

    def _optimize_on_train(self, split: WFOSplit) -> dict[str, Any]:
        """
        Optimize parameters on training data

        Simple grid search for best Sharpe ratio
        """
        type_spec = self.config.type_specific
        top_k = type_spec.get("top_k", 5)

        if not self.param_grid:
            return self.config.strategy_params

        # Generate all parameter combinations
        param_combinations = self._generate_param_combinations()

        # Limit search space
        if len(param_combinations) > 100:
            np.random.seed(self.config.seed)
            indices = np.random.choice(len(param_combinations), 100, replace=False)
            param_combinations = [param_combinations[i] for i in indices]

        best_sharpe = -999
        best_params = self.config.strategy_params

        for params in param_combinations:
            try:
                result = self._run_backtest(split.train_df, params)
                if result.sharpe_ratio > best_sharpe and result.trade_count >= 5:
                    best_sharpe = result.sharpe_ratio
                    best_params = params
            except Exception as e:
                logger.debug(f"Failed to run params {params}: {e}")
                continue

        return best_params

    def _validate_on_test(self, split: WFOSplit, params: dict[str, Any]) -> ScenarioResult:
        """Validate parameters on test data"""
        result = self._run_backtest(split.test_df, params)

        scenario_result = ScenarioResult(
            scenario_id=f"split_{split.split_id}",
            scenario_params={
                "split_id": split.split_id,
                "test_start": split.test_start,
                "test_end": split.test_end,
                "optimal_params": params,
            },
            net_pnl=result.net_pnl,
            cagr=result.cagr,
            max_drawdown=result.max_drawdown,
            sharpe_ratio=result.sharpe_ratio,
            profit_factor=result.profit_factor,
            win_rate=result.win_rate,
            trade_count=result.trade_count,
        )

        return scenario_result

    def _run_backtest(
        self, df: pd.DataFrame, params: dict[str, Any]
    ) -> ScenarioResult:
        """Run a simple backtest with given parameters"""
        # Create strategy
        strategy = self._create_strategy(params)

        # Run simulation
        equity = self.config.initial_equity
        peak_equity = equity
        max_drawdown = 0.0

        position_side = "flat"
        position_qty = 0.0
        position_entry_price = 0.0

        trades = []

        for i in range(len(df)):
            row = df.iloc[i]
            bar = Bar(
                timestamp=row.name if hasattr(row.name, "isoformat") else str(row.name),
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
            )

            # Get signal
            position_obj = type("Position", (), {"side": position_side, "entry_price": position_entry_price})()
            signal = strategy.on_bar(bar, position_obj)

            # Execute signal
            exec_price = row["close"]
            fee_pct = 0.0004  # 4 bps

            if signal == "long" and position_side != "long":
                # Close short if any
                if position_side == "short":
                    pnl = position_qty * (position_entry_price - exec_price)
                    fee = position_qty * exec_price * fee_pct
                    equity += pnl - fee
                    trades.append(pnl - fee)

                # Open long
                position_qty = (equity * 0.95) / exec_price
                fee = position_qty * exec_price * fee_pct
                equity -= fee
                position_side = "long"
                position_entry_price = exec_price

            elif signal == "short" and position_side != "short":
                # Close long if any
                if position_side == "long":
                    pnl = position_qty * (exec_price - position_entry_price)
                    fee = position_qty * exec_price * fee_pct
                    equity += pnl - fee
                    trades.append(pnl - fee)

                # Open short
                position_qty = (equity * 0.95) / exec_price
                fee = position_qty * exec_price * fee_pct
                equity -= fee
                position_side = "short"
                position_entry_price = exec_price

            elif signal == "exit" and position_side != "flat":
                if position_side == "long":
                    pnl = position_qty * (exec_price - position_entry_price)
                else:
                    pnl = position_qty * (position_entry_price - exec_price)

                fee = position_qty * exec_price * fee_pct
                equity += pnl - fee
                trades.append(pnl - fee)
                position_side = "flat"
                position_qty = 0.0

            # Track drawdown
            peak_equity = max(peak_equity, equity)
            drawdown = (equity - peak_equity) / peak_equity * 100
            max_drawdown = min(max_drawdown, drawdown)

        # Close final position
        if position_side != "flat":
            final_price = df.iloc[-1]["close"]
            if position_side == "long":
                pnl = position_qty * (final_price - position_entry_price)
            else:
                pnl = position_qty * (position_entry_price - final_price)
            equity += pnl
            trades.append(pnl)

        # Calculate metrics
        net_pnl = equity - self.config.initial_equity
        total_return = net_pnl / self.config.initial_equity

        # CAGR
        days = (pd.to_datetime(df.index[-1]) - pd.to_datetime(df.index[0])).days
        years = max(days / 365.25, 0.01)
        cagr = (1 + total_return) ** (1 / years) - 1

        # Sharpe
        if trades:
            trade_returns = np.array(trades) / self.config.initial_equity
            sharpe_ratio = (
                np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(252)
                if np.std(trade_returns) > 0
                else 0.0
            )
        else:
            sharpe_ratio = 0.0

        # Profit factor
        wins = [t for t in trades if t > 0]
        losses = [abs(t) for t in trades if t < 0]
        profit_factor = sum(wins) / sum(losses) if losses else (100.0 if wins else 0.0)

        # Win rate
        win_rate = len(wins) / len(trades) * 100 if trades else 0.0

        return ScenarioResult(
            scenario_id="temp",
            scenario_params={},
            net_pnl=net_pnl,
            cagr=cagr * 100,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            win_rate=win_rate,
            trade_count=len(trades),
        )

    def _create_strategy(self, params: dict[str, Any]) -> Strategy:
        """Create strategy instance"""
        # Handle original strategies
        if self.config.strategy_name == "ema_cross":
            from trader.strategy.ema_cross import EMACrossStrategy
            return EMACrossStrategy(**params)
        elif self.config.strategy_name == "rsi":
            from trader.strategy.rsi import RSIStrategy
            return RSIStrategy(**params)
        elif self.config.strategy_name == "macd":
            from trader.strategy.macd import MACDStrategy
            return MACDStrategy(**params)
        elif self.config.strategy_name == "bollinger":
            from trader.strategy.bollinger import BollingerBandStrategy
            return BollingerBandStrategy(**params)

        # Handle strategy families
        for family_name, factory in STRATEGY_FACTORIES.items():
            if self.config.strategy_name.startswith(family_name):
                return factory(self.config.strategy_name, params)

        raise ValueError(f"Unknown strategy: {self.config.strategy_name}")

    def _generate_param_combinations(self) -> list[dict[str, Any]]:
        """Generate all parameter combinations from grid"""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        combinations = []
        for vals in self._cartesian_product(values):
            combinations.append(dict(zip(keys, vals)))

        return combinations

    def _cartesian_product(self, arrays: list[list[Any]]) -> list[tuple]:
        """Cartesian product of lists"""
        if not arrays:
            return []

        result = [[]]
        for pool in arrays:
            result = [x + [y] for x in result for y in pool]

        return [tuple(r) for r in result]

    def _param_stability(self, params_per_split: list[dict[str, Any]]) -> float:
        """
        Calculate parameter stability score

        Lower variance in optimal parameters = higher stability
        """
        if not params_per_split or not self.param_grid:
            return 1.0

        # Calculate coefficient of variation for each parameter
        cvs = []
        for param_name in self.param_grid.keys():
            values = [p.get(param_name, 0) for p in params_per_split]
            if not values:
                continue

            mean_val = np.mean(values)
            std_val = np.std(values)

            if mean_val > 0:
                cv = std_val / mean_val
                cvs.append(cv)

        # Lower CV = higher stability
        if cvs:
            avg_cv = np.mean(cvs)
            stability = 1.0 / (1.0 + avg_cv)
        else:
            stability = 1.0

        return stability

    def _calculate_robustness(
        self, results: list[ScenarioResult], params_per_split: list[dict[str, Any]]
    ) -> float:
        """
        Calculate robustness score

        Combination of:
        - OOS positive ratio (50% weight)
        - Parameter stability (30% weight)
        - Mean OOS Sharpe (20% weight)
        """
        if not results:
            return 0.0

        # OOS positive ratio
        positive_count = sum(1 for r in results if r.net_pnl > 0)
        oos_positive_ratio = positive_count / len(results)

        # Parameter stability
        param_stability = self._param_stability(params_per_split)

        # Mean OOS Sharpe (normalized to 0-1)
        sharpe_ratios = [r.sharpe_ratio for r in results]
        mean_sharpe = np.mean(sharpe_ratios)
        sharpe_score = min(max(mean_sharpe, 0) / 2.0, 1.0)  # Normalize: 2.0 Sharpe = 1.0 score

        # Weighted combination
        robustness = (
            0.5 * oos_positive_ratio + 0.3 * param_stability + 0.2 * sharpe_score
        )

        return robustness

    def _empty_result(self) -> ExperimentResult:
        """Return empty result when no valid splits"""
        return ExperimentResult(
            config=self.config,
            summary={
                "oos_positive_ratio": 0.0,
                "oos_mean_sharpe": 0.0,
                "param_stability_score": 0.0,
                "split_count": 0,
            },
            scenarios=[],
            robustness_score=0.0,
            verdict="NO EDGE",
        )

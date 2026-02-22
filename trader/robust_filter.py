"""
Robust Filtering and Ensemble Optimization

Features:
- Overfitting detection filters (min trades, DD, PF, Sharpe)
- Walk-Forward Optimization (WFO) validation
- Monte Carlo simulation for statistical significance
- Ensemble weight search with constraints
- Out-of-Sample (OOS) validation
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class RobustFilter:
    """Overfitting detection filter configuration"""
    min_trades: int = 30
    max_drawdown_pct: float = -40.0
    min_profit_factor: float = 1.0
    min_sharpe: float = 0.5
    min_win_rate: float = 40.0
    max_trades_per_day: float = 10.0
    min_avg_trade_pnl: float = 0.0
    wfo_positive_ratio: float = 0.6
    mc_percentile_threshold: float = 60.0


@dataclass
class WFOResult:
    """Walk-Forward Optimization result"""
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_return_pct: float
    test_return_pct: float
    train_sharpe: float
    test_sharpe: float
    is_positive: bool


@dataclass
class MCResult:
    """Monte Carlo simulation result"""
    n_simulations: int
    actual_return: float
    sim_mean: float
    sim_std: float
    percentile: float
    p_value: float
    is_significant: bool


@dataclass
class EnsembleWeight:
    """Ensemble strategy weight"""
    config_id: str
    weight: float
    family: str
    strategy_type: str


@dataclass
class EnsembleResult:
    """Ensemble optimization result"""
    weights: list[EnsembleWeight]
    in_sample_return: float
    in_sample_sharpe: float
    out_of_sample_return: float
    out_of_sample_sharpe: float
    diversification_ratio: float


class RobustFilterEngine:
    """Apply robust filters to backtest results"""

    def __init__(self, config: RobustFilter | None = None):
        self.config = config or RobustFilter()

    def apply_basic_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply basic statistical filters"""
        filtered = df.copy()

        # Min trades
        if "total_trades" in filtered.columns:
            filtered = filtered[filtered["total_trades"] >= self.config.min_trades]

        # Max drawdown
        if "max_drawdown_pct" in filtered.columns:
            filtered = filtered[filtered["max_drawdown_pct"] >= self.config.max_drawdown_pct]

        # Min profit factor
        if "profit_factor" in filtered.columns:
            filtered = filtered[filtered["profit_factor"] >= self.config.min_profit_factor]

        # Min sharpe
        if "sharpe_ratio" in filtered.columns:
            filtered = filtered[filtered["sharpe_ratio"] >= self.config.min_sharpe]

        # Min win rate
        if "win_rate" in filtered.columns:
            filtered = filtered[filtered["win_rate"] >= self.config.min_win_rate]

        # Max trades per day
        if "trades_per_day" in filtered.columns:
            filtered = filtered[filtered["trades_per_day"] <= self.config.max_trades_per_day]

        return filtered

    def filter_by_wfo(
        self,
        df: pd.DataFrame,
        wfo_results: dict[str, list[WFOResult]],
    ) -> pd.DataFrame:
        """Filter by Walk-Forward Optimization results"""
        filtered_ids = []

        for config_id, results in wfo_results.items():
            if not results:
                continue

            positive_count = sum(1 for r in results if r.is_positive)
            ratio = positive_count / len(results)

            if ratio >= self.config.wfo_positive_ratio:
                filtered_ids.append(config_id)

        if "config_id" in df.columns:
            return df[df["config_id"].isin(filtered_ids)]
        return df

    def filter_by_mc(
        self,
        df: pd.DataFrame,
        mc_results: dict[str, MCResult],
    ) -> pd.DataFrame:
        """Filter by Monte Carlo simulation results"""
        filtered_ids = []

        for config_id, result in mc_results.items():
            if result.percentile >= self.config.mc_percentile_threshold:
                filtered_ids.append(config_id)

        if "config_id" in df.columns:
            return df[df["config_id"].isin(filtered_ids)]
        return df


class WalkForwardOptimizer:
    """Walk-Forward Optimization for robustness testing"""

    def __init__(
        self,
        train_ratio: float = 0.7,
        n_splits: int = 5,
        min_train_bars: int = 500,
    ):
        self.train_ratio = train_ratio
        self.n_splits = n_splits
        self.min_train_bars = min_train_bars

    def create_splits(
        self,
        df: pd.DataFrame,
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Create train/test splits for walk-forward"""
        n = len(df)
        splits = []

        split_size = n // self.n_splits
        train_size = int(split_size * self.train_ratio)
        test_size = split_size - train_size

        if train_size < self.min_train_bars:
            # Not enough data for requested splits
            train_size = int(n * self.train_ratio)
            test_size = n - train_size
            splits.append((df.iloc[:train_size], df.iloc[train_size:]))
            return splits

        for i in range(self.n_splits):
            start_idx = i * split_size
            train_end = start_idx + train_size
            test_end = min(start_idx + split_size, n)

            if test_end <= train_end:
                continue

            train_df = df.iloc[start_idx:train_end]
            test_df = df.iloc[train_end:test_end]

            if len(train_df) >= self.min_train_bars and len(test_df) > 0:
                splits.append((train_df, test_df))

        return splits

    def run_wfo(
        self,
        df: pd.DataFrame,
        backtest_func: Callable[[pd.DataFrame], dict],
    ) -> list[WFOResult]:
        """Run walk-forward optimization"""
        splits = self.create_splits(df)
        results = []

        for train_df, test_df in splits:
            train_result = backtest_func(train_df)
            test_result = backtest_func(test_df)

            train_return = train_result.get("return_pct", 0)
            test_return = test_result.get("return_pct", 0)
            train_sharpe = train_result.get("sharpe_ratio", 0)
            test_sharpe = test_result.get("sharpe_ratio", 0)

            wfo_result = WFOResult(
                train_start=str(train_df.index[0]),
                train_end=str(train_df.index[-1]),
                test_start=str(test_df.index[0]),
                test_end=str(test_df.index[-1]),
                train_return_pct=train_return,
                test_return_pct=test_return,
                train_sharpe=train_sharpe,
                test_sharpe=test_sharpe,
                is_positive=test_return > 0,
            )
            results.append(wfo_result)

        return results


class MonteCarloSimulator:
    """Monte Carlo simulation for statistical significance"""

    def __init__(
        self,
        n_simulations: int = 1000,
        significance_level: float = 0.05,
    ):
        self.n_simulations = n_simulations
        self.significance_level = significance_level

    def simulate_random_returns(
        self,
        returns: np.ndarray,
        n_trades: int,
    ) -> np.ndarray:
        """Generate random returns by shuffling"""
        simulated = np.zeros(self.n_simulations)

        for i in range(self.n_simulations):
            shuffled = np.random.choice(returns, size=n_trades, replace=True)
            simulated[i] = np.sum(shuffled)

        return simulated

    def run_simulation(
        self,
        actual_return: float,
        trade_returns: list[float],
    ) -> MCResult:
        """Run Monte Carlo simulation"""
        if len(trade_returns) < 5:
            return MCResult(
                n_simulations=0,
                actual_return=actual_return,
                sim_mean=0,
                sim_std=0,
                percentile=50,
                p_value=1.0,
                is_significant=False,
            )

        returns_arr = np.array(trade_returns)
        n_trades = len(returns_arr)

        # Simulate
        simulated = self.simulate_random_returns(returns_arr, n_trades)

        # Calculate statistics
        sim_mean = np.mean(simulated)
        sim_std = np.std(simulated)

        # Percentile of actual return (what % of simulations are below actual)
        percentile = np.sum(simulated <= actual_return) / len(simulated) * 100

        # P-value (two-tailed)
        if sim_std > 0:
            z_score = (actual_return - sim_mean) / sim_std
            p_value = 2 * min(percentile / 100, 1 - percentile / 100)
        else:
            p_value = 1.0

        is_significant = p_value < self.significance_level

        return MCResult(
            n_simulations=self.n_simulations,
            actual_return=actual_return,
            sim_mean=sim_mean,
            sim_std=sim_std,
            percentile=percentile,
            p_value=p_value,
            is_significant=is_significant,
        )


class EnsembleOptimizer:
    """Optimize ensemble weights with constraints"""

    def __init__(
        self,
        max_strategies: int = 10,
        min_weight: float = 0.05,
        max_weight: float = 0.5,
        max_family_weight: float = 0.6,
        oos_ratio: float = 0.3,
    ):
        self.max_strategies = max_strategies
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.max_family_weight = max_family_weight
        self.oos_ratio = oos_ratio

    def select_candidates(
        self,
        df: pd.DataFrame,
        n_candidates: int = 20,
    ) -> pd.DataFrame:
        """Select top candidates for ensemble"""
        # Sort by Sharpe ratio
        sorted_df = df.sort_values("sharpe_ratio", ascending=False)

        # Diversify by family
        selected = []
        family_counts: dict[str, int] = {}
        max_per_family = n_candidates // 3

        for _, row in sorted_df.iterrows():
            family = row.get("family", "unknown")
            if family_counts.get(family, 0) < max_per_family:
                selected.append(row)
                family_counts[family] = family_counts.get(family, 0) + 1

            if len(selected) >= n_candidates:
                break

        return pd.DataFrame(selected)

    def optimize_weights(
        self,
        returns_matrix: np.ndarray,
        n_strategies: int,
    ) -> np.ndarray:
        """Optimize portfolio weights using mean-variance optimization"""
        n = n_strategies

        # Calculate expected returns and covariance
        mean_returns = np.mean(returns_matrix, axis=0)
        cov_matrix = np.cov(returns_matrix.T)

        # Handle edge cases
        if cov_matrix.ndim == 0:
            return np.ones(n) / n

        # Objective: maximize Sharpe ratio (or minimize negative Sharpe)
        def neg_sharpe(weights):
            port_return = np.dot(weights, mean_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            if port_vol == 0:
                return 0
            return -port_return / port_vol

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},  # Sum to 1
        ]

        # Bounds
        bounds = [(self.min_weight, self.max_weight) for _ in range(n)]

        # Initial guess (equal weights)
        x0 = np.ones(n) / n

        # Optimize
        result = minimize(
            neg_sharpe,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            weights = result.x
            # Normalize
            weights = np.maximum(weights, 0)
            weights = weights / np.sum(weights)
            return weights
        else:
            return np.ones(n) / n

    def build_ensemble(
        self,
        candidates: pd.DataFrame,
        returns_data: dict[str, np.ndarray],
    ) -> EnsembleResult:
        """Build optimized ensemble"""
        # Split data for IS/OOS
        config_ids = candidates["config_id"].tolist()
        n_strategies = min(len(config_ids), self.max_strategies)

        if n_strategies == 0:
            return EnsembleResult(
                weights=[],
                in_sample_return=0,
                in_sample_sharpe=0,
                out_of_sample_return=0,
                out_of_sample_sharpe=0,
                diversification_ratio=0,
            )

        # Build returns matrix
        all_returns = []
        valid_ids = []

        for config_id in config_ids[:n_strategies]:
            if config_id in returns_data:
                all_returns.append(returns_data[config_id])
                valid_ids.append(config_id)

        if not all_returns:
            return EnsembleResult(
                weights=[],
                in_sample_return=0,
                in_sample_sharpe=0,
                out_of_sample_return=0,
                out_of_sample_sharpe=0,
                diversification_ratio=0,
            )

        # Align lengths
        min_len = min(len(r) for r in all_returns)
        returns_matrix = np.array([r[:min_len] for r in all_returns]).T

        # Split IS/OOS
        split_idx = int(len(returns_matrix) * (1 - self.oos_ratio))
        is_returns = returns_matrix[:split_idx]
        oos_returns = returns_matrix[split_idx:]

        # Optimize on IS
        weights = self.optimize_weights(is_returns, len(valid_ids))

        # Calculate IS performance
        is_portfolio_returns = np.dot(is_returns, weights)
        is_return = np.sum(is_portfolio_returns) * 100
        is_sharpe = np.mean(is_portfolio_returns) / (np.std(is_portfolio_returns) + 1e-10) * np.sqrt(252 * 24)

        # Calculate OOS performance
        if len(oos_returns) > 0:
            oos_portfolio_returns = np.dot(oos_returns, weights)
            oos_return = np.sum(oos_portfolio_returns) * 100
            oos_sharpe = np.mean(oos_portfolio_returns) / (np.std(oos_portfolio_returns) + 1e-10) * np.sqrt(252 * 24)
        else:
            oos_return = 0
            oos_sharpe = 0

        # Diversification ratio
        individual_vols = np.std(returns_matrix, axis=0)
        portfolio_vol = np.std(np.dot(returns_matrix, weights))
        weighted_avg_vol = np.dot(weights, individual_vols)
        div_ratio = weighted_avg_vol / (portfolio_vol + 1e-10)

        # Build weight list
        ensemble_weights = []
        for i, config_id in enumerate(valid_ids):
            row = candidates[candidates["config_id"] == config_id].iloc[0]
            ensemble_weights.append(EnsembleWeight(
                config_id=config_id,
                weight=float(weights[i]),
                family=row.get("family", "unknown"),
                strategy_type=row.get("strategy_type", "unknown"),
            ))

        return EnsembleResult(
            weights=ensemble_weights,
            in_sample_return=is_return,
            in_sample_sharpe=is_sharpe,
            out_of_sample_return=oos_return,
            out_of_sample_sharpe=oos_sharpe,
            diversification_ratio=div_ratio,
        )


def apply_robust_filters(
    df: pd.DataFrame,
    config: RobustFilter | None = None,
) -> pd.DataFrame:
    """Apply robust filters to backtest results"""
    engine = RobustFilterEngine(config)
    return engine.apply_basic_filters(df)


def run_ensemble_optimization(
    df: pd.DataFrame,
    returns_data: dict[str, np.ndarray],
    n_candidates: int = 20,
) -> EnsembleResult:
    """Run ensemble optimization"""
    optimizer = EnsembleOptimizer()
    candidates = optimizer.select_candidates(df, n_candidates)
    return optimizer.build_ensemble(candidates, returns_data)


def generate_robustness_report(
    results_df: pd.DataFrame,
    wfo_results: dict[str, list[WFOResult]] | None = None,
    mc_results: dict[str, MCResult] | None = None,
    ensemble_result: EnsembleResult | None = None,
    output_dir: Path = Path("out/reports"),
) -> Path:
    """Generate comprehensive robustness report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = output_dir / f"robustness_report_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Basic filters summary
    filter_engine = RobustFilterEngine()
    filtered = filter_engine.apply_basic_filters(results_df)

    summary = {
        "total_strategies": len(results_df),
        "after_basic_filters": len(filtered),
        "filter_pass_rate": len(filtered) / len(results_df) * 100 if len(results_df) > 0 else 0,
    }

    # WFO summary
    if wfo_results:
        wfo_positive = sum(
            1 for results in wfo_results.values()
            if len(results) > 0 and sum(r.is_positive for r in results) / len(results) >= 0.6
        )
        summary["wfo_pass_count"] = wfo_positive
        summary["wfo_pass_rate"] = wfo_positive / len(wfo_results) * 100 if wfo_results else 0

    # MC summary
    if mc_results:
        mc_significant = sum(1 for r in mc_results.values() if r.is_significant)
        summary["mc_significant_count"] = mc_significant
        summary["mc_significant_rate"] = mc_significant / len(mc_results) * 100 if mc_results else 0

    # Ensemble summary
    if ensemble_result:
        summary["ensemble_strategies"] = len(ensemble_result.weights)
        summary["ensemble_is_return"] = ensemble_result.in_sample_return
        summary["ensemble_is_sharpe"] = ensemble_result.in_sample_sharpe
        summary["ensemble_oos_return"] = ensemble_result.out_of_sample_return
        summary["ensemble_oos_sharpe"] = ensemble_result.out_of_sample_sharpe
        summary["ensemble_diversification"] = ensemble_result.diversification_ratio

        # Save ensemble weights
        weights_data = [
            {
                "config_id": w.config_id,
                "weight": w.weight,
                "family": w.family,
                "strategy_type": w.strategy_type,
            }
            for w in ensemble_result.weights
        ]
        with open(report_dir / "ensemble_weights.json", "w") as f:
            json.dump(weights_data, f, indent=2)

    # Save summary
    with open(report_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save filtered results
    filtered.to_csv(report_dir / "filtered_results.csv", index=False)

    logger.info(f"Robustness report saved to: {report_dir}")
    return report_dir

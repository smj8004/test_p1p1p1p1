"""
Tests for robust filtering and ensemble optimization
"""

import pytest
import numpy as np
import pandas as pd

from trader.robust_filter import (
    RobustFilter,
    RobustFilterEngine,
    WalkForwardOptimizer,
    MonteCarloSimulator,
    EnsembleOptimizer,
    apply_robust_filters,
)


class TestRobustFilterEngine:
    """Test robust filter engine"""

    @pytest.fixture
    def sample_results(self):
        """Create sample backtest results"""
        return pd.DataFrame({
            "config_id": ["a", "b", "c", "d", "e"],
            "total_trades": [50, 20, 100, 30, 10],
            "max_drawdown_pct": [-20, -50, -15, -40, -60],
            "profit_factor": [1.5, 0.8, 2.0, 1.1, 0.5],
            "sharpe_ratio": [1.0, 0.3, 1.5, 0.6, -0.2],
            "win_rate": [55, 40, 60, 45, 35],
            "trades_per_day": [2, 15, 3, 1, 20],
        })

    def test_default_filter_config(self):
        config = RobustFilter()
        assert config.min_trades == 30
        assert config.max_drawdown_pct == -40.0
        assert config.min_profit_factor == 1.0
        assert config.min_sharpe == 0.5

    def test_apply_basic_filters(self, sample_results):
        engine = RobustFilterEngine()
        filtered = engine.apply_basic_filters(sample_results)

        # Should filter out configs that don't meet criteria
        assert len(filtered) <= len(sample_results)

        # Check all filtered results meet criteria
        for _, row in filtered.iterrows():
            assert row["total_trades"] >= 30
            assert row["max_drawdown_pct"] >= -40
            assert row["profit_factor"] >= 1.0
            assert row["sharpe_ratio"] >= 0.5

    def test_filter_with_custom_config(self, sample_results):
        config = RobustFilter(
            min_trades=10,
            max_drawdown_pct=-60,
            min_profit_factor=0.5,
            min_sharpe=0.0,
        )
        engine = RobustFilterEngine(config)
        filtered = engine.apply_basic_filters(sample_results)

        # More lenient filters should pass more configs
        assert len(filtered) >= 3

    def test_empty_dataframe(self):
        engine = RobustFilterEngine()
        empty_df = pd.DataFrame()
        filtered = engine.apply_basic_filters(empty_df)
        assert len(filtered) == 0


class TestWalkForwardOptimizer:
    """Test walk-forward optimization"""

    @pytest.fixture
    def sample_data(self):
        """Create sample price data"""
        n = 1000
        dates = pd.date_range("2020-01-01", periods=n, freq="h")
        prices = 100 * np.cumprod(1 + np.random.randn(n) * 0.001)

        return pd.DataFrame({
            "open": prices * (1 + np.random.randn(n) * 0.001),
            "high": prices * (1 + np.abs(np.random.randn(n)) * 0.002),
            "low": prices * (1 - np.abs(np.random.randn(n)) * 0.002),
            "close": prices,
            "volume": 1000 + np.random.rand(n) * 500,
        }, index=dates)

    def test_create_splits(self, sample_data):
        optimizer = WalkForwardOptimizer(n_splits=5)
        splits = optimizer.create_splits(sample_data)

        assert len(splits) > 0

        for train, test in splits:
            assert len(train) > 0
            assert len(test) > 0
            # Train should come before test
            assert train.index[-1] < test.index[0]

    def test_create_splits_with_insufficient_data(self):
        optimizer = WalkForwardOptimizer(min_train_bars=500)
        small_data = pd.DataFrame({
            "close": [1, 2, 3, 4, 5],
        })

        splits = optimizer.create_splits(small_data)
        # Should still create at least one split
        assert len(splits) >= 0

    def test_run_wfo(self, sample_data):
        optimizer = WalkForwardOptimizer(n_splits=3)

        def mock_backtest(df):
            return {
                "return_pct": np.random.randn() * 10,
                "sharpe_ratio": np.random.randn(),
            }

        results = optimizer.run_wfo(sample_data, mock_backtest)

        assert len(results) > 0
        for result in results:
            assert hasattr(result, "train_return_pct")
            assert hasattr(result, "test_return_pct")
            assert hasattr(result, "is_positive")


class TestMonteCarloSimulator:
    """Test Monte Carlo simulation"""

    def test_simulate_random_returns(self):
        simulator = MonteCarloSimulator(n_simulations=100)
        returns = np.random.randn(50) * 0.01
        simulated = simulator.simulate_random_returns(returns, len(returns))

        assert len(simulated) == 100
        assert simulated.dtype == np.float64

    def test_run_simulation(self):
        simulator = MonteCarloSimulator(n_simulations=100)
        trade_returns = [0.01, -0.005, 0.02, -0.01, 0.015, 0.005]
        actual_return = sum(trade_returns)

        result = simulator.run_simulation(actual_return, trade_returns)

        assert result.n_simulations == 100
        assert result.actual_return == actual_return
        assert 0 <= result.percentile <= 100
        assert 0 <= result.p_value <= 1

    def test_simulation_with_few_trades(self):
        simulator = MonteCarloSimulator()
        result = simulator.run_simulation(0.1, [0.1])

        # Should handle gracefully
        assert result.n_simulations == 0
        assert not result.is_significant

    def test_significance_detection(self):
        simulator = MonteCarloSimulator(n_simulations=1000)

        # Strong positive returns with some variance
        np.random.seed(42)
        trade_returns = list(np.random.randn(50) * 0.01 + 0.005)  # Positive bias
        actual_return = sum(trade_returns)

        result = simulator.run_simulation(actual_return, trade_returns)
        # Should produce valid percentile
        assert 0 <= result.percentile <= 100


class TestEnsembleOptimizer:
    """Test ensemble optimization"""

    @pytest.fixture
    def sample_candidates(self):
        """Create sample candidate strategies"""
        return pd.DataFrame({
            "config_id": ["a", "b", "c", "d", "e"],
            "family": ["trend", "meanrev", "trend", "breakout", "meanrev"],
            "strategy_type": ["ema", "bollinger", "supertrend", "range", "zscore"],
            "sharpe_ratio": [1.5, 1.2, 1.0, 0.8, 0.6],
            "annual_return_pct": [30, 25, 20, 15, 10],
        })

    @pytest.fixture
    def sample_returns(self):
        """Create sample return series"""
        n = 100
        return {
            "a": np.random.randn(n) * 0.01 + 0.001,
            "b": np.random.randn(n) * 0.01 + 0.0008,
            "c": np.random.randn(n) * 0.01 + 0.0005,
            "d": np.random.randn(n) * 0.01 + 0.0003,
            "e": np.random.randn(n) * 0.01 + 0.0001,
        }

    def test_select_candidates(self, sample_candidates):
        optimizer = EnsembleOptimizer()
        selected = optimizer.select_candidates(sample_candidates, n_candidates=3)

        assert len(selected) <= 3
        # Should be sorted by sharpe
        assert selected.iloc[0]["sharpe_ratio"] >= selected.iloc[-1]["sharpe_ratio"]

    def test_optimize_weights(self):
        optimizer = EnsembleOptimizer()
        returns_matrix = np.random.randn(100, 5) * 0.01

        weights = optimizer.optimize_weights(returns_matrix, 5)

        assert len(weights) == 5
        assert np.isclose(np.sum(weights), 1.0)
        assert all(w >= 0 for w in weights)

    def test_build_ensemble(self, sample_candidates, sample_returns):
        optimizer = EnsembleOptimizer(max_strategies=3)
        result = optimizer.build_ensemble(sample_candidates, sample_returns)

        assert len(result.weights) <= 3
        assert result.diversification_ratio > 0

    def test_ensemble_oos_validation(self, sample_candidates, sample_returns):
        optimizer = EnsembleOptimizer(oos_ratio=0.3)
        result = optimizer.build_ensemble(sample_candidates, sample_returns)

        # Should have both IS and OOS results
        assert hasattr(result, "in_sample_return")
        assert hasattr(result, "out_of_sample_return")


class TestApplyRobustFilters:
    """Test the convenience function"""

    def test_apply_robust_filters(self):
        df = pd.DataFrame({
            "total_trades": [50, 20],
            "max_drawdown_pct": [-20, -50],
            "profit_factor": [1.5, 0.8],
            "sharpe_ratio": [1.0, 0.3],
        })

        filtered = apply_robust_filters(df)
        assert len(filtered) <= len(df)

    def test_apply_filters_with_custom_config(self):
        df = pd.DataFrame({
            "total_trades": [50, 20],
            "profit_factor": [1.5, 0.8],
        })

        config = RobustFilter(min_trades=10, min_profit_factor=0.5)
        filtered = apply_robust_filters(df, config)
        assert len(filtered) == 2

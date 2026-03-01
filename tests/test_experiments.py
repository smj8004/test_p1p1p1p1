"""
Tests for experiment framework

Smoke tests with minimal data to ensure experiments run without errors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trader.experiments import (
    CostStressExperiment,
    ExperimentConfig,
    ExperimentType,
    RegimeGateExperiment,
    WalkForwardExperiment,
)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Generate minimal sample OHLCV data for testing"""
    np.random.seed(42)

    dates = pd.date_range("2024-01-01", periods=500, freq="1h")
    close = 100 + np.cumsum(np.random.randn(500) * 0.5)
    high = close + np.abs(np.random.randn(500) * 0.2)
    low = close - np.abs(np.random.randn(500) * 0.2)
    open_price = close + np.random.randn(500) * 0.1
    volume = np.random.rand(500) * 1000

    df = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )

    return df


@pytest.fixture
def base_config() -> ExperimentConfig:
    """Base experiment configuration"""
    return ExperimentConfig(
        experiment_type=ExperimentType.COST_STRESS,
        experiment_id="test_001",
        strategy_name="ema_cross",
        strategy_params={"short_window": 12, "long_window": 26, "allow_short": True},
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-01-15",
        seed=42,
    )


def test_cost_stress_generates_scenarios(base_config, sample_data):
    """Test that cost stress generates scenarios correctly"""
    config = base_config
    config.type_specific = {
        "fee_multipliers": [1.0, 2.0],
        "slippage_modes": ["fixed"],
        "latency_bars": [0, 1],
    }

    experiment = CostStressExperiment(config, sample_data)
    scenarios = experiment.generate_scenarios()

    assert len(scenarios) > 0
    assert all(hasattr(s, "fee_multiplier") for s in scenarios)
    assert all(hasattr(s, "slippage_bps") for s in scenarios)
    assert all(hasattr(s, "latency_bars") for s in scenarios)


def test_cost_stress_runs_without_error(base_config, sample_data):
    """Test that cost stress experiment runs without error"""
    config = base_config
    config.type_specific = {
        "fee_multipliers": [1.0],
        "slippage_modes": ["fixed"],
        "latency_bars": [0],
    }

    experiment = CostStressExperiment(config, sample_data)
    result = experiment.run()

    assert result is not None
    assert result.verdict in ["HAS EDGE", "UNCERTAIN", "NO EDGE"]
    assert 0 <= result.robustness_score <= 1
    assert len(result.scenarios) > 0


def test_wfo_creates_splits(base_config, sample_data):
    """Test that WFO creates splits correctly"""
    config = base_config
    config.experiment_type = ExperimentType.WALK_FORWARD
    config.type_specific = {
        "train_days": 10,
        "test_days": 5,
        "n_splits": 3,
        "top_k": 5,
    }

    experiment = WalkForwardExperiment(config, sample_data)
    splits = experiment.create_splits()

    assert len(splits) > 0
    for split in splits:
        assert len(split.train_df) > 0
        assert len(split.test_df) > 0
        assert split.train_start < split.train_end
        assert split.test_start < split.test_end


def test_wfo_runs_without_error(base_config, sample_data):
    """Test that WFO experiment runs without error"""
    config = base_config
    config.experiment_type = ExperimentType.WALK_FORWARD
    config.type_specific = {
        "train_days": 10,
        "test_days": 5,
        "n_splits": 2,
        "top_k": 3,
    }

    # Simple param grid
    param_grid = {
        "short_window": [8, 12],
        "long_window": [20, 26],
    }

    experiment = WalkForwardExperiment(config, sample_data, param_grid=param_grid)
    result = experiment.run()

    assert result is not None
    assert result.verdict in ["HAS EDGE", "UNCERTAIN", "NO EDGE"]
    assert 0 <= result.robustness_score <= 1


def test_regime_detects_regimes(base_config, sample_data):
    """Test that regime detector works"""
    config = base_config
    config.experiment_type = ExperimentType.REGIME_GATE
    config.type_specific = {
        "regime_mode": "trend",
        "gating_mode": "on_off",
    }

    experiment = RegimeGateExperiment(config, sample_data)
    regimes = experiment.detector.detect_regimes(sample_data)

    assert len(regimes) == len(sample_data)
    assert all(hasattr(r, "trend") for r in regimes)
    assert all(hasattr(r, "volatility") for r in regimes)


def test_regime_runs_without_error(base_config, sample_data):
    """Test that regime experiment runs without error"""
    config = base_config
    config.experiment_type = ExperimentType.REGIME_GATE
    config.type_specific = {
        "regime_mode": "trend",
        "gating_mode": "on_off",
    }

    experiment = RegimeGateExperiment(config, sample_data)
    result = experiment.run()

    assert result is not None
    assert result.verdict in ["HAS EDGE", "UNCERTAIN", "NO EDGE"]
    assert 0 <= result.robustness_score <= 1
    assert len(result.scenarios) > 0


def test_experiment_config_serialization(base_config):
    """Test that experiment config can be serialized"""
    config_dict = base_config.to_dict()

    assert config_dict["experiment_id"] == "test_001"
    assert config_dict["strategy_name"] == "ema_cross"
    assert config_dict["symbol"] == "BTC/USDT"
    assert "strategy_params" in config_dict


def test_scenario_result_serialization(base_config, sample_data):
    """Test that scenario results can be serialized"""
    config = base_config
    config.type_specific = {
        "fee_multipliers": [1.0],
        "slippage_modes": ["fixed"],
        "latency_bars": [0],
    }

    experiment = CostStressExperiment(config, sample_data)
    result = experiment.run()

    # Check first scenario
    scenario = result.scenarios[0]
    scenario_dict = scenario.to_dict()

    assert "scenario_id" in scenario_dict
    assert "net_pnl" in scenario_dict
    assert "sharpe_ratio" in scenario_dict
    assert "trade_count" in scenario_dict


def test_experiment_result_serialization(base_config, sample_data):
    """Test that experiment result can be serialized"""
    config = base_config
    config.type_specific = {
        "fee_multipliers": [1.0],
        "slippage_modes": ["fixed"],
        "latency_bars": [0],
    }

    experiment = CostStressExperiment(config, sample_data)
    result = experiment.run()

    result_dict = result.to_dict()

    assert "config" in result_dict
    assert "summary" in result_dict
    assert "scenarios" in result_dict
    assert "robustness_score" in result_dict
    assert "verdict" in result_dict


def test_verdict_calculation():
    """Test verdict calculation logic"""
    from trader.experiments.core import ExperimentResult

    assert ExperimentResult.calculate_verdict(0.8) == "HAS EDGE"
    assert ExperimentResult.calculate_verdict(0.7) == "HAS EDGE"
    assert ExperimentResult.calculate_verdict(0.6) == "UNCERTAIN"
    assert ExperimentResult.calculate_verdict(0.4) == "UNCERTAIN"
    assert ExperimentResult.calculate_verdict(0.3) == "NO EDGE"
    assert ExperimentResult.calculate_verdict(0.0) == "NO EDGE"


def test_cost_stress_with_atr_slippage(base_config, sample_data):
    """Test cost stress with ATR-based slippage"""
    config = base_config
    config.type_specific = {
        "fee_multipliers": [1.0],
        "slippage_modes": ["atr"],
        "latency_bars": [0],
    }

    experiment = CostStressExperiment(config, sample_data)
    result = experiment.run()

    assert result is not None
    assert len(result.scenarios) > 0

    # Check that ATR slippage scenarios were created
    atr_scenarios = [
        s for s in result.scenarios if s.scenario_params.get("slippage_type") == "atr"
    ]
    assert len(atr_scenarios) > 0


def test_wfo_param_stability(base_config, sample_data):
    """Test parameter stability calculation"""
    config = base_config
    config.experiment_type = ExperimentType.WALK_FORWARD
    config.type_specific = {
        "train_days": 10,
        "test_days": 5,
        "n_splits": 2,
        "top_k": 3,
    }

    param_grid = {
        "short_window": [8, 12],
        "long_window": [20, 26],
    }

    experiment = WalkForwardExperiment(config, sample_data, param_grid=param_grid)
    result = experiment.run()

    # Check that param_stability_score is in summary
    assert "param_stability_score" in result.summary
    assert 0 <= result.summary["param_stability_score"] <= 1


def test_regime_baseline_vs_gated(base_config, sample_data):
    """Test that baseline and gated scenarios are both generated"""
    config = base_config
    config.experiment_type = ExperimentType.REGIME_GATE
    config.type_specific = {
        "regime_mode": "trend",
        "gating_mode": "on_off",
    }

    experiment = RegimeGateExperiment(config, sample_data)
    result = experiment.run()

    # Check that baseline exists
    baseline = [s for s in result.scenarios if s.scenario_id == "baseline"]
    assert len(baseline) == 1

    # Check that regime scenarios exist
    regime_scenarios = [s for s in result.scenarios if s.scenario_id != "baseline"]
    assert len(regime_scenarios) > 0

"""
Tests for grid configuration loading
"""

import pytest
import tempfile
from pathlib import Path
import yaml

from trader.massive_backtest import (
    GridConfigLoader,
    BacktestConfig,
    CostProfile,
    COST_PROFILES,
    generate_config_id,
)


class TestCostProfile:
    """Test cost profile"""

    def test_cost_profile_slippage_conversion(self):
        profile = CostProfile("test", 5, 4)
        assert profile.slippage_pct == 0.0005  # 5 bps = 0.05%
        assert profile.fee_pct == 0.0004  # 4 bps = 0.04%

    def test_predefined_profiles(self):
        assert "conservative" in COST_PROFILES
        assert "base" in COST_PROFILES
        assert "aggressive" in COST_PROFILES

        # Conservative should have higher costs
        assert COST_PROFILES["conservative"].slippage_bps > COST_PROFILES["aggressive"].slippage_bps


class TestBacktestConfig:
    """Test backtest configuration"""

    def test_config_creation(self):
        config = BacktestConfig(
            config_id="test123",
            family="trend",
            strategy_type="ema_cross",
            params={"fast": 12, "slow": 26},
            symbol="BTCUSDT",
            timeframe="1h",
            leverage=1,
            allow_short=True,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            cost_profile="base",
            price_source="next_open",
        )

        assert config.config_id == "test123"
        assert config.family == "trend"
        assert config.params["fast"] == 12

    def test_config_to_dict(self):
        config = BacktestConfig(
            config_id="test123",
            family="trend",
            strategy_type="ema_cross",
            params={"fast": 12},
            symbol="BTCUSDT",
            timeframe="1h",
            leverage=1,
            allow_short=True,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            cost_profile="base",
            price_source="next_open",
        )

        d = config.to_dict()
        assert d["config_id"] == "test123"
        assert d["family"] == "trend"
        assert "params" in d


class TestGenerateConfigId:
    """Test config ID generation"""

    def test_same_config_same_id(self):
        config1 = BacktestConfig(
            config_id="",
            family="trend",
            strategy_type="ema_cross",
            params={"fast": 12, "slow": 26},
            symbol="BTCUSDT",
            timeframe="1h",
            leverage=1,
            allow_short=True,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            cost_profile="base",
            price_source="next_open",
        )

        config2 = BacktestConfig(
            config_id="",
            family="trend",
            strategy_type="ema_cross",
            params={"fast": 12, "slow": 26},
            symbol="BTCUSDT",
            timeframe="1h",
            leverage=1,
            allow_short=True,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            cost_profile="base",
            price_source="next_open",
        )

        id1 = generate_config_id(config1)
        id2 = generate_config_id(config2)

        assert id1 == id2

    def test_different_config_different_id(self):
        config1 = BacktestConfig(
            config_id="",
            family="trend",
            strategy_type="ema_cross",
            params={"fast": 12, "slow": 26},
            symbol="BTCUSDT",
            timeframe="1h",
            leverage=1,
            allow_short=True,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            cost_profile="base",
            price_source="next_open",
        )

        config2 = BacktestConfig(
            config_id="",
            family="trend",
            strategy_type="ema_cross",
            params={"fast": 10, "slow": 30},  # Different params
            symbol="BTCUSDT",
            timeframe="1h",
            leverage=1,
            allow_short=True,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            cost_profile="base",
            price_source="next_open",
        )

        id1 = generate_config_id(config1)
        id2 = generate_config_id(config2)

        assert id1 != id2


class TestGridConfigLoader:
    """Test grid config loading"""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temp directory with test config"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)

            # Create test config
            test_config = {
                "family": "test",
                "strategies": {
                    "test_strategy": {
                        "param1": [1, 2],
                        "param2": [10, 20],
                    },
                },
                "risk": {
                    "stop_loss_pct": [0.01, 0.02],
                    "take_profit_pct": [0.02, 0.04],
                },
                "position": {
                    "leverage": [1, 2],
                    "allow_short": [True],
                },
                "symbols": ["BTCUSDT"],
                "timeframes": ["1h"],
                "price_source": ["next_open"],
                "costs": {
                    "profiles": {
                        "base": {
                            "slippage_bps": 3,
                            "fee_taker_bps": 4,
                        },
                    },
                },
            }

            with open(config_dir / "test.yaml", "w") as f:
                yaml.dump(test_config, f)

            yield config_dir

    def test_load_family_grid(self, temp_config_dir):
        loader = GridConfigLoader(temp_config_dir)
        grid = loader.load_family_grid("test")

        assert grid["family"] == "test"
        assert "strategies" in grid
        assert "test_strategy" in grid["strategies"]

    def test_expand_grid(self, temp_config_dir):
        loader = GridConfigLoader(temp_config_dir)
        grid = loader.load_family_grid("test")
        configs = loader.expand_grid(grid)

        # Should generate all combinations
        # 2 param1 * 2 param2 * 2 sl * 2 tp * 2 leverage = 32 configs
        assert len(configs) > 0

        for config in configs:
            assert config.family == "test"
            assert config.strategy_type == "test_strategy"
            assert config.config_id != ""

    def test_load_nonexistent_family(self, temp_config_dir):
        loader = GridConfigLoader(temp_config_dir)

        with pytest.raises(FileNotFoundError):
            loader.load_family_grid("nonexistent")

    def test_expand_grid_generates_unique_ids(self, temp_config_dir):
        loader = GridConfigLoader(temp_config_dir)
        grid = loader.load_family_grid("test")
        configs = loader.expand_grid(grid)

        ids = [c.config_id for c in configs]
        # All IDs should be unique
        assert len(ids) == len(set(ids))


class TestGridExpansionCombinations:
    """Test that grid expansion generates correct combinations"""

    def test_expansion_counts(self):
        """Test that expansion generates expected number of combinations"""
        grid = {
            "family": "test",
            "strategies": {
                "strat1": {
                    "a": [1, 2, 3],
                    "b": [10, 20],
                },
            },
            "risk": {
                "stop_loss_pct": [0.01],
                "take_profit_pct": [0.02],
            },
            "position": {
                "leverage": [1],
                "allow_short": [True],
            },
            "symbols": ["SYM1", "SYM2"],
            "timeframes": ["1h"],
            "price_source": ["next_open"],
            "costs": {
                "profiles": {
                    "base": {},
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            with open(config_dir / "test.yaml", "w") as f:
                yaml.dump(grid, f)

            loader = GridConfigLoader(config_dir)
            loaded_grid = loader.load_family_grid("test")
            configs = loader.expand_grid(loaded_grid)

            # 3 a values * 2 b values * 2 symbols = 12 configs
            expected = 3 * 2 * 2
            assert len(configs) == expected

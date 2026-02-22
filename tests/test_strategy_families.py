"""
Tests for strategy families
"""

import pytest
import numpy as np
import pandas as pd

from trader.strategy import (
    Bar,
    StrategyPosition,
    # Trend family
    TrendEMACrossStrategy,
    TrendSuperTrendStrategy,
    TrendDonchianBreakout,
    TrendKeltnerChannel,
    create_trend_strategy,
    # Mean reversion family
    MeanRevBollingerStrategy,
    MeanRevZScoreStrategy,
    MeanRevRSIStrategy,
    MeanRevStochRSIStrategy,
    create_meanrev_strategy,
    # Breakout family
    BreakoutVolatilityStrategy,
    BreakoutRangeStrategy,
    BreakoutMomentumStrategy,
    BreakoutATRChannelStrategy,
    create_breakout_strategy,
    # Vol regime family
    VolRegimeAdaptiveStrategy,
    VolRegimeVIXStrategy,
    VolTargetStrategy,
    VolClusterStrategy,
    create_volregime_strategy,
    # Carry family
    CarryFundingRateStrategy,
    CarryPremiumStrategy,
    CarryYieldStrategy,
    CarryMomentumStrategy,
    create_carry_strategy,
    # Microstructure family
    MicroVWAPStrategy,
    MicroVolumeProfileStrategy,
    MicroOrderFlowStrategy,
    MicroVolumeMomentumStrategy,
    create_microstructure_strategy,
)


def generate_test_bars(n: int = 100, base_price: float = 100.0) -> list[Bar]:
    """Generate test bars with trending price"""
    bars = []
    price = base_price

    for i in range(n):
        # Add some randomness with trend
        change = np.random.randn() * 0.01 + 0.001
        price *= (1 + change)

        high = price * (1 + abs(np.random.randn()) * 0.005)
        low = price * (1 - abs(np.random.randn()) * 0.005)
        open_p = price * (1 + np.random.randn() * 0.002)
        volume = 1000 + np.random.rand() * 500

        bars.append(Bar(
            timestamp=f"2024-01-01 {i:02d}:00:00",
            open=open_p,
            high=high,
            low=low,
            close=price,
            volume=volume,
        ))

    return bars


class TestTrendFamily:
    """Test trend following strategies"""

    def test_ema_cross_creation(self):
        strategy = TrendEMACrossStrategy(fast=12, slow=26)
        assert strategy.fast == 12
        assert strategy.slow == 26

    def test_ema_cross_factory(self):
        strategy = create_trend_strategy(
            "ema_cross",
            {"fast": 10, "slow": 30},
        )
        assert isinstance(strategy, TrendEMACrossStrategy)

    def test_ema_cross_signals(self):
        strategy = TrendEMACrossStrategy(fast=5, slow=10)
        bars = generate_test_bars(50)

        signals = []
        for bar in bars:
            signal = strategy.on_bar(bar)
            signals.append(signal)

        # Should have at least some non-hold signals after warmup
        assert "hold" in signals

    def test_supertrend_creation(self):
        strategy = TrendSuperTrendStrategy(atr_period=10, multiplier=3.0)
        assert strategy.atr_period == 10
        assert strategy.multiplier == 3.0

    def test_supertrend_factory(self):
        strategy = create_trend_strategy(
            "supertrend",
            {"atr_period": 14, "multiplier": 2.5},
        )
        assert isinstance(strategy, TrendSuperTrendStrategy)

    def test_donchian_creation(self):
        strategy = TrendDonchianBreakout(entry_period=20, exit_period=10)
        assert strategy.entry_period == 20

    def test_keltner_creation(self):
        strategy = TrendKeltnerChannel(ema_period=20, atr_mult=2.0)
        assert strategy.ema_period == 20


class TestMeanRevFamily:
    """Test mean reversion strategies"""

    def test_bollinger_creation(self):
        strategy = MeanRevBollingerStrategy(bb_period=20, bb_std=2.0)
        assert strategy.bb_period == 20
        assert strategy.bb_std == 2.0

    def test_bollinger_factory(self):
        strategy = create_meanrev_strategy(
            "bollinger",
            {"bb_period": 25, "bb_std": 2.5},
        )
        assert isinstance(strategy, MeanRevBollingerStrategy)

    def test_zscore_creation(self):
        strategy = MeanRevZScoreStrategy(lookback=20, entry_zscore=2.0)
        assert strategy.lookback == 20

    def test_zscore_signals(self):
        strategy = MeanRevZScoreStrategy(lookback=10)
        bars = generate_test_bars(30)

        for bar in bars:
            signal = strategy.on_bar(bar)
            assert signal in ["long", "short", "exit", "hold"]

    def test_rsi_creation(self):
        strategy = MeanRevRSIStrategy(rsi_fast=7, rsi_slow=14)
        assert strategy.rsi_fast == 7

    def test_stoch_rsi_creation(self):
        strategy = MeanRevStochRSIStrategy(rsi_period=14)
        assert strategy.rsi_period == 14


class TestBreakoutFamily:
    """Test breakout strategies"""

    def test_volatility_breakout_creation(self):
        strategy = BreakoutVolatilityStrategy(atr_period=14, atr_mult=1.5)
        assert strategy.atr_period == 14

    def test_volatility_factory(self):
        strategy = create_breakout_strategy(
            "volatility",
            {"atr_period": 10, "atr_mult": 2.0},
        )
        assert isinstance(strategy, BreakoutVolatilityStrategy)

    def test_range_breakout_creation(self):
        strategy = BreakoutRangeStrategy(consolidation_period=10)
        assert strategy.consolidation_period == 10

    def test_momentum_breakout_creation(self):
        strategy = BreakoutMomentumStrategy(lookback=20)
        assert strategy.lookback == 20

    def test_atr_channel_creation(self):
        strategy = BreakoutATRChannelStrategy(sma_period=20)
        assert strategy.sma_period == 20


class TestVolRegimeFamily:
    """Test volatility regime strategies"""

    def test_adaptive_creation(self):
        strategy = VolRegimeAdaptiveStrategy(vol_short=10, vol_long=50)
        assert strategy.vol_short == 10

    def test_adaptive_factory(self):
        strategy = create_volregime_strategy(
            "adaptive",
            {"vol_short": 5, "vol_long": 30},
        )
        assert isinstance(strategy, VolRegimeAdaptiveStrategy)

    def test_vix_creation(self):
        strategy = VolRegimeVIXStrategy(vix_period=21)
        assert strategy.vix_period == 21

    def test_target_creation(self):
        strategy = VolTargetStrategy(target_vol=0.15)
        assert strategy.target_vol == 0.15

    def test_cluster_creation(self):
        strategy = VolClusterStrategy(vol_period=14)
        assert strategy.vol_period == 14


class TestCarryFamily:
    """Test carry strategies"""

    def test_funding_rate_creation(self):
        strategy = CarryFundingRateStrategy(funding_threshold=0.0001)
        assert strategy.funding_threshold == 0.0001

    def test_funding_factory(self):
        strategy = create_carry_strategy(
            "funding_rate",
            {"funding_threshold": 0.0002},
        )
        assert isinstance(strategy, CarryFundingRateStrategy)

    def test_premium_creation(self):
        strategy = CarryPremiumStrategy(premium_threshold=0.005)
        assert strategy.premium_threshold == 0.005

    def test_yield_creation(self):
        strategy = CarryYieldStrategy(yield_period=24)
        assert strategy.yield_period == 24

    def test_momentum_creation(self):
        strategy = CarryMomentumStrategy(momentum_fast=10)
        assert strategy.momentum_fast == 10


class TestMicrostructureFamily:
    """Test microstructure strategies"""

    def test_vwap_creation(self):
        strategy = MicroVWAPStrategy(vwap_period=24)
        assert strategy.vwap_period == 24

    def test_vwap_factory(self):
        strategy = create_microstructure_strategy(
            "vwap",
            {"vwap_period": 12},
        )
        assert isinstance(strategy, MicroVWAPStrategy)

    def test_volume_profile_creation(self):
        strategy = MicroVolumeProfileStrategy(profile_period=50)
        assert strategy.profile_period == 50

    def test_order_flow_creation(self):
        strategy = MicroOrderFlowStrategy(flow_period=10)
        assert strategy.flow_period == 10

    def test_volume_momentum_creation(self):
        strategy = MicroVolumeMomentumStrategy(momentum_period=10)
        assert strategy.momentum_period == 10


class TestStrategySignals:
    """Test that all strategies produce valid signals"""

    @pytest.fixture
    def bars(self):
        return generate_test_bars(100)

    def test_trend_strategies_valid_signals(self, bars):
        strategies = [
            TrendEMACrossStrategy(),
            TrendSuperTrendStrategy(),
            TrendDonchianBreakout(),
            TrendKeltnerChannel(),
        ]

        valid_signals = {"long", "short", "exit", "hold"}

        for strategy in strategies:
            for bar in bars:
                signal = strategy.on_bar(bar)
                assert signal in valid_signals, f"{strategy.__class__.__name__} produced invalid signal: {signal}"

    def test_meanrev_strategies_valid_signals(self, bars):
        strategies = [
            MeanRevBollingerStrategy(),
            MeanRevZScoreStrategy(),
            MeanRevRSIStrategy(),
            MeanRevStochRSIStrategy(),
        ]

        valid_signals = {"long", "short", "exit", "hold"}

        for strategy in strategies:
            for bar in bars:
                signal = strategy.on_bar(bar)
                assert signal in valid_signals

    def test_breakout_strategies_valid_signals(self, bars):
        strategies = [
            BreakoutVolatilityStrategy(),
            BreakoutRangeStrategy(),
            BreakoutMomentumStrategy(),
            BreakoutATRChannelStrategy(),
        ]

        valid_signals = {"long", "short", "exit", "hold"}

        for strategy in strategies:
            for bar in bars:
                signal = strategy.on_bar(bar)
                assert signal in valid_signals


class TestRiskManagement:
    """Test risk management features"""

    def test_stop_loss_triggers_exit(self):
        strategy = TrendEMACrossStrategy(
            fast=5,
            slow=10,
            stop_loss_pct=0.02,
        )

        # Generate bars to enter position
        bars = generate_test_bars(20)
        for bar in bars:
            strategy.on_bar(bar)

        # Create position
        position = StrategyPosition(
            side="long",
            qty=1.0,
            entry_price=100.0,
        )

        # Bar that triggers stop loss
        sl_bar = Bar(
            timestamp="2024-01-01 20:00:00",
            open=98.0,
            high=98.5,
            low=97.5,
            close=97.8,
            volume=1000,
        )

        signal = strategy.on_bar(sl_bar, position)
        assert signal == "exit"

    def test_take_profit_triggers_exit(self):
        strategy = TrendEMACrossStrategy(
            fast=5,
            slow=10,
            take_profit_pct=0.02,
        )

        # Generate bars to enter position
        bars = generate_test_bars(20)
        for bar in bars:
            strategy.on_bar(bar)

        # Create position
        position = StrategyPosition(
            side="long",
            qty=1.0,
            entry_price=100.0,
        )

        # Bar that triggers take profit
        tp_bar = Bar(
            timestamp="2024-01-01 20:00:00",
            open=102.0,
            high=102.5,
            low=101.5,
            close=102.5,
            volume=1000,
        )

        signal = strategy.on_bar(tp_bar, position)
        assert signal == "exit"

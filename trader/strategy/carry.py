"""
Carry Strategy Family

Based on:
- Funding rate arbitrage (crypto specific)
- Interest rate carry (adapted for crypto)
- Premium/discount based trading
- Contango/Backwardation concepts
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal

from .base import Bar, Signal, Strategy, StrategyPosition


class CarryFundingRateStrategy(Strategy):
    """
    Funding rate based carry strategy

    Positive funding = longs pay shorts -> favor short
    Negative funding = shorts pay longs -> favor long
    """

    name = "CarryFundingRate"

    def __init__(
        self,
        funding_threshold: float = 0.0001,  # 0.01% threshold
        ema_period: int = 20,
        lookback: int = 24,  # 24 bars for funding avg (if 1h = 24h)
        *,
        allow_short: bool = True,
        stop_loss_pct: float = 0.015,
        take_profit_pct: float = 0.02,
    ) -> None:
        self.funding_threshold = funding_threshold
        self.ema_period = ema_period
        self.lookback = lookback
        self.allow_short = allow_short
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self._prices: list[float] = []
        self._funding_rates: list[float] = []
        self._last_funding: float = 0.0

    def set_funding_rate(self, rate: float) -> None:
        """Set current funding rate (called externally)"""
        self._last_funding = rate
        self._funding_rates.append(rate)

    def _risk_exit(self, bar: Bar, position: StrategyPosition | None) -> bool:
        if position is None or position.side == "flat" or position.entry_price <= 0:
            return False

        if position.side == "long":
            if self.stop_loss_pct > 0 and bar.close <= position.entry_price * (1 - self.stop_loss_pct):
                return True
            if self.take_profit_pct > 0 and bar.close >= position.entry_price * (1 + self.take_profit_pct):
                return True
        elif position.side == "short":
            if self.stop_loss_pct > 0 and bar.close >= position.entry_price * (1 + self.stop_loss_pct):
                return True
            if self.take_profit_pct > 0 and bar.close <= position.entry_price * (1 - self.take_profit_pct):
                return True
        return False

    def on_bar(self, bar: Bar, position: StrategyPosition | None = None) -> Signal:
        self._prices.append(bar.close)

        if self._risk_exit(bar, position):
            return "exit"

        if len(self._prices) < self.ema_period:
            return "hold"

        # Average funding rate
        if len(self._funding_rates) >= self.lookback:
            avg_funding = np.mean(self._funding_rates[-self.lookback:])
        else:
            avg_funding = self._last_funding

        # Trend filter
        ema = np.mean(self._prices[-self.ema_period:])

        signal: Signal = "hold"

        # High positive funding (longs pay shorts)
        if avg_funding > self.funding_threshold:
            if bar.close < ema:  # Align with price trend
                signal = "short" if self.allow_short else "hold"

        # High negative funding (shorts pay longs)
        elif avg_funding < -self.funding_threshold:
            if bar.close > ema:
                signal = "long"

        # Exit when funding normalizes
        if position and position.side == "long" and avg_funding > self.funding_threshold * 0.5:
            signal = "exit"
        elif position and position.side == "short" and avg_funding < -self.funding_threshold * 0.5:
            signal = "exit"

        return signal


class CarryPremiumStrategy(Strategy):
    """
    Futures premium/discount based strategy

    Premium (futures > spot) = contango, expect convergence down
    Discount (futures < spot) = backwardation, expect convergence up
    """

    name = "CarryPremium"

    def __init__(
        self,
        premium_threshold: float = 0.005,  # 0.5% premium threshold
        lookback: int = 24,
        ema_period: int = 20,
        *,
        allow_short: bool = True,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.03,
    ) -> None:
        self.premium_threshold = premium_threshold
        self.lookback = lookback
        self.ema_period = ema_period
        self.allow_short = allow_short
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self._prices: list[float] = []
        self._premiums: list[float] = []
        self._last_premium: float = 0.0

    def set_premium(self, premium: float) -> None:
        """Set current premium (futures price / spot price - 1)"""
        self._last_premium = premium
        self._premiums.append(premium)

    def _estimate_premium_from_price(self) -> float:
        """Estimate premium from price momentum (proxy when no real data)"""
        if len(self._prices) < self.lookback:
            return 0.0

        short_ma = np.mean(self._prices[-5:])
        long_ma = np.mean(self._prices[-self.lookback:])
        return (short_ma - long_ma) / long_ma

    def _risk_exit(self, bar: Bar, position: StrategyPosition | None) -> bool:
        if position is None or position.side == "flat" or position.entry_price <= 0:
            return False

        if position.side == "long":
            if self.stop_loss_pct > 0 and bar.close <= position.entry_price * (1 - self.stop_loss_pct):
                return True
            if self.take_profit_pct > 0 and bar.close >= position.entry_price * (1 + self.take_profit_pct):
                return True
        elif position.side == "short":
            if self.stop_loss_pct > 0 and bar.close >= position.entry_price * (1 + self.stop_loss_pct):
                return True
            if self.take_profit_pct > 0 and bar.close <= position.entry_price * (1 - self.take_profit_pct):
                return True
        return False

    def on_bar(self, bar: Bar, position: StrategyPosition | None = None) -> Signal:
        self._prices.append(bar.close)

        if self._risk_exit(bar, position):
            return "exit"

        if len(self._prices) < max(self.lookback, self.ema_period):
            return "hold"

        # Use real premium if available, otherwise estimate
        if len(self._premiums) >= self.lookback:
            avg_premium = np.mean(self._premiums[-self.lookback:])
        else:
            avg_premium = self._estimate_premium_from_price()

        ema = np.mean(self._prices[-self.ema_period:])

        signal: Signal = "hold"

        # High premium (contango) - expect price to fall
        if avg_premium > self.premium_threshold:
            if bar.close < ema:
                signal = "short" if self.allow_short else "hold"

        # Low/negative premium (backwardation) - expect price to rise
        elif avg_premium < -self.premium_threshold:
            if bar.close > ema:
                signal = "long"

        # Exit when premium normalizes
        if position and position.side == "long" and avg_premium > self.premium_threshold * 0.5:
            signal = "exit"
        elif position and position.side == "short" and avg_premium < -self.premium_threshold * 0.5:
            signal = "exit"

        return signal


class CarryYieldStrategy(Strategy):
    """
    Yield/Interest rate differential based strategy

    Simulates carry trade concepts for crypto
    Uses rolling returns as proxy for yield
    """

    name = "CarryYield"

    def __init__(
        self,
        yield_period: int = 24,  # Period to calculate yield
        yield_threshold: float = 0.002,  # 0.2% yield threshold
        momentum_period: int = 12,
        *,
        allow_short: bool = True,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
    ) -> None:
        self.yield_period = yield_period
        self.yield_threshold = yield_threshold
        self.momentum_period = momentum_period
        self.allow_short = allow_short
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self._prices: list[float] = []

    def _calc_rolling_yield(self) -> float:
        """Calculate rolling yield (annualized return rate)"""
        if len(self._prices) < self.yield_period + 1:
            return 0.0

        start_price = self._prices[-self.yield_period - 1]
        end_price = self._prices[-1]
        return (end_price - start_price) / start_price

    def _risk_exit(self, bar: Bar, position: StrategyPosition | None) -> bool:
        if position is None or position.side == "flat" or position.entry_price <= 0:
            return False

        if position.side == "long":
            if self.stop_loss_pct > 0 and bar.close <= position.entry_price * (1 - self.stop_loss_pct):
                return True
            if self.take_profit_pct > 0 and bar.close >= position.entry_price * (1 + self.take_profit_pct):
                return True
        elif position.side == "short":
            if self.stop_loss_pct > 0 and bar.close >= position.entry_price * (1 + self.stop_loss_pct):
                return True
            if self.take_profit_pct > 0 and bar.close <= position.entry_price * (1 - self.take_profit_pct):
                return True
        return False

    def on_bar(self, bar: Bar, position: StrategyPosition | None = None) -> Signal:
        self._prices.append(bar.close)

        if self._risk_exit(bar, position):
            return "exit"

        if len(self._prices) < self.yield_period + 1:
            return "hold"

        rolling_yield = self._calc_rolling_yield()

        # Momentum confirmation
        momentum = 0.0
        if len(self._prices) > self.momentum_period:
            momentum = (bar.close - self._prices[-self.momentum_period - 1]) / self._prices[-self.momentum_period - 1]

        signal: Signal = "hold"

        # Positive yield + positive momentum
        if rolling_yield > self.yield_threshold and momentum > 0:
            signal = "long"

        # Negative yield + negative momentum
        elif rolling_yield < -self.yield_threshold and momentum < 0:
            signal = "short" if self.allow_short else "hold"

        # Exit when yield reverses
        if position and position.side == "long" and rolling_yield < 0:
            signal = "exit"
        elif position and position.side == "short" and rolling_yield > 0:
            signal = "exit"

        return signal


class CarryMomentumStrategy(Strategy):
    """
    Combined carry and momentum strategy

    Uses momentum as primary signal, carry for confirmation
    """

    name = "CarryMomentum"

    def __init__(
        self,
        momentum_fast: int = 10,
        momentum_slow: int = 30,
        carry_period: int = 24,
        carry_weight: float = 0.3,
        *,
        allow_short: bool = True,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
    ) -> None:
        self.momentum_fast = momentum_fast
        self.momentum_slow = momentum_slow
        self.carry_period = carry_period
        self.carry_weight = carry_weight
        self.allow_short = allow_short
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self._prices: list[float] = []
        self._funding_rates: list[float] = []

    def set_funding_rate(self, rate: float) -> None:
        """Set current funding rate"""
        self._funding_rates.append(rate)

    def _risk_exit(self, bar: Bar, position: StrategyPosition | None) -> bool:
        if position is None or position.side == "flat" or position.entry_price <= 0:
            return False

        if position.side == "long":
            if self.stop_loss_pct > 0 and bar.close <= position.entry_price * (1 - self.stop_loss_pct):
                return True
            if self.take_profit_pct > 0 and bar.close >= position.entry_price * (1 + self.take_profit_pct):
                return True
        elif position.side == "short":
            if self.stop_loss_pct > 0 and bar.close >= position.entry_price * (1 + self.stop_loss_pct):
                return True
            if self.take_profit_pct > 0 and bar.close <= position.entry_price * (1 - self.take_profit_pct):
                return True
        return False

    def on_bar(self, bar: Bar, position: StrategyPosition | None = None) -> Signal:
        self._prices.append(bar.close)

        if self._risk_exit(bar, position):
            return "exit"

        min_period = max(self.momentum_slow, self.carry_period)
        if len(self._prices) < min_period + 1:
            return "hold"

        # Momentum signals
        mom_fast = (bar.close - self._prices[-self.momentum_fast - 1]) / self._prices[-self.momentum_fast - 1]
        mom_slow = (bar.close - self._prices[-self.momentum_slow - 1]) / self._prices[-self.momentum_slow - 1]

        # Carry signal (from funding or estimated)
        if len(self._funding_rates) >= self.carry_period:
            carry_signal = -np.mean(self._funding_rates[-self.carry_period:])  # Negative because high funding = favor short
        else:
            # Estimate from price mean reversion
            ma = np.mean(self._prices[-self.carry_period:])
            carry_signal = (ma - bar.close) / bar.close * 10  # Scaled

        # Combined score
        momentum_score = (mom_fast + mom_slow) / 2
        carry_score = carry_signal * self.carry_weight
        combined = momentum_score + carry_score

        signal: Signal = "hold"

        if combined > 0.01:
            signal = "long"
        elif combined < -0.01:
            signal = "short" if self.allow_short else "hold"

        return signal


# Factory function
def create_carry_strategy(
    strategy_type: str,
    params: dict,
    allow_short: bool = True,
    stop_loss_pct: float = 0.02,
    take_profit_pct: float = 0.03,
) -> Strategy:
    """Create carry strategy from type and params"""

    if strategy_type == "funding_rate":
        return CarryFundingRateStrategy(
            funding_threshold=params.get("funding_threshold", 0.0001),
            ema_period=params.get("ema_period", 20),
            lookback=params.get("lookback", 24),
            allow_short=allow_short,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
    elif strategy_type == "premium":
        return CarryPremiumStrategy(
            premium_threshold=params.get("premium_threshold", 0.005),
            lookback=params.get("lookback", 24),
            ema_period=params.get("ema_period", 20),
            allow_short=allow_short,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
    elif strategy_type == "yield":
        return CarryYieldStrategy(
            yield_period=params.get("yield_period", 24),
            yield_threshold=params.get("yield_threshold", 0.002),
            momentum_period=params.get("momentum_period", 12),
            allow_short=allow_short,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
    elif strategy_type == "momentum":
        return CarryMomentumStrategy(
            momentum_fast=params.get("momentum_fast", 10),
            momentum_slow=params.get("momentum_slow", 30),
            carry_period=params.get("carry_period", 24),
            carry_weight=params.get("carry_weight", 0.3),
            allow_short=allow_short,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
    else:
        raise ValueError(f"Unknown carry strategy: {strategy_type}")


CARRY_STRATEGIES = {
    "funding_rate": {
        "class": CarryFundingRateStrategy,
        "params": ["funding_threshold", "ema_period", "lookback"],
    },
    "premium": {
        "class": CarryPremiumStrategy,
        "params": ["premium_threshold", "lookback", "ema_period"],
    },
    "yield": {
        "class": CarryYieldStrategy,
        "params": ["yield_period", "yield_threshold", "momentum_period"],
    },
    "momentum": {
        "class": CarryMomentumStrategy,
        "params": ["momentum_fast", "momentum_slow", "carry_period", "carry_weight"],
    },
}

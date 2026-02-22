"""
Breakout Strategy Family

Based on:
- Volatility expansion breakouts
- Support/Resistance breakouts
- Volume-confirmed breakouts
- ATR-based breakouts
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal

from .base import Bar, Signal, Strategy, StrategyPosition


class BreakoutVolatilityStrategy(Strategy):
    """Volatility expansion breakout strategy"""

    name = "BreakoutVolatility"

    def __init__(
        self,
        atr_period: int = 14,
        atr_mult: float = 1.5,
        lookback: int = 20,
        vol_threshold: float = 1.2,
        *,
        allow_short: bool = True,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
    ) -> None:
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.lookback = lookback
        self.vol_threshold = vol_threshold
        self.allow_short = allow_short
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self._prices: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._volumes: list[float] = []

    def _calc_atr(self) -> float:
        if len(self._prices) < self.atr_period + 1:
            return 0.0

        highs = np.array(self._highs[-(self.atr_period + 1):])
        lows = np.array(self._lows[-(self.atr_period + 1):])
        closes = np.array(self._prices[-(self.atr_period + 1):])

        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )
        return np.mean(tr)

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
        self._highs.append(bar.high)
        self._lows.append(bar.low)
        self._volumes.append(bar.volume)

        if self._risk_exit(bar, position):
            return "exit"

        min_period = max(self.atr_period + 1, self.lookback)
        if len(self._prices) < min_period:
            return "hold"

        atr = self._calc_atr()
        range_high = max(self._highs[-(self.lookback + 1):-1])
        range_low = min(self._lows[-(self.lookback + 1):-1])

        # Volume filter
        vol_ma = np.mean(self._volumes[-20:]) if len(self._volumes) >= 20 else bar.volume
        vol_ratio = bar.volume / (vol_ma + 1e-10)

        signal: Signal = "hold"

        # Breakout with volume confirmation
        if vol_ratio > self.vol_threshold:
            if bar.close > range_high + atr * (self.atr_mult - 1):
                signal = "long"
            elif bar.close < range_low - atr * (self.atr_mult - 1):
                signal = "short" if self.allow_short else "hold"

        return signal


class BreakoutRangeStrategy(Strategy):
    """Range breakout with consolidation detection"""

    name = "BreakoutRange"

    def __init__(
        self,
        consolidation_period: int = 10,
        breakout_threshold: float = 0.5,
        atr_period: int = 14,
        *,
        allow_short: bool = True,
        stop_loss_pct: float = 0.015,
        take_profit_pct: float = 0.03,
    ) -> None:
        self.consolidation_period = consolidation_period
        self.breakout_threshold = breakout_threshold
        self.atr_period = atr_period
        self.allow_short = allow_short
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self._prices: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []

    def _calc_atr(self) -> float:
        if len(self._prices) < self.atr_period + 1:
            return 0.0

        highs = np.array(self._highs[-(self.atr_period + 1):])
        lows = np.array(self._lows[-(self.atr_period + 1):])
        closes = np.array(self._prices[-(self.atr_period + 1):])

        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )
        return np.mean(tr)

    def _is_consolidating(self) -> bool:
        """Check if price is in consolidation (low volatility range)"""
        if len(self._highs) < self.consolidation_period:
            return False

        recent_highs = self._highs[-self.consolidation_period:]
        recent_lows = self._lows[-self.consolidation_period:]

        range_high = max(recent_highs)
        range_low = min(recent_lows)
        range_pct = (range_high - range_low) / range_low * 100

        atr = self._calc_atr()
        atr_pct = atr / self._prices[-1] * 100 if self._prices[-1] > 0 else 0

        # Consolidation: narrow range relative to ATR
        return range_pct < atr_pct * 2

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
        self._highs.append(bar.high)
        self._lows.append(bar.low)

        if self._risk_exit(bar, position):
            return "exit"

        min_period = max(self.consolidation_period, self.atr_period + 1)
        if len(self._prices) < min_period:
            return "hold"

        # Check consolidation first
        if not self._is_consolidating():
            return "hold"

        atr = self._calc_atr()
        range_high = max(self._highs[-(self.consolidation_period + 1):-1])
        range_low = min(self._lows[-(self.consolidation_period + 1):-1])

        signal: Signal = "hold"

        # Breakout from consolidation
        if bar.close > range_high + atr * self.breakout_threshold:
            signal = "long"
        elif bar.close < range_low - atr * self.breakout_threshold:
            signal = "short" if self.allow_short else "hold"

        return signal


class BreakoutMomentumStrategy(Strategy):
    """Momentum-confirmed breakout"""

    name = "BreakoutMomentum"

    def __init__(
        self,
        lookback: int = 20,
        momentum_period: int = 10,
        momentum_threshold: float = 1.0,
        volume_mult: float = 1.5,
        *,
        allow_short: bool = True,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
    ) -> None:
        self.lookback = lookback
        self.momentum_period = momentum_period
        self.momentum_threshold = momentum_threshold
        self.volume_mult = volume_mult
        self.allow_short = allow_short
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self._prices: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._volumes: list[float] = []

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
        self._highs.append(bar.high)
        self._lows.append(bar.low)
        self._volumes.append(bar.volume)

        if self._risk_exit(bar, position):
            return "exit"

        min_period = max(self.lookback, self.momentum_period) + 1
        if len(self._prices) < min_period:
            return "hold"

        # Range
        range_high = max(self._highs[-(self.lookback + 1):-1])
        range_low = min(self._lows[-(self.lookback + 1):-1])

        # Momentum
        momentum = (bar.close - self._prices[-self.momentum_period - 1]) / self._prices[-self.momentum_period - 1] * 100

        # Volume
        vol_ma = np.mean(self._volumes[-20:]) if len(self._volumes) >= 20 else bar.volume
        vol_ratio = bar.volume / (vol_ma + 1e-10)

        signal: Signal = "hold"

        # Breakout with momentum and volume confirmation
        if bar.close > range_high:
            if momentum > self.momentum_threshold and vol_ratio > self.volume_mult:
                signal = "long"
        elif bar.close < range_low:
            if momentum < -self.momentum_threshold and vol_ratio > self.volume_mult:
                signal = "short" if self.allow_short else "hold"

        return signal


class BreakoutATRChannelStrategy(Strategy):
    """ATR Channel breakout (similar to Keltner but simpler)"""

    name = "BreakoutATRChannel"

    def __init__(
        self,
        sma_period: int = 20,
        atr_period: int = 14,
        atr_mult: float = 2.0,
        *,
        allow_short: bool = True,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
    ) -> None:
        self.sma_period = sma_period
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.allow_short = allow_short
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self._prices: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._was_inside: bool = True

    def _calc_atr(self) -> float:
        if len(self._prices) < self.atr_period + 1:
            return 0.0

        highs = np.array(self._highs[-(self.atr_period + 1):])
        lows = np.array(self._lows[-(self.atr_period + 1):])
        closes = np.array(self._prices[-(self.atr_period + 1):])

        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )
        return np.mean(tr)

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
        self._highs.append(bar.high)
        self._lows.append(bar.low)

        if self._risk_exit(bar, position):
            return "exit"

        min_period = max(self.sma_period, self.atr_period + 1)
        if len(self._prices) < min_period:
            return "hold"

        sma = np.mean(self._prices[-self.sma_period:])
        atr = self._calc_atr()

        upper_band = sma + self.atr_mult * atr
        lower_band = sma - self.atr_mult * atr

        was_inside = self._was_inside
        is_inside = lower_band <= bar.close <= upper_band
        self._was_inside = is_inside

        signal: Signal = "hold"

        # Breakout from channel
        if was_inside and bar.close > upper_band:
            signal = "long"
        elif was_inside and bar.close < lower_band:
            signal = "short" if self.allow_short else "hold"

        # Exit when price returns to SMA
        if position and position.side == "long" and bar.close < sma:
            signal = "exit"
        elif position and position.side == "short" and bar.close > sma:
            signal = "exit"

        return signal


# Factory function
def create_breakout_strategy(
    strategy_type: str,
    params: dict,
    allow_short: bool = True,
    stop_loss_pct: float = 0.02,
    take_profit_pct: float = 0.04,
) -> Strategy:
    """Create breakout strategy from type and params"""

    if strategy_type == "volatility":
        return BreakoutVolatilityStrategy(
            atr_period=params.get("atr_period", 14),
            atr_mult=params.get("atr_mult", 1.5),
            lookback=params.get("lookback", 20),
            vol_threshold=params.get("vol_threshold", 1.2),
            allow_short=allow_short,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
    elif strategy_type == "range":
        return BreakoutRangeStrategy(
            consolidation_period=params.get("consolidation_period", 10),
            breakout_threshold=params.get("breakout_threshold", 0.5),
            atr_period=params.get("atr_period", 14),
            allow_short=allow_short,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
    elif strategy_type == "momentum":
        return BreakoutMomentumStrategy(
            lookback=params.get("lookback", 20),
            momentum_period=params.get("momentum_period", 10),
            momentum_threshold=params.get("momentum_threshold", 1.0),
            volume_mult=params.get("volume_mult", 1.5),
            allow_short=allow_short,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
    elif strategy_type == "atr_channel":
        return BreakoutATRChannelStrategy(
            sma_period=params.get("sma_period", 20),
            atr_period=params.get("atr_period", 14),
            atr_mult=params.get("atr_mult", 2.0),
            allow_short=allow_short,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
    else:
        raise ValueError(f"Unknown breakout strategy: {strategy_type}")


BREAKOUT_STRATEGIES = {
    "volatility": {
        "class": BreakoutVolatilityStrategy,
        "params": ["atr_period", "atr_mult", "lookback", "vol_threshold"],
    },
    "range": {
        "class": BreakoutRangeStrategy,
        "params": ["consolidation_period", "breakout_threshold", "atr_period"],
    },
    "momentum": {
        "class": BreakoutMomentumStrategy,
        "params": ["lookback", "momentum_period", "momentum_threshold", "volume_mult"],
    },
    "atr_channel": {
        "class": BreakoutATRChannelStrategy,
        "params": ["sma_period", "atr_period", "atr_mult"],
    },
}

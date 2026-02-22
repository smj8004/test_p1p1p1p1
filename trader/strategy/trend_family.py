"""
Trend Following Strategy Family

Based on:
- Jegadeesh & Titman (1993) momentum research
- AQR Time Series Momentum
- CTA trend-following approaches (Winton, Man AHL)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal

from .base import Bar, Signal, Strategy, StrategyPosition


@dataclass
class TrendConfig:
    """Trend strategy configuration"""
    # EMA parameters
    ema_fast: int = 20
    ema_slow: int = 50
    ema_trend: int = 200

    # ADX parameters
    adx_period: int = 14
    adx_threshold: float = 25.0

    # Volume filter
    vol_ma_period: int = 20
    vol_threshold: float = 1.0

    # Risk management
    atr_period: int = 14
    stop_loss_atr: float = 2.0
    take_profit_atr: float = 4.0

    # Mode
    mode: Literal["conservative", "moderate", "aggressive"] = "moderate"


class TrendEMACrossStrategy(Strategy):
    """EMA Cross with trend filter and ADX confirmation"""

    name = "TrendEMACross"

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        trend: int = 200,
        adx_threshold: float = 20.0,
        *,
        allow_short: bool = True,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
    ) -> None:
        self.fast = fast
        self.slow = slow
        self.trend = trend
        self.adx_threshold = adx_threshold
        self.allow_short = allow_short
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self._prices: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._last_signal: Signal = "hold"

    def _calc_adx(self) -> float:
        """Calculate ADX"""
        if len(self._prices) < 28:
            return 0.0

        highs = np.array(self._highs[-28:])
        lows = np.array(self._lows[-28:])
        closes = np.array(self._prices[-28:])

        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )

        up_move = highs[1:] - highs[:-1]
        down_move = lows[:-1] - lows[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        atr = pd.Series(tr).rolling(14).mean().iloc[-1]
        plus_di = 100 * pd.Series(plus_dm).rolling(14).mean().iloc[-1] / (atr + 1e-10)
        minus_di = 100 * pd.Series(minus_dm).rolling(14).mean().iloc[-1] / (atr + 1e-10)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        return dx

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

        if len(self._prices) < max(self.slow, self.trend, 28):
            return "hold"

        series = pd.Series(self._prices, dtype="float64")
        ema_fast = series.ewm(span=self.fast, adjust=False).mean().iloc[-1]
        ema_slow = series.ewm(span=self.slow, adjust=False).mean().iloc[-1]
        ema_trend = series.ewm(span=self.trend, adjust=False).mean().iloc[-1]

        adx = self._calc_adx()

        prev_fast = series.ewm(span=self.fast, adjust=False).mean().iloc[-2]
        prev_slow = series.ewm(span=self.slow, adjust=False).mean().iloc[-2]

        # Crossover detection
        cross_up = prev_fast <= prev_slow and ema_fast > ema_slow
        cross_down = prev_fast >= prev_slow and ema_fast < ema_slow

        # Trend filter
        uptrend = bar.close > ema_trend
        downtrend = bar.close < ema_trend

        # ADX filter
        strong_trend = adx > self.adx_threshold

        signal: Signal = "hold"

        if cross_up and uptrend and strong_trend:
            signal = "long"
        elif cross_down and downtrend and strong_trend:
            signal = "short" if self.allow_short else "exit"
        elif cross_down and position and position.side == "long":
            signal = "exit"
        elif cross_up and position and position.side == "short":
            signal = "exit"

        self._last_signal = signal
        return signal


class TrendSuperTrendStrategy(Strategy):
    """SuperTrend indicator based trend following"""

    name = "TrendSuperTrend"

    def __init__(
        self,
        atr_period: int = 10,
        multiplier: float = 3.0,
        *,
        allow_short: bool = True,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
    ) -> None:
        self.atr_period = atr_period
        self.multiplier = multiplier
        self.allow_short = allow_short
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self._prices: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._supertrend: float = 0.0
        self._direction: int = 1  # 1=up, -1=down

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

        if len(self._prices) < self.atr_period + 2:
            return "hold"

        atr = self._calc_atr()
        hl2 = (bar.high + bar.low) / 2

        upper_band = hl2 + self.multiplier * atr
        lower_band = hl2 - self.multiplier * atr

        prev_direction = self._direction

        if bar.close > self._supertrend:
            self._direction = 1
            self._supertrend = max(lower_band, self._supertrend) if self._direction == 1 else lower_band
        else:
            self._direction = -1
            self._supertrend = min(upper_band, self._supertrend) if self._direction == -1 else upper_band

        signal: Signal = "hold"

        if prev_direction == -1 and self._direction == 1:
            signal = "long"
        elif prev_direction == 1 and self._direction == -1:
            signal = "short" if self.allow_short else "exit"

        return signal


class TrendDonchianBreakout(Strategy):
    """Donchian Channel breakout (Turtle Trading)"""

    name = "TrendDonchian"

    def __init__(
        self,
        entry_period: int = 20,
        exit_period: int = 10,
        *,
        allow_short: bool = True,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.06,
    ) -> None:
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.allow_short = allow_short
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self._highs: list[float] = []
        self._lows: list[float] = []
        self._closes: list[float] = []

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
        self._highs.append(bar.high)
        self._lows.append(bar.low)
        self._closes.append(bar.close)

        if self._risk_exit(bar, position):
            return "exit"

        if len(self._highs) < self.entry_period + 1:
            return "hold"

        # Entry channels (excluding current bar)
        entry_high = max(self._highs[-(self.entry_period + 1):-1])
        entry_low = min(self._lows[-(self.entry_period + 1):-1])

        # Exit channels
        exit_high = max(self._highs[-(self.exit_period + 1):-1]) if len(self._highs) > self.exit_period else entry_high
        exit_low = min(self._lows[-(self.exit_period + 1):-1]) if len(self._lows) > self.exit_period else entry_low

        signal: Signal = "hold"

        # Entry signals
        if bar.close > entry_high:
            if position is None or position.side != "long":
                signal = "long"
        elif bar.close < entry_low:
            if position is None or position.side != "short":
                signal = "short" if self.allow_short else "hold"

        # Exit signals
        if position and position.side == "long" and bar.close < exit_low:
            signal = "exit"
        elif position and position.side == "short" and bar.close > exit_high:
            signal = "exit"

        return signal


class TrendKeltnerChannel(Strategy):
    """Keltner Channel trend following"""

    name = "TrendKeltner"

    def __init__(
        self,
        ema_period: int = 20,
        atr_period: int = 10,
        atr_mult: float = 2.0,
        *,
        allow_short: bool = True,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
    ) -> None:
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.allow_short = allow_short
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self._prices: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._in_channel: bool = True

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

        min_period = max(self.ema_period, self.atr_period + 1)
        if len(self._prices) < min_period:
            return "hold"

        ema = pd.Series(self._prices, dtype="float64").ewm(span=self.ema_period, adjust=False).mean().iloc[-1]
        atr = self._calc_atr()

        upper_band = ema + self.atr_mult * atr
        lower_band = ema - self.atr_mult * atr

        prev_in_channel = self._in_channel
        self._in_channel = lower_band <= bar.close <= upper_band

        signal: Signal = "hold"

        # Breakout signals
        if prev_in_channel and bar.close > upper_band:
            signal = "long"
        elif prev_in_channel and bar.close < lower_band:
            signal = "short" if self.allow_short else "hold"

        # Exit when price returns to EMA
        if position and position.side == "long" and bar.close < ema:
            signal = "exit"
        elif position and position.side == "short" and bar.close > ema:
            signal = "exit"

        return signal


# Factory function for grid search
def create_trend_strategy(
    strategy_type: str,
    params: dict,
    allow_short: bool = True,
    stop_loss_pct: float = 0.02,
    take_profit_pct: float = 0.04,
) -> Strategy:
    """Create trend strategy from type and params"""

    if strategy_type == "ema_cross":
        return TrendEMACrossStrategy(
            fast=params.get("fast", 12),
            slow=params.get("slow", 26),
            trend=params.get("trend", 200),
            adx_threshold=params.get("adx_threshold", 20.0),
            allow_short=allow_short,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
    elif strategy_type == "supertrend":
        return TrendSuperTrendStrategy(
            atr_period=params.get("atr_period", 10),
            multiplier=params.get("multiplier", 3.0),
            allow_short=allow_short,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
    elif strategy_type == "donchian":
        return TrendDonchianBreakout(
            entry_period=params.get("entry_period", 20),
            exit_period=params.get("exit_period", 10),
            allow_short=allow_short,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
    elif strategy_type == "keltner":
        return TrendKeltnerChannel(
            ema_period=params.get("ema_period", 20),
            atr_period=params.get("atr_period", 10),
            atr_mult=params.get("atr_mult", 2.0),
            allow_short=allow_short,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
    else:
        raise ValueError(f"Unknown trend strategy: {strategy_type}")


# Available strategies for grid search
TREND_STRATEGIES = {
    "ema_cross": {
        "class": TrendEMACrossStrategy,
        "params": ["fast", "slow", "trend", "adx_threshold"],
    },
    "supertrend": {
        "class": TrendSuperTrendStrategy,
        "params": ["atr_period", "multiplier"],
    },
    "donchian": {
        "class": TrendDonchianBreakout,
        "params": ["entry_period", "exit_period"],
    },
    "keltner": {
        "class": TrendKeltnerChannel,
        "params": ["ema_period", "atr_period", "atr_mult"],
    },
}

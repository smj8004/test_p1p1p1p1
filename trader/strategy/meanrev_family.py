"""
Mean Reversion Strategy Family

Based on:
- Bollinger Bands mean reversion
- RSI oversold/overbought
- Z-score statistical arbitrage
- Pair trading concepts (modified for single asset)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal

from .base import Bar, Signal, Strategy, StrategyPosition


@dataclass
class MeanRevConfig:
    """Mean reversion configuration"""
    # Bollinger parameters
    bb_period: int = 20
    bb_std: float = 2.0

    # RSI parameters
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70

    # Z-score parameters
    zscore_period: int = 20
    zscore_threshold: float = 2.0

    # Risk management
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.02

    # Mode
    require_trend_filter: bool = True


class MeanRevBollingerStrategy(Strategy):
    """Bollinger Band mean reversion with RSI confirmation"""

    name = "MeanRevBollinger"

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: int = 30,
        rsi_overbought: int = 70,
        *,
        allow_short: bool = True,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.02,
    ) -> None:
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.allow_short = allow_short
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self._prices: list[float] = []

    def _calc_rsi(self) -> float:
        if len(self._prices) < self.rsi_period + 1:
            return 50.0

        prices = np.array(self._prices[-(self.rsi_period + 1):])
        deltas = np.diff(prices)

        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

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

        min_period = max(self.bb_period, self.rsi_period + 1)
        if len(self._prices) < min_period:
            return "hold"

        # Bollinger Bands
        prices = self._prices[-self.bb_period:]
        bb_mid = np.mean(prices)
        bb_std = np.std(prices)
        bb_upper = bb_mid + self.bb_std * bb_std
        bb_lower = bb_mid - self.bb_std * bb_std

        rsi = self._calc_rsi()

        signal: Signal = "hold"

        # Oversold - buy signal
        if bar.close < bb_lower and rsi < self.rsi_oversold:
            signal = "long"
        # Overbought - sell signal
        elif bar.close > bb_upper and rsi > self.rsi_overbought:
            signal = "short" if self.allow_short else "hold"
        # Exit at middle band
        elif position and position.side == "long" and bar.close >= bb_mid:
            signal = "exit"
        elif position and position.side == "short" and bar.close <= bb_mid:
            signal = "exit"

        return signal


class MeanRevZScoreStrategy(Strategy):
    """Z-Score based mean reversion"""

    name = "MeanRevZScore"

    def __init__(
        self,
        lookback: int = 20,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        *,
        allow_short: bool = True,
        stop_loss_pct: float = 0.03,
        take_profit_pct: float = 0.02,
    ) -> None:
        self.lookback = lookback
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.allow_short = allow_short
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self._prices: list[float] = []

    def _calc_zscore(self) -> float:
        if len(self._prices) < self.lookback:
            return 0.0

        prices = np.array(self._prices[-self.lookback:])
        mean = np.mean(prices)
        std = np.std(prices)

        if std == 0:
            return 0.0

        return (self._prices[-1] - mean) / std

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

        if len(self._prices) < self.lookback:
            return "hold"

        zscore = self._calc_zscore()

        signal: Signal = "hold"

        # Entry signals
        if zscore < -self.entry_zscore:
            signal = "long"
        elif zscore > self.entry_zscore:
            signal = "short" if self.allow_short else "hold"

        # Exit signals
        if position and position.side == "long" and zscore > -self.exit_zscore:
            signal = "exit"
        elif position and position.side == "short" and zscore < self.exit_zscore:
            signal = "exit"

        return signal


class MeanRevRSIStrategy(Strategy):
    """Pure RSI mean reversion with multiple timeframe confirmation"""

    name = "MeanRevRSI"

    def __init__(
        self,
        rsi_fast: int = 7,
        rsi_slow: int = 14,
        oversold: int = 30,
        overbought: int = 70,
        *,
        allow_short: bool = True,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.02,
    ) -> None:
        self.rsi_fast = rsi_fast
        self.rsi_slow = rsi_slow
        self.oversold = oversold
        self.overbought = overbought
        self.allow_short = allow_short
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self._prices: list[float] = []

    def _calc_rsi(self, period: int) -> float:
        if len(self._prices) < period + 1:
            return 50.0

        prices = np.array(self._prices[-(period + 1):])
        deltas = np.diff(prices)

        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

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

        min_period = max(self.rsi_fast, self.rsi_slow) + 1
        if len(self._prices) < min_period:
            return "hold"

        rsi_fast = self._calc_rsi(self.rsi_fast)
        rsi_slow = self._calc_rsi(self.rsi_slow)

        signal: Signal = "hold"

        # Both RSIs oversold
        if rsi_fast < self.oversold and rsi_slow < self.oversold + 10:
            signal = "long"
        # Both RSIs overbought
        elif rsi_fast > self.overbought and rsi_slow > self.overbought - 10:
            signal = "short" if self.allow_short else "hold"
        # Exit when fast RSI crosses 50
        elif position and position.side == "long" and rsi_fast > 50:
            signal = "exit"
        elif position and position.side == "short" and rsi_fast < 50:
            signal = "exit"

        return signal


class MeanRevStochRSIStrategy(Strategy):
    """Stochastic RSI mean reversion"""

    name = "MeanRevStochRSI"

    def __init__(
        self,
        rsi_period: int = 14,
        stoch_period: int = 14,
        k_period: int = 3,
        d_period: int = 3,
        oversold: int = 20,
        overbought: int = 80,
        *,
        allow_short: bool = True,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.02,
    ) -> None:
        self.rsi_period = rsi_period
        self.stoch_period = stoch_period
        self.k_period = k_period
        self.d_period = d_period
        self.oversold = oversold
        self.overbought = overbought
        self.allow_short = allow_short
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self._prices: list[float] = []
        self._rsi_history: list[float] = []

    def _calc_rsi(self, period: int, prices: list[float]) -> float:
        if len(prices) < period + 1:
            return 50.0

        price_arr = np.array(prices[-(period + 1):])
        deltas = np.diff(price_arr)

        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

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

        min_period = self.rsi_period + self.stoch_period + self.k_period + self.d_period
        if len(self._prices) < min_period:
            return "hold"

        # Calculate RSI
        rsi = self._calc_rsi(self.rsi_period, self._prices)
        self._rsi_history.append(rsi)

        if len(self._rsi_history) < self.stoch_period:
            return "hold"

        # Stochastic of RSI
        rsi_arr = np.array(self._rsi_history[-self.stoch_period:])
        rsi_min = np.min(rsi_arr)
        rsi_max = np.max(rsi_arr)

        if rsi_max - rsi_min == 0:
            stoch_rsi = 50.0
        else:
            stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min) * 100

        signal: Signal = "hold"

        # Entry signals
        if stoch_rsi < self.oversold:
            signal = "long"
        elif stoch_rsi > self.overbought:
            signal = "short" if self.allow_short else "hold"

        # Exit at neutral
        if position and position.side == "long" and stoch_rsi > 50:
            signal = "exit"
        elif position and position.side == "short" and stoch_rsi < 50:
            signal = "exit"

        return signal


# Factory function
def create_meanrev_strategy(
    strategy_type: str,
    params: dict,
    allow_short: bool = True,
    stop_loss_pct: float = 0.02,
    take_profit_pct: float = 0.02,
) -> Strategy:
    """Create mean reversion strategy from type and params"""

    if strategy_type == "bollinger":
        return MeanRevBollingerStrategy(
            bb_period=params.get("bb_period", 20),
            bb_std=params.get("bb_std", 2.0),
            rsi_period=params.get("rsi_period", 14),
            rsi_oversold=params.get("rsi_oversold", 30),
            rsi_overbought=params.get("rsi_overbought", 70),
            allow_short=allow_short,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
    elif strategy_type == "zscore":
        return MeanRevZScoreStrategy(
            lookback=params.get("lookback", 20),
            entry_zscore=params.get("entry_zscore", 2.0),
            exit_zscore=params.get("exit_zscore", 0.5),
            allow_short=allow_short,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
    elif strategy_type == "rsi":
        return MeanRevRSIStrategy(
            rsi_fast=params.get("rsi_fast", 7),
            rsi_slow=params.get("rsi_slow", 14),
            oversold=params.get("oversold", 30),
            overbought=params.get("overbought", 70),
            allow_short=allow_short,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
    elif strategy_type == "stoch_rsi":
        return MeanRevStochRSIStrategy(
            rsi_period=params.get("rsi_period", 14),
            stoch_period=params.get("stoch_period", 14),
            oversold=params.get("oversold", 20),
            overbought=params.get("overbought", 80),
            allow_short=allow_short,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
    else:
        raise ValueError(f"Unknown mean reversion strategy: {strategy_type}")


MEANREV_STRATEGIES = {
    "bollinger": {
        "class": MeanRevBollingerStrategy,
        "params": ["bb_period", "bb_std", "rsi_period", "rsi_oversold", "rsi_overbought"],
    },
    "zscore": {
        "class": MeanRevZScoreStrategy,
        "params": ["lookback", "entry_zscore", "exit_zscore"],
    },
    "rsi": {
        "class": MeanRevRSIStrategy,
        "params": ["rsi_fast", "rsi_slow", "oversold", "overbought"],
    },
    "stoch_rsi": {
        "class": MeanRevStochRSIStrategy,
        "params": ["rsi_period", "stoch_period", "oversold", "overbought"],
    },
}

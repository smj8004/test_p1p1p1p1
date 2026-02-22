"""
Volatility Regime Strategy Family

Based on:
- VIX-style volatility regime detection
- Volatility clustering (GARCH concept simplified)
- Risk-on/Risk-off regime switching
- Volatility targeting (Bridgewater/AQR style)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from .base import Bar, Signal, Strategy, StrategyPosition


class VolRegime(Enum):
    """Volatility regime states"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


class VolRegimeAdaptiveStrategy(Strategy):
    """Adaptive strategy based on volatility regime"""

    name = "VolRegimeAdaptive"

    def __init__(
        self,
        vol_short: int = 10,
        vol_long: int = 50,
        low_vol_mult: float = 0.8,
        high_vol_mult: float = 1.5,
        extreme_vol_mult: float = 2.5,
        ema_fast: int = 12,
        ema_slow: int = 26,
        *,
        allow_short: bool = True,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
    ) -> None:
        self.vol_short = vol_short
        self.vol_long = vol_long
        self.low_vol_mult = low_vol_mult
        self.high_vol_mult = high_vol_mult
        self.extreme_vol_mult = extreme_vol_mult
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.allow_short = allow_short
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self._prices: list[float] = []
        self._returns: list[float] = []
        self._current_regime = VolRegime.NORMAL

    def _calc_volatility(self, period: int) -> float:
        """Calculate rolling volatility"""
        if len(self._returns) < period:
            return 0.0
        return np.std(self._returns[-period:]) * np.sqrt(252 * 24)  # Annualized

    def _detect_regime(self) -> VolRegime:
        """Detect current volatility regime"""
        if len(self._returns) < self.vol_long:
            return VolRegime.NORMAL

        vol_short = self._calc_volatility(self.vol_short)
        vol_long = self._calc_volatility(self.vol_long)

        if vol_long == 0:
            return VolRegime.NORMAL

        vol_ratio = vol_short / vol_long

        if vol_ratio > self.extreme_vol_mult:
            return VolRegime.EXTREME
        elif vol_ratio > self.high_vol_mult:
            return VolRegime.HIGH
        elif vol_ratio < self.low_vol_mult:
            return VolRegime.LOW
        else:
            return VolRegime.NORMAL

    def _risk_exit(self, bar: Bar, position: StrategyPosition | None) -> bool:
        if position is None or position.side == "flat" or position.entry_price <= 0:
            return False

        # Adjust SL/TP based on regime
        sl_mult = 1.0
        tp_mult = 1.0

        if self._current_regime == VolRegime.HIGH:
            sl_mult = 1.5
            tp_mult = 1.5
        elif self._current_regime == VolRegime.EXTREME:
            sl_mult = 2.0
            tp_mult = 0.5  # Quick exit

        adjusted_sl = self.stop_loss_pct * sl_mult
        adjusted_tp = self.take_profit_pct * tp_mult

        if position.side == "long":
            if adjusted_sl > 0 and bar.close <= position.entry_price * (1 - adjusted_sl):
                return True
            if adjusted_tp > 0 and bar.close >= position.entry_price * (1 + adjusted_tp):
                return True
        elif position.side == "short":
            if adjusted_sl > 0 and bar.close >= position.entry_price * (1 + adjusted_sl):
                return True
            if adjusted_tp > 0 and bar.close <= position.entry_price * (1 - adjusted_tp):
                return True
        return False

    def on_bar(self, bar: Bar, position: StrategyPosition | None = None) -> Signal:
        self._prices.append(bar.close)

        if len(self._prices) > 1:
            ret = (bar.close - self._prices[-2]) / self._prices[-2]
            self._returns.append(ret)

        if self._risk_exit(bar, position):
            return "exit"

        min_period = max(self.vol_long, self.ema_slow)
        if len(self._prices) < min_period:
            return "hold"

        self._current_regime = self._detect_regime()

        # No new trades in extreme volatility
        if self._current_regime == VolRegime.EXTREME:
            if position and position.side != "flat":
                return "exit"  # Exit existing positions
            return "hold"

        # Calculate EMAs
        series = pd.Series(self._prices, dtype="float64")
        ema_fast = series.ewm(span=self.ema_fast, adjust=False).mean().iloc[-1]
        ema_slow = series.ewm(span=self.ema_slow, adjust=False).mean().iloc[-1]

        prev_ema_fast = series.ewm(span=self.ema_fast, adjust=False).mean().iloc[-2]
        prev_ema_slow = series.ewm(span=self.ema_slow, adjust=False).mean().iloc[-2]

        cross_up = prev_ema_fast <= prev_ema_slow and ema_fast > ema_slow
        cross_down = prev_ema_fast >= prev_ema_slow and ema_fast < ema_slow

        signal: Signal = "hold"

        # Regime-adjusted signals
        if self._current_regime == VolRegime.LOW:
            # Low vol: trend following more reliable
            if cross_up:
                signal = "long"
            elif cross_down:
                signal = "short" if self.allow_short else "exit"

        elif self._current_regime == VolRegime.NORMAL:
            # Normal: standard trend following
            if cross_up:
                signal = "long"
            elif cross_down:
                signal = "short" if self.allow_short else "exit"

        elif self._current_regime == VolRegime.HIGH:
            # High vol: mean reversion more reliable
            vol_short = self._calc_volatility(self.vol_short)
            zscore = (bar.close - ema_slow) / (vol_short * bar.close + 1e-10)

            if zscore < -2.0:
                signal = "long"
            elif zscore > 2.0:
                signal = "short" if self.allow_short else "hold"

        return signal


class VolRegimeVIXStrategy(Strategy):
    """VIX-like volatility index based trading"""

    name = "VolRegimeVIX"

    def __init__(
        self,
        vix_period: int = 21,
        vix_low: float = 15.0,
        vix_high: float = 25.0,
        ema_period: int = 20,
        *,
        allow_short: bool = True,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
    ) -> None:
        self.vix_period = vix_period
        self.vix_low = vix_low
        self.vix_high = vix_high
        self.ema_period = ema_period
        self.allow_short = allow_short
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self._prices: list[float] = []
        self._returns: list[float] = []

    def _calc_vix(self) -> float:
        """Calculate VIX-like volatility index"""
        if len(self._returns) < self.vix_period:
            return 20.0

        returns = np.array(self._returns[-self.vix_period:])
        vol = np.std(returns) * np.sqrt(252 * 24) * 100  # Annualized %
        return vol

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

        if len(self._prices) > 1:
            ret = (bar.close - self._prices[-2]) / self._prices[-2]
            self._returns.append(ret)

        if self._risk_exit(bar, position):
            return "exit"

        min_period = max(self.vix_period, self.ema_period)
        if len(self._prices) < min_period:
            return "hold"

        vix = self._calc_vix()
        ema = np.mean(self._prices[-self.ema_period:])

        signal: Signal = "hold"

        # Low VIX: Risk-on, follow trend
        if vix < self.vix_low:
            if bar.close > ema:
                signal = "long"
            elif bar.close < ema:
                signal = "short" if self.allow_short else "hold"

        # High VIX: Risk-off, mean reversion
        elif vix > self.vix_high:
            # Exit existing positions
            if position and position.side != "flat":
                return "exit"

        # Normal VIX: Neutral
        else:
            pass

        return signal


class VolTargetStrategy(Strategy):
    """Volatility targeting strategy (position sizing based on vol)"""

    name = "VolTarget"

    def __init__(
        self,
        target_vol: float = 0.15,
        vol_lookback: int = 20,
        ema_fast: int = 10,
        ema_slow: int = 30,
        min_weight: float = 0.1,
        max_weight: float = 2.0,
        *,
        allow_short: bool = True,
        stop_loss_pct: float = 0.03,
        take_profit_pct: float = 0.06,
    ) -> None:
        self.target_vol = target_vol
        self.vol_lookback = vol_lookback
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.allow_short = allow_short
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self._prices: list[float] = []
        self._returns: list[float] = []
        self._position_weight: float = 1.0

    def _calc_realized_vol(self) -> float:
        """Calculate realized volatility"""
        if len(self._returns) < self.vol_lookback:
            return self.target_vol

        returns = np.array(self._returns[-self.vol_lookback:])
        return np.std(returns) * np.sqrt(252 * 24)

    def _calc_position_weight(self) -> float:
        """Calculate position weight based on vol targeting"""
        realized_vol = self._calc_realized_vol()
        if realized_vol == 0:
            return 1.0

        weight = self.target_vol / realized_vol
        return np.clip(weight, self.min_weight, self.max_weight)

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

        if len(self._prices) > 1:
            ret = (bar.close - self._prices[-2]) / self._prices[-2]
            self._returns.append(ret)

        if self._risk_exit(bar, position):
            return "exit"

        min_period = max(self.vol_lookback, self.ema_slow)
        if len(self._prices) < min_period:
            return "hold"

        self._position_weight = self._calc_position_weight()

        # EMA signals
        series = pd.Series(self._prices, dtype="float64")
        ema_fast = series.ewm(span=self.ema_fast, adjust=False).mean().iloc[-1]
        ema_slow = series.ewm(span=self.ema_slow, adjust=False).mean().iloc[-1]

        signal: Signal = "hold"

        if ema_fast > ema_slow:
            signal = "long"
        elif ema_fast < ema_slow:
            signal = "short" if self.allow_short else "exit"

        return signal

    def get_position_weight(self) -> float:
        """Get current position weight for sizing"""
        return self._position_weight


class VolClusterStrategy(Strategy):
    """Volatility clustering based strategy"""

    name = "VolCluster"

    def __init__(
        self,
        vol_period: int = 14,
        cluster_threshold: float = 1.5,
        ema_period: int = 20,
        *,
        allow_short: bool = True,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
    ) -> None:
        self.vol_period = vol_period
        self.cluster_threshold = cluster_threshold
        self.ema_period = ema_period
        self.allow_short = allow_short
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self._prices: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._atr_history: list[float] = []
        self._in_cluster: bool = False

    def _calc_atr(self) -> float:
        if len(self._prices) < self.vol_period + 1:
            return 0.0

        highs = np.array(self._highs[-(self.vol_period + 1):])
        lows = np.array(self._lows[-(self.vol_period + 1):])
        closes = np.array(self._prices[-(self.vol_period + 1):])

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

        min_period = max(self.vol_period + 1, self.ema_period, 50)
        if len(self._prices) < min_period:
            return "hold"

        atr = self._calc_atr()
        self._atr_history.append(atr)

        if len(self._atr_history) < 20:
            return "hold"

        # Check for volatility clustering
        atr_ma = np.mean(self._atr_history[-20:])
        is_cluster = atr > atr_ma * self.cluster_threshold

        ema = np.mean(self._prices[-self.ema_period:])

        signal: Signal = "hold"

        # Volatility cluster: expect continuation
        if is_cluster and not self._in_cluster:
            # New cluster starting - trade the direction
            if bar.close > ema:
                signal = "long"
            else:
                signal = "short" if self.allow_short else "hold"

        # Cluster ending - exit
        if self._in_cluster and not is_cluster:
            if position and position.side != "flat":
                signal = "exit"

        self._in_cluster = is_cluster

        return signal


# Factory function
def create_volregime_strategy(
    strategy_type: str,
    params: dict,
    allow_short: bool = True,
    stop_loss_pct: float = 0.02,
    take_profit_pct: float = 0.04,
) -> Strategy:
    """Create volatility regime strategy from type and params"""

    if strategy_type == "adaptive":
        return VolRegimeAdaptiveStrategy(
            vol_short=params.get("vol_short", 10),
            vol_long=params.get("vol_long", 50),
            low_vol_mult=params.get("low_vol_mult", 0.8),
            high_vol_mult=params.get("high_vol_mult", 1.5),
            extreme_vol_mult=params.get("extreme_vol_mult", 2.5),
            ema_fast=params.get("ema_fast", 12),
            ema_slow=params.get("ema_slow", 26),
            allow_short=allow_short,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
    elif strategy_type == "vix":
        return VolRegimeVIXStrategy(
            vix_period=params.get("vix_period", 21),
            vix_low=params.get("vix_low", 15.0),
            vix_high=params.get("vix_high", 25.0),
            ema_period=params.get("ema_period", 20),
            allow_short=allow_short,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
    elif strategy_type == "target":
        return VolTargetStrategy(
            target_vol=params.get("target_vol", 0.15),
            vol_lookback=params.get("vol_lookback", 20),
            ema_fast=params.get("ema_fast", 10),
            ema_slow=params.get("ema_slow", 30),
            allow_short=allow_short,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
    elif strategy_type == "cluster":
        return VolClusterStrategy(
            vol_period=params.get("vol_period", 14),
            cluster_threshold=params.get("cluster_threshold", 1.5),
            ema_period=params.get("ema_period", 20),
            allow_short=allow_short,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
    else:
        raise ValueError(f"Unknown vol regime strategy: {strategy_type}")


VOLREGIME_STRATEGIES = {
    "adaptive": {
        "class": VolRegimeAdaptiveStrategy,
        "params": ["vol_short", "vol_long", "low_vol_mult", "high_vol_mult", "extreme_vol_mult"],
    },
    "vix": {
        "class": VolRegimeVIXStrategy,
        "params": ["vix_period", "vix_low", "vix_high", "ema_period"],
    },
    "target": {
        "class": VolTargetStrategy,
        "params": ["target_vol", "vol_lookback", "ema_fast", "ema_slow"],
    },
    "cluster": {
        "class": VolClusterStrategy,
        "params": ["vol_period", "cluster_threshold", "ema_period"],
    },
}

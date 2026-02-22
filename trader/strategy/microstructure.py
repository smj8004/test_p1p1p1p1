"""
Microstructure-lite Strategy Family

Based on (simplified for retail):
- Volume profile analysis
- Order flow imbalance
- VWAP deviations
- Volume-weighted momentum
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal

from .base import Bar, Signal, Strategy, StrategyPosition


class MicroVWAPStrategy(Strategy):
    """
    VWAP (Volume Weighted Average Price) based strategy

    Trade deviations from VWAP with volume confirmation
    """

    name = "MicroVWAP"

    def __init__(
        self,
        vwap_period: int = 24,
        deviation_threshold: float = 0.01,  # 1% deviation
        volume_mult: float = 1.2,
        *,
        allow_short: bool = True,
        stop_loss_pct: float = 0.015,
        take_profit_pct: float = 0.02,
    ) -> None:
        self.vwap_period = vwap_period
        self.deviation_threshold = deviation_threshold
        self.volume_mult = volume_mult
        self.allow_short = allow_short
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self._prices: list[float] = []
        self._volumes: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []

    def _calc_vwap(self) -> float:
        """Calculate VWAP"""
        if len(self._prices) < self.vwap_period:
            return self._prices[-1] if self._prices else 0.0

        prices = np.array(self._prices[-self.vwap_period:])
        volumes = np.array(self._volumes[-self.vwap_period:])

        typical_prices = prices  # Using close; could use (H+L+C)/3
        return np.sum(typical_prices * volumes) / np.sum(volumes) if np.sum(volumes) > 0 else prices[-1]

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
        self._volumes.append(bar.volume)
        self._highs.append(bar.high)
        self._lows.append(bar.low)

        if self._risk_exit(bar, position):
            return "exit"

        if len(self._prices) < self.vwap_period:
            return "hold"

        vwap = self._calc_vwap()
        deviation = (bar.close - vwap) / vwap

        # Volume filter
        vol_ma = np.mean(self._volumes[-20:]) if len(self._volumes) >= 20 else bar.volume
        vol_ratio = bar.volume / (vol_ma + 1e-10)

        signal: Signal = "hold"

        # Mean reversion to VWAP
        if deviation < -self.deviation_threshold and vol_ratio > self.volume_mult:
            signal = "long"
        elif deviation > self.deviation_threshold and vol_ratio > self.volume_mult:
            signal = "short" if self.allow_short else "hold"

        # Exit at VWAP
        if position and position.side == "long" and bar.close >= vwap:
            signal = "exit"
        elif position and position.side == "short" and bar.close <= vwap:
            signal = "exit"

        return signal


class MicroVolumeProfileStrategy(Strategy):
    """
    Volume profile based strategy

    Identifies high volume nodes (support/resistance)
    Trades bounces off these levels
    """

    name = "MicroVolumeProfile"

    def __init__(
        self,
        profile_period: int = 50,
        num_bins: int = 20,
        poc_proximity: float = 0.005,  # 0.5% proximity to POC
        *,
        allow_short: bool = True,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.03,
    ) -> None:
        self.profile_period = profile_period
        self.num_bins = num_bins
        self.poc_proximity = poc_proximity
        self.allow_short = allow_short
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self._prices: list[float] = []
        self._volumes: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []

    def _calc_volume_profile(self) -> tuple[float, float, float]:
        """
        Calculate volume profile and return POC, VAH, VAL

        POC = Point of Control (highest volume price)
        VAH = Value Area High
        VAL = Value Area Low
        """
        if len(self._prices) < self.profile_period:
            return self._prices[-1], self._prices[-1], self._prices[-1]

        prices = np.array(self._prices[-self.profile_period:])
        highs = np.array(self._highs[-self.profile_period:])
        lows = np.array(self._lows[-self.profile_period:])
        volumes = np.array(self._volumes[-self.profile_period:])

        price_min = np.min(lows)
        price_max = np.max(highs)

        if price_max == price_min:
            return prices[-1], prices[-1], prices[-1]

        bin_edges = np.linspace(price_min, price_max, self.num_bins + 1)
        bin_volumes = np.zeros(self.num_bins)

        for i, (p, v) in enumerate(zip(prices, volumes)):
            bin_idx = int((p - price_min) / (price_max - price_min) * (self.num_bins - 1))
            bin_idx = min(bin_idx, self.num_bins - 1)
            bin_volumes[bin_idx] += v

        # POC - bin with highest volume
        poc_idx = np.argmax(bin_volumes)
        poc = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2

        # Value area (70% of volume)
        total_vol = np.sum(bin_volumes)
        target_vol = total_vol * 0.7

        sorted_indices = np.argsort(bin_volumes)[::-1]
        cumulative_vol = 0
        value_indices = []

        for idx in sorted_indices:
            cumulative_vol += bin_volumes[idx]
            value_indices.append(idx)
            if cumulative_vol >= target_vol:
                break

        val = (bin_edges[min(value_indices)] + bin_edges[min(value_indices) + 1]) / 2
        vah = (bin_edges[max(value_indices)] + bin_edges[max(value_indices) + 1]) / 2

        return poc, vah, val

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
        self._volumes.append(bar.volume)
        self._highs.append(bar.high)
        self._lows.append(bar.low)

        if self._risk_exit(bar, position):
            return "exit"

        if len(self._prices) < self.profile_period:
            return "hold"

        poc, vah, val = self._calc_volume_profile()

        signal: Signal = "hold"

        # Near POC - potential support/resistance
        poc_dist = abs(bar.close - poc) / poc

        # Price near VAL (support) - potential long
        if bar.close < val * (1 + self.poc_proximity):
            signal = "long"

        # Price near VAH (resistance) - potential short
        elif bar.close > vah * (1 - self.poc_proximity):
            signal = "short" if self.allow_short else "hold"

        # Exit at POC
        if position and position.side == "long" and bar.close >= poc:
            signal = "exit"
        elif position and position.side == "short" and bar.close <= poc:
            signal = "exit"

        return signal


class MicroOrderFlowStrategy(Strategy):
    """
    Order flow imbalance strategy (simplified)

    Uses volume delta proxy (close position in bar)
    """

    name = "MicroOrderFlow"

    def __init__(
        self,
        flow_period: int = 10,
        imbalance_threshold: float = 0.6,  # 60% imbalance
        confirmation_period: int = 3,
        *,
        allow_short: bool = True,
        stop_loss_pct: float = 0.015,
        take_profit_pct: float = 0.025,
    ) -> None:
        self.flow_period = flow_period
        self.imbalance_threshold = imbalance_threshold
        self.confirmation_period = confirmation_period
        self.allow_short = allow_short
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self._prices: list[float] = []
        self._opens: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._volumes: list[float] = []
        self._deltas: list[float] = []

    def _calc_volume_delta(self, open_p: float, high: float, low: float, close: float, volume: float) -> float:
        """
        Estimate volume delta using bar position

        If close > open: more buying pressure
        If close < open: more selling pressure
        Scale by where close is in the range
        """
        if high == low:
            return 0.0

        # Close position in range [0, 1]
        position = (close - low) / (high - low)

        # Delta estimate: volume * (2 * position - 1)
        # At high: delta = +volume
        # At low: delta = -volume
        # At middle: delta = 0
        delta = volume * (2 * position - 1)
        return delta

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
        self._opens.append(bar.open)
        self._highs.append(bar.high)
        self._lows.append(bar.low)
        self._volumes.append(bar.volume)

        delta = self._calc_volume_delta(bar.open, bar.high, bar.low, bar.close, bar.volume)
        self._deltas.append(delta)

        if self._risk_exit(bar, position):
            return "exit"

        if len(self._deltas) < self.flow_period:
            return "hold"

        # Cumulative delta over period
        cum_delta = sum(self._deltas[-self.flow_period:])
        total_volume = sum(self._volumes[-self.flow_period:])

        if total_volume == 0:
            return "hold"

        # Imbalance ratio
        imbalance = cum_delta / total_volume

        # Confirmation: consecutive bars in same direction
        recent_deltas = self._deltas[-self.confirmation_period:]
        all_positive = all(d > 0 for d in recent_deltas)
        all_negative = all(d < 0 for d in recent_deltas)

        signal: Signal = "hold"

        # Strong buying imbalance
        if imbalance > self.imbalance_threshold and all_positive:
            signal = "long"

        # Strong selling imbalance
        elif imbalance < -self.imbalance_threshold and all_negative:
            signal = "short" if self.allow_short else "hold"

        # Exit when imbalance reverses
        if position and position.side == "long" and imbalance < 0:
            signal = "exit"
        elif position and position.side == "short" and imbalance > 0:
            signal = "exit"

        return signal


class MicroVolumeMomentumStrategy(Strategy):
    """
    Volume-weighted momentum strategy

    Weights price momentum by relative volume
    """

    name = "MicroVolumeMomentum"

    def __init__(
        self,
        momentum_period: int = 10,
        volume_lookback: int = 20,
        momentum_threshold: float = 0.005,
        volume_mult: float = 1.3,
        *,
        allow_short: bool = True,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
    ) -> None:
        self.momentum_period = momentum_period
        self.volume_lookback = volume_lookback
        self.momentum_threshold = momentum_threshold
        self.volume_mult = volume_mult
        self.allow_short = allow_short
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self._prices: list[float] = []
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
        self._volumes.append(bar.volume)

        if self._risk_exit(bar, position):
            return "exit"

        min_period = max(self.momentum_period, self.volume_lookback) + 1
        if len(self._prices) < min_period:
            return "hold"

        # Price momentum
        momentum = (bar.close - self._prices[-self.momentum_period - 1]) / self._prices[-self.momentum_period - 1]

        # Volume factor
        vol_ma = np.mean(self._volumes[-self.volume_lookback:])
        vol_ratio = bar.volume / (vol_ma + 1e-10)

        # Volume-weighted momentum
        vol_momentum = momentum * vol_ratio

        signal: Signal = "hold"

        # Strong volume-weighted momentum
        if vol_momentum > self.momentum_threshold and vol_ratio > self.volume_mult:
            signal = "long"
        elif vol_momentum < -self.momentum_threshold and vol_ratio > self.volume_mult:
            signal = "short" if self.allow_short else "hold"

        # Exit on momentum reversal
        if position and position.side == "long" and momentum < 0:
            signal = "exit"
        elif position and position.side == "short" and momentum > 0:
            signal = "exit"

        return signal


# Factory function
def create_microstructure_strategy(
    strategy_type: str,
    params: dict,
    allow_short: bool = True,
    stop_loss_pct: float = 0.02,
    take_profit_pct: float = 0.03,
) -> Strategy:
    """Create microstructure strategy from type and params"""

    if strategy_type == "vwap":
        return MicroVWAPStrategy(
            vwap_period=params.get("vwap_period", 24),
            deviation_threshold=params.get("deviation_threshold", 0.01),
            volume_mult=params.get("volume_mult", 1.2),
            allow_short=allow_short,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
    elif strategy_type == "volume_profile":
        return MicroVolumeProfileStrategy(
            profile_period=params.get("profile_period", 50),
            num_bins=params.get("num_bins", 20),
            poc_proximity=params.get("poc_proximity", 0.005),
            allow_short=allow_short,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
    elif strategy_type == "order_flow":
        return MicroOrderFlowStrategy(
            flow_period=params.get("flow_period", 10),
            imbalance_threshold=params.get("imbalance_threshold", 0.6),
            confirmation_period=params.get("confirmation_period", 3),
            allow_short=allow_short,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
    elif strategy_type == "volume_momentum":
        return MicroVolumeMomentumStrategy(
            momentum_period=params.get("momentum_period", 10),
            volume_lookback=params.get("volume_lookback", 20),
            momentum_threshold=params.get("momentum_threshold", 0.005),
            volume_mult=params.get("volume_mult", 1.3),
            allow_short=allow_short,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
    else:
        raise ValueError(f"Unknown microstructure strategy: {strategy_type}")


MICROSTRUCTURE_STRATEGIES = {
    "vwap": {
        "class": MicroVWAPStrategy,
        "params": ["vwap_period", "deviation_threshold", "volume_mult"],
    },
    "volume_profile": {
        "class": MicroVolumeProfileStrategy,
        "params": ["profile_period", "num_bins", "poc_proximity"],
    },
    "order_flow": {
        "class": MicroOrderFlowStrategy,
        "params": ["flow_period", "imbalance_threshold", "confirmation_period"],
    },
    "volume_momentum": {
        "class": MicroVolumeMomentumStrategy,
        "params": ["momentum_period", "volume_lookback", "momentum_threshold", "volume_mult"],
    },
}

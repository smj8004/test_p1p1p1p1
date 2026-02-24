"""
Multi-Timeframe (MTF) Futures Backtester

This is the most realistic backtesting approach:
- Uses 1m as base timeframe, calculates higher TFs in real-time
- Multiple timeframe confirmation for entries/exits
- Proper lookahead bias prevention (only uses CLOSED candles)
- Realistic execution at next bar's open with slippage
- Funding rate costs and liquidation simulation
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from trader.logger_utils import get_logger

logger = get_logger(__name__)


@dataclass
class MTFBar:
    """Multi-timeframe bar data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool = True  # False if bar is still forming


@dataclass
class MTFBars:
    """Container for all timeframe bars at a given moment."""
    m1: MTFBar          # 1 minute (always closed, base timeframe)
    m5: MTFBar          # 5 minutes
    m15: MTFBar         # 15 minutes
    h1: MTFBar          # 1 hour
    h4: MTFBar          # 4 hours

    # Closed bars only (for indicator calculation)
    m5_closed: MTFBar | None = None
    m15_closed: MTFBar | None = None
    h1_closed: MTFBar | None = None
    h4_closed: MTFBar | None = None


@dataclass
class MTFIndicators:
    """Pre-calculated indicators for all timeframes."""
    # 1m indicators
    m1_ema_9: float = 0.0
    m1_ema_21: float = 0.0
    m1_rsi_14: float = 50.0
    m1_volume_sma_20: float = 0.0

    # 5m indicators
    m5_ema_9: float = 0.0
    m5_ema_21: float = 0.0
    m5_rsi_14: float = 50.0
    m5_macd: float = 0.0
    m5_macd_signal: float = 0.0
    m5_macd_hist: float = 0.0

    # 15m indicators
    m15_ema_9: float = 0.0
    m15_ema_21: float = 0.0
    m15_ema_50: float = 0.0
    m15_rsi_14: float = 50.0
    m15_macd: float = 0.0
    m15_macd_signal: float = 0.0
    m15_bb_upper: float = 0.0
    m15_bb_middle: float = 0.0
    m15_bb_lower: float = 0.0

    # 1h indicators (Setup timeframe)
    h1_ema_9: float = 0.0
    h1_ema_21: float = 0.0
    h1_ema_50: float = 0.0
    h1_ema_200: float = 0.0
    h1_rsi_14: float = 50.0
    h1_macd: float = 0.0
    h1_macd_signal: float = 0.0
    h1_macd_hist: float = 0.0
    h1_adx: float = 0.0
    h1_atr: float = 0.0

    # 4h indicators (Trend timeframe)
    h4_ema_21: float = 0.0
    h4_ema_50: float = 0.0
    h4_ema_200: float = 0.0
    h4_rsi_14: float = 50.0
    h4_macd: float = 0.0
    h4_macd_signal: float = 0.0
    h4_adx: float = 0.0
    h4_trend: str = "neutral"  # "bullish", "bearish", "neutral"


class MTFStrategy(Protocol):
    """Protocol for MTF strategies."""
    name: str

    def on_bar(
        self,
        bars: MTFBars,
        indicators: MTFIndicators,
        position: str,  # "long", "short", "flat"
        entry_price: float | None,
    ) -> str:
        """
        Generate trading signal based on multiple timeframes.

        Returns: "long", "short", "exit", or "hold"
        """
        ...


@dataclass
class MTFBacktestConfig:
    """Configuration for MTF backtesting."""
    initial_capital: float = 10000.0
    leverage: int = 5
    position_size_pct: float = 0.95  # Use 95% of capital
    slippage_pct: float = 0.05  # 0.05% slippage
    maker_fee: float = 0.0002  # 0.02%
    taker_fee: float = 0.0005  # 0.05%

    # Risk management
    use_stop_loss: bool = True
    stop_loss_pct: float = 2.0  # 2% stop loss
    use_take_profit: bool = True
    take_profit_pct: float = 4.0  # 4% take profit (2:1 RR)
    use_trailing_stop: bool = False
    trailing_stop_pct: float = 1.5

    # Trade management
    min_holding_bars: int = 60  # Minimum 60 bars (1 hour on 1m) before signal exit
    cooldown_bars: int = 15  # Wait 15 bars after exit before new entry


@dataclass
class Trade:
    """Completed trade record."""
    entry_time: datetime
    exit_time: datetime
    direction: str  # "long" or "short"
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    exit_reason: str  # "signal", "stop_loss", "take_profit", "trailing_stop", "liquidation"
    funding_paid: float = 0.0
    fees_paid: float = 0.0


class MTFIndicatorCalculator:
    """
    Calculates indicators for multiple timeframes efficiently.
    Uses incremental calculation where possible.
    """

    def __init__(self):
        # Price history for each timeframe
        self.prices: dict[str, list[float]] = {
            "1m": [], "5m": [], "15m": [], "1h": [], "4h": []
        }
        self.highs: dict[str, list[float]] = {
            "1m": [], "5m": [], "15m": [], "1h": [], "4h": []
        }
        self.lows: dict[str, list[float]] = {
            "1m": [], "5m": [], "15m": [], "1h": [], "4h": []
        }
        self.volumes: dict[str, list[float]] = {
            "1m": [], "5m": [], "15m": [], "1h": [], "4h": []
        }

        # EMA states for incremental calculation
        self.ema_states: dict[str, dict[int, float]] = {}

        # RSI states
        self.rsi_states: dict[str, dict] = {}

        # MACD states
        self.macd_states: dict[str, dict] = {}

        # Max history to keep
        self.max_history = 500

    def update(self, tf: str, close: float, high: float, low: float, volume: float):
        """Update price history for a timeframe."""
        self.prices[tf].append(close)
        self.highs[tf].append(high)
        self.lows[tf].append(low)
        self.volumes[tf].append(volume)

        # Trim history
        if len(self.prices[tf]) > self.max_history:
            self.prices[tf] = self.prices[tf][-self.max_history:]
            self.highs[tf] = self.highs[tf][-self.max_history:]
            self.lows[tf] = self.lows[tf][-self.max_history:]
            self.volumes[tf] = self.volumes[tf][-self.max_history:]

    def ema(self, tf: str, period: int) -> float:
        """Calculate EMA incrementally."""
        prices = self.prices[tf]
        if len(prices) < period:
            return prices[-1] if prices else 0.0

        key = f"{tf}_{period}"
        if key not in self.ema_states:
            # Initialize with SMA
            self.ema_states[key] = sum(prices[:period]) / period

        # Incremental EMA
        multiplier = 2 / (period + 1)
        self.ema_states[key] = (prices[-1] - self.ema_states[key]) * multiplier + self.ema_states[key]

        return self.ema_states[key]

    def rsi(self, tf: str, period: int = 14) -> float:
        """Calculate RSI."""
        prices = self.prices[tf]
        if len(prices) < period + 1:
            return 50.0

        key = f"{tf}_{period}"

        if key not in self.rsi_states:
            # Initialize
            gains = []
            losses = []
            for i in range(1, period + 1):
                change = prices[i] - prices[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))

            avg_gain = sum(gains) / period
            avg_loss = sum(losses) / period
            self.rsi_states[key] = {"avg_gain": avg_gain, "avg_loss": avg_loss}

        # Incremental update
        change = prices[-1] - prices[-2]
        gain = max(change, 0)
        loss = abs(min(change, 0))

        state = self.rsi_states[key]
        state["avg_gain"] = (state["avg_gain"] * (period - 1) + gain) / period
        state["avg_loss"] = (state["avg_loss"] * (period - 1) + loss) / period

        if state["avg_loss"] == 0:
            return 100.0

        rs = state["avg_gain"] / state["avg_loss"]
        return 100 - (100 / (1 + rs))

    def macd(self, tf: str, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[float, float, float]:
        """Calculate MACD."""
        prices = self.prices[tf]
        if len(prices) < slow:
            return 0.0, 0.0, 0.0

        key = f"{tf}_macd"

        if key not in self.macd_states:
            # Initialize EMAs
            fast_ema = sum(prices[:fast]) / fast
            slow_ema = sum(prices[:slow]) / slow
            macd_line = fast_ema - slow_ema
            signal_ema = macd_line
            self.macd_states[key] = {
                "fast_ema": fast_ema,
                "slow_ema": slow_ema,
                "signal_ema": signal_ema
            }

        state = self.macd_states[key]

        # Update EMAs
        fast_mult = 2 / (fast + 1)
        slow_mult = 2 / (slow + 1)
        signal_mult = 2 / (signal + 1)

        state["fast_ema"] = (prices[-1] - state["fast_ema"]) * fast_mult + state["fast_ema"]
        state["slow_ema"] = (prices[-1] - state["slow_ema"]) * slow_mult + state["slow_ema"]

        macd_line = state["fast_ema"] - state["slow_ema"]
        state["signal_ema"] = (macd_line - state["signal_ema"]) * signal_mult + state["signal_ema"]

        histogram = macd_line - state["signal_ema"]

        return macd_line, state["signal_ema"], histogram

    def bollinger_bands(self, tf: str, period: int = 20, std_dev: float = 2.0) -> tuple[float, float, float]:
        """Calculate Bollinger Bands."""
        prices = self.prices[tf]
        if len(prices) < period:
            price = prices[-1] if prices else 0.0
            return price, price, price

        recent = prices[-period:]
        middle = sum(recent) / period
        std = (sum((p - middle) ** 2 for p in recent) / period) ** 0.5

        upper = middle + std_dev * std
        lower = middle - std_dev * std

        return upper, middle, lower

    def atr(self, tf: str, period: int = 14) -> float:
        """Calculate ATR."""
        if len(self.highs[tf]) < period + 1:
            return 0.0

        tr_values = []
        for i in range(-period, 0):
            high = self.highs[tf][i]
            low = self.lows[tf][i]
            prev_close = self.prices[tf][i-1]

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            tr_values.append(tr)

        return sum(tr_values) / period

    def adx(self, tf: str, period: int = 14) -> float:
        """Calculate ADX (simplified)."""
        if len(self.highs[tf]) < period + 1:
            return 0.0

        # Simplified ADX calculation
        plus_dm = []
        minus_dm = []
        tr_values = []

        for i in range(-period, 0):
            high = self.highs[tf][i]
            low = self.lows[tf][i]
            prev_high = self.highs[tf][i-1]
            prev_low = self.lows[tf][i-1]
            prev_close = self.prices[tf][i-1]

            up_move = high - prev_high
            down_move = prev_low - low

            plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0)
            minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0)

            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)

        atr = sum(tr_values) / period
        if atr == 0:
            return 0.0

        plus_di = 100 * sum(plus_dm) / period / atr
        minus_di = 100 * sum(minus_dm) / period / atr

        if plus_di + minus_di == 0:
            return 0.0

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)

        return dx  # Simplified: returning DX instead of smoothed ADX

    def volume_sma(self, tf: str, period: int = 20) -> float:
        """Calculate volume SMA."""
        volumes = self.volumes[tf]
        if len(volumes) < period:
            return volumes[-1] if volumes else 0.0

        return sum(volumes[-period:]) / period

    def get_indicators(self) -> MTFIndicators:
        """Get all indicators for current state."""
        ind = MTFIndicators()

        # 1m indicators
        if self.prices["1m"]:
            ind.m1_ema_9 = self.ema("1m", 9)
            ind.m1_ema_21 = self.ema("1m", 21)
            ind.m1_rsi_14 = self.rsi("1m", 14)
            ind.m1_volume_sma_20 = self.volume_sma("1m", 20)

        # 5m indicators
        if self.prices["5m"]:
            ind.m5_ema_9 = self.ema("5m", 9)
            ind.m5_ema_21 = self.ema("5m", 21)
            ind.m5_rsi_14 = self.rsi("5m", 14)
            ind.m5_macd, ind.m5_macd_signal, ind.m5_macd_hist = self.macd("5m")

        # 15m indicators
        if self.prices["15m"]:
            ind.m15_ema_9 = self.ema("15m", 9)
            ind.m15_ema_21 = self.ema("15m", 21)
            ind.m15_ema_50 = self.ema("15m", 50)
            ind.m15_rsi_14 = self.rsi("15m", 14)
            ind.m15_macd, ind.m15_macd_signal, _ = self.macd("15m")
            ind.m15_bb_upper, ind.m15_bb_middle, ind.m15_bb_lower = self.bollinger_bands("15m")

        # 1h indicators
        if self.prices["1h"]:
            ind.h1_ema_9 = self.ema("1h", 9)
            ind.h1_ema_21 = self.ema("1h", 21)
            ind.h1_ema_50 = self.ema("1h", 50)
            ind.h1_ema_200 = self.ema("1h", 200)
            ind.h1_rsi_14 = self.rsi("1h", 14)
            ind.h1_macd, ind.h1_macd_signal, ind.h1_macd_hist = self.macd("1h")
            ind.h1_adx = self.adx("1h")
            ind.h1_atr = self.atr("1h")

        # 4h indicators
        if self.prices["4h"]:
            ind.h4_ema_21 = self.ema("4h", 21)
            ind.h4_ema_50 = self.ema("4h", 50)
            ind.h4_ema_200 = self.ema("4h", 200)
            ind.h4_rsi_14 = self.rsi("4h", 14)
            ind.h4_macd, ind.h4_macd_signal, _ = self.macd("4h")
            ind.h4_adx = self.adx("4h")

            # Determine trend
            if self.prices["4h"]:
                price = self.prices["4h"][-1]
                if price > ind.h4_ema_50 and ind.h4_ema_21 > ind.h4_ema_50:
                    ind.h4_trend = "bullish"
                elif price < ind.h4_ema_50 and ind.h4_ema_21 < ind.h4_ema_50:
                    ind.h4_trend = "bearish"
                else:
                    ind.h4_trend = "neutral"

        return ind


class MTFBarBuilder:
    """
    Builds higher timeframe bars from 1-minute bars in real-time.
    Ensures no lookahead bias by only marking bars as closed at proper times.
    """

    def __init__(self):
        # Current forming bars
        self.current_bars: dict[str, dict] = {
            "5m": None, "15m": None, "1h": None, "4h": None
        }
        # Last closed bars
        self.last_closed: dict[str, MTFBar | None] = {
            "5m": None, "15m": None, "1h": None, "4h": None
        }

        # Timeframe minutes
        self.tf_minutes = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}

    def _get_tf_start(self, timestamp: datetime, minutes: int) -> datetime:
        """Get the start time of the timeframe period."""
        total_minutes = timestamp.hour * 60 + timestamp.minute
        period_start_minutes = (total_minutes // minutes) * minutes
        return timestamp.replace(
            hour=period_start_minutes // 60,
            minute=period_start_minutes % 60,
            second=0,
            microsecond=0
        )

    def update(self, m1_bar: MTFBar) -> MTFBars:
        """
        Update all timeframe bars with new 1m bar.
        Returns MTFBars with current state of all timeframes.
        """
        ts = m1_bar.timestamp

        for tf, minutes in self.tf_minutes.items():
            period_start = self._get_tf_start(ts, minutes)

            current = self.current_bars[tf]

            if current is None or current["start"] != period_start:
                # New period started - close previous bar
                if current is not None:
                    self.last_closed[tf] = MTFBar(
                        timestamp=current["start"],
                        open=current["open"],
                        high=current["high"],
                        low=current["low"],
                        close=current["close"],
                        volume=current["volume"],
                        is_closed=True
                    )

                # Start new bar
                self.current_bars[tf] = {
                    "start": period_start,
                    "open": m1_bar.open,
                    "high": m1_bar.high,
                    "low": m1_bar.low,
                    "close": m1_bar.close,
                    "volume": m1_bar.volume
                }
            else:
                # Update current bar
                current["high"] = max(current["high"], m1_bar.high)
                current["low"] = min(current["low"], m1_bar.low)
                current["close"] = m1_bar.close
                current["volume"] += m1_bar.volume

        # Build MTFBars
        def make_bar(tf: str) -> MTFBar:
            current = self.current_bars[tf]
            if current is None:
                return MTFBar(
                    timestamp=m1_bar.timestamp,
                    open=m1_bar.open,
                    high=m1_bar.high,
                    low=m1_bar.low,
                    close=m1_bar.close,
                    volume=m1_bar.volume,
                    is_closed=False
                )
            return MTFBar(
                timestamp=current["start"],
                open=current["open"],
                high=current["high"],
                low=current["low"],
                close=current["close"],
                volume=current["volume"],
                is_closed=False
            )

        return MTFBars(
            m1=m1_bar,
            m5=make_bar("5m"),
            m15=make_bar("15m"),
            h1=make_bar("1h"),
            h4=make_bar("4h"),
            m5_closed=self.last_closed["5m"],
            m15_closed=self.last_closed["15m"],
            h1_closed=self.last_closed["1h"],
            h4_closed=self.last_closed["4h"],
        )


# =============================================================================
# MTF Strategies
# =============================================================================

class TrendFollowMTF:
    """
    Triple-screen trend following strategy.

    Screen 1 (4h): Trend direction - only trade with the trend
    Screen 2 (1h): Setup - wait for pullback and momentum shift
    Screen 3 (15m): Entry - precise entry timing
    """

    name = "TrendFollow_MTF"

    def __init__(
        self,
        trend_adx_threshold: float = 20.0,
        pullback_rsi_low: float = 40.0,
        pullback_rsi_high: float = 60.0,
        entry_rsi_threshold: float = 50.0,
    ):
        self.trend_adx_threshold = trend_adx_threshold
        self.pullback_rsi_low = pullback_rsi_low
        self.pullback_rsi_high = pullback_rsi_high
        self.entry_rsi_threshold = entry_rsi_threshold

        # State
        self.setup_ready = False
        self.setup_direction: str | None = None

    def on_bar(
        self,
        bars: MTFBars,
        indicators: MTFIndicators,
        position: str,
        entry_price: float | None,
    ) -> str:
        # Screen 1: 4h Trend
        trend = indicators.h4_trend
        trend_strong = indicators.h4_adx > self.trend_adx_threshold

        # Screen 2: 1h Setup (pullback in trend)
        if trend == "bullish" and trend_strong:
            # Look for pullback (RSI dropped)
            if indicators.h1_rsi_14 < self.pullback_rsi_low:
                self.setup_ready = True
                self.setup_direction = "long"
        elif trend == "bearish" and trend_strong:
            if indicators.h1_rsi_14 > self.pullback_rsi_high:
                self.setup_ready = True
                self.setup_direction = "short"

        # Screen 3: 15m Entry
        if self.setup_ready and position == "flat":
            if self.setup_direction == "long":
                # Enter when 15m shows momentum resuming
                if (indicators.m15_rsi_14 > self.entry_rsi_threshold and
                    indicators.m15_macd > indicators.m15_macd_signal):
                    self.setup_ready = False
                    return "long"
            elif self.setup_direction == "short":
                if (indicators.m15_rsi_14 < self.entry_rsi_threshold and
                    indicators.m15_macd < indicators.m15_macd_signal):
                    self.setup_ready = False
                    return "short"

        # Exit: trend reversal or momentum loss
        if position == "long":
            if trend == "bearish" or indicators.h1_macd_hist < 0:
                return "exit"
        elif position == "short":
            if trend == "bullish" or indicators.h1_macd_hist > 0:
                return "exit"

        # Clear setup if trend changes
        if trend == "neutral" or not trend_strong:
            self.setup_ready = False

        return "hold"


class MomentumBreakoutMTF:
    """
    Multi-timeframe momentum breakout strategy.

    4h: Overall trend direction
    1h: Bollinger Band squeeze detection
    15m: Breakout confirmation with volume
    5m: Entry timing
    """

    name = "MomentumBreakout_MTF"

    def __init__(
        self,
        bb_squeeze_threshold: float = 0.02,  # 2% band width
        volume_multiplier: float = 1.5,
        rsi_confirmation: float = 55.0,
    ):
        self.bb_squeeze_threshold = bb_squeeze_threshold
        self.volume_multiplier = volume_multiplier
        self.rsi_confirmation = rsi_confirmation

        self.squeeze_detected = False
        self.breakout_direction: str | None = None

    def on_bar(
        self,
        bars: MTFBars,
        indicators: MTFIndicators,
        position: str,
        entry_price: float | None,
    ) -> str:
        # Check 1h trend alignment
        trend = indicators.h4_trend

        # 15m Bollinger Band analysis
        bb_width = (indicators.m15_bb_upper - indicators.m15_bb_lower) / indicators.m15_bb_middle if indicators.m15_bb_middle > 0 else 0

        # Detect squeeze
        if bb_width < self.bb_squeeze_threshold:
            self.squeeze_detected = True

        # 5m breakout with volume
        if self.squeeze_detected and position == "flat":
            current_volume = bars.m5.volume
            avg_volume = indicators.m1_volume_sma_20 * 5  # Approximate 5m volume
            volume_spike = current_volume > avg_volume * self.volume_multiplier if avg_volume > 0 else False

            if bars.m5.close > indicators.m15_bb_upper:
                # Bullish breakout
                if trend != "bearish" and volume_spike and indicators.m5_rsi_14 > self.rsi_confirmation:
                    self.squeeze_detected = False
                    return "long"
            elif bars.m5.close < indicators.m15_bb_lower:
                # Bearish breakout
                if trend != "bullish" and volume_spike and indicators.m5_rsi_14 < (100 - self.rsi_confirmation):
                    self.squeeze_detected = False
                    return "short"

        # Exit on band reversion
        if position == "long":
            if bars.m15.close < indicators.m15_bb_middle:
                return "exit"
        elif position == "short":
            if bars.m15.close > indicators.m15_bb_middle:
                return "exit"

        return "hold"


class MACDDivergenceMTF:
    """
    Multi-timeframe MACD divergence strategy.

    4h: Trend filter
    1h: Divergence detection
    15m: Entry confirmation
    """

    name = "MACDDivergence_MTF"

    def __init__(
        self,
        divergence_bars: int = 10,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
    ):
        self.divergence_bars = divergence_bars
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

        # Price and MACD history for divergence
        self.h1_price_lows: list[float] = []
        self.h1_macd_lows: list[float] = []
        self.h1_price_highs: list[float] = []
        self.h1_macd_highs: list[float] = []

        self.bullish_divergence = False
        self.bearish_divergence = False

    def on_bar(
        self,
        bars: MTFBars,
        indicators: MTFIndicators,
        position: str,
        entry_price: float | None,
    ) -> str:
        # Update history when 1h bar closes
        if bars.h1_closed:
            # Track for bullish divergence (lower lows in price, higher lows in MACD)
            self.h1_price_lows.append(bars.h1_closed.low)
            self.h1_macd_lows.append(indicators.h1_macd)

            # Track for bearish divergence
            self.h1_price_highs.append(bars.h1_closed.high)
            self.h1_macd_highs.append(indicators.h1_macd)

            # Keep limited history
            if len(self.h1_price_lows) > self.divergence_bars:
                self.h1_price_lows.pop(0)
                self.h1_macd_lows.pop(0)
                self.h1_price_highs.pop(0)
                self.h1_macd_highs.pop(0)

            # Detect divergences
            if len(self.h1_price_lows) >= 3:
                # Bullish: price making lower lows, MACD making higher lows
                if (self.h1_price_lows[-1] < self.h1_price_lows[-3] and
                    self.h1_macd_lows[-1] > self.h1_macd_lows[-3] and
                    indicators.h1_rsi_14 < self.rsi_oversold + 10):
                    self.bullish_divergence = True

                # Bearish: price making higher highs, MACD making lower highs
                if (self.h1_price_highs[-1] > self.h1_price_highs[-3] and
                    self.h1_macd_highs[-1] < self.h1_macd_highs[-3] and
                    indicators.h1_rsi_14 > self.rsi_overbought - 10):
                    self.bearish_divergence = True

        # Entry on 15m confirmation
        if position == "flat":
            if self.bullish_divergence:
                # Wait for 15m MACD cross up
                if indicators.m15_macd > indicators.m15_macd_signal:
                    self.bullish_divergence = False
                    return "long"

            if self.bearish_divergence:
                if indicators.m15_macd < indicators.m15_macd_signal:
                    self.bearish_divergence = False
                    return "short"

        # Exit on opposite signal or momentum loss
        if position == "long":
            if indicators.h1_macd < indicators.h1_macd_signal and indicators.h1_rsi_14 > 60:
                return "exit"
        elif position == "short":
            if indicators.h1_macd > indicators.h1_macd_signal and indicators.h1_rsi_14 < 40:
                return "exit"

        return "hold"


class RSIMeanReversionMTF:
    """
    Multi-timeframe RSI mean reversion strategy.

    4h: Trend identification (trade reversions within trend)
    1h: Extreme RSI detection
    15m/5m: Entry timing on RSI recovery
    """

    name = "RSIMeanReversion_MTF"

    def __init__(
        self,
        h1_rsi_oversold: float = 25.0,
        h1_rsi_overbought: float = 75.0,
        m15_rsi_recovery: float = 35.0,
        require_trend: bool = True,
    ):
        self.h1_rsi_oversold = h1_rsi_oversold
        self.h1_rsi_overbought = h1_rsi_overbought
        self.m15_rsi_recovery = m15_rsi_recovery
        self.require_trend = require_trend

        self.oversold_detected = False
        self.overbought_detected = False

    def on_bar(
        self,
        bars: MTFBars,
        indicators: MTFIndicators,
        position: str,
        entry_price: float | None,
    ) -> str:
        trend = indicators.h4_trend

        # Detect extreme RSI on 1h
        if indicators.h1_rsi_14 < self.h1_rsi_oversold:
            self.oversold_detected = True
        elif indicators.h1_rsi_14 > self.h1_rsi_overbought:
            self.overbought_detected = True

        # Entry on 15m recovery
        if position == "flat":
            if self.oversold_detected:
                # Only trade long in bullish/neutral trend
                if not self.require_trend or trend != "bearish":
                    if indicators.m15_rsi_14 > self.m15_rsi_recovery:
                        self.oversold_detected = False
                        return "long"

            if self.overbought_detected:
                if not self.require_trend or trend != "bullish":
                    if indicators.m15_rsi_14 < (100 - self.m15_rsi_recovery):
                        self.overbought_detected = False
                        return "short"

        # Exit at mean or opposite extreme
        if position == "long":
            if indicators.h1_rsi_14 > 55 or indicators.h1_rsi_14 > self.h1_rsi_overbought:
                return "exit"
        elif position == "short":
            if indicators.h1_rsi_14 < 45 or indicators.h1_rsi_14 < self.h1_rsi_oversold:
                return "exit"

        # Reset if RSI normalizes without entry
        if 40 < indicators.h1_rsi_14 < 60:
            self.oversold_detected = False
            self.overbought_detected = False

        return "hold"


class AdaptiveTrendMTF:
    """
    Adaptive trend strategy that adjusts based on market conditions.

    Uses ADX to determine trending vs ranging market.
    Trending: Follow trend with EMA crossovers
    Ranging: Mean reversion with Bollinger Bands
    """

    name = "AdaptiveTrend_MTF"

    def __init__(
        self,
        adx_trending_threshold: float = 25.0,
        adx_ranging_threshold: float = 20.0,
        ema_fast: int = 9,
        ema_slow: int = 21,
    ):
        self.adx_trending_threshold = adx_trending_threshold
        self.adx_ranging_threshold = adx_ranging_threshold
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow

        self.prev_h1_ema_fast = 0.0
        self.prev_h1_ema_slow = 0.0

    def on_bar(
        self,
        bars: MTFBars,
        indicators: MTFIndicators,
        position: str,
        entry_price: float | None,
    ) -> str:
        is_trending = indicators.h1_adx > self.adx_trending_threshold
        is_ranging = indicators.h1_adx < self.adx_ranging_threshold

        signal = "hold"

        if is_trending:
            # Trend following mode
            trend = indicators.h4_trend

            # EMA crossover on 1h, confirmed by 15m
            ema_cross_up = (self.prev_h1_ema_fast <= self.prev_h1_ema_slow and
                           indicators.h1_ema_9 > indicators.h1_ema_21)
            ema_cross_down = (self.prev_h1_ema_fast >= self.prev_h1_ema_slow and
                             indicators.h1_ema_9 < indicators.h1_ema_21)

            if position == "flat":
                if ema_cross_up and trend == "bullish":
                    if indicators.m15_rsi_14 > 50:  # 15m confirmation
                        signal = "long"
                elif ema_cross_down and trend == "bearish":
                    if indicators.m15_rsi_14 < 50:
                        signal = "short"
            elif position == "long":
                if ema_cross_down or trend == "bearish":
                    signal = "exit"
            elif position == "short":
                if ema_cross_up or trend == "bullish":
                    signal = "exit"

        elif is_ranging:
            # Mean reversion mode
            if position == "flat":
                if bars.h1.close < indicators.m15_bb_lower:
                    if indicators.m15_rsi_14 < 30:
                        signal = "long"
                elif bars.h1.close > indicators.m15_bb_upper:
                    if indicators.m15_rsi_14 > 70:
                        signal = "short"
            elif position == "long":
                if bars.h1.close > indicators.m15_bb_middle:
                    signal = "exit"
            elif position == "short":
                if bars.h1.close < indicators.m15_bb_middle:
                    signal = "exit"

        # Update state
        self.prev_h1_ema_fast = indicators.h1_ema_9
        self.prev_h1_ema_slow = indicators.h1_ema_21

        return signal


# =============================================================================
# MTF Backtester
# =============================================================================

class MTFBacktester:
    """
    Multi-Timeframe Backtester with realistic execution.
    """

    def __init__(
        self,
        config: MTFBacktestConfig,
        strategy: MTFStrategy,
        funding_rates: pd.DataFrame | None = None,
    ):
        self.config = config
        self.strategy = strategy
        self.funding_rates = funding_rates

        # State
        self.capital = config.initial_capital
        self.position: str = "flat"  # "long", "short", "flat"
        self.position_size: float = 0.0
        self.entry_price: float | None = None
        self.entry_time: datetime | None = None
        self.stop_loss_price: float | None = None
        self.take_profit_price: float | None = None
        self.trailing_stop_price: float | None = None
        self.highest_price: float = 0.0  # For trailing stop
        self.lowest_price: float = float("inf")

        # Tracking
        self.trades: list[Trade] = []
        self.equity_curve: list[dict] = []
        self.funding_paid: float = 0.0
        self.fees_paid: float = 0.0
        self.current_trade_funding: float = 0.0

        # Pending signal (for next-bar execution)
        self.pending_signal: str | None = None

        # Trade management
        self.bars_in_position: int = 0
        self.cooldown_remaining: int = 0

        # Components
        self.bar_builder = MTFBarBuilder()
        self.indicator_calc = MTFIndicatorCalculator()

    def _calculate_liquidation_price(self, entry_price: float, direction: str) -> float:
        """Calculate liquidation price based on leverage."""
        # Simplified: liquidation at ~(100/leverage)% loss
        liq_pct = 0.9 / self.config.leverage  # 90% of margin

        if direction == "long":
            return entry_price * (1 - liq_pct)
        else:
            return entry_price * (1 + liq_pct)

    def _apply_funding_rate(self, timestamp: datetime, mark_price: float) -> float:
        """Apply funding rate if at funding time."""
        if self.funding_rates is None or self.position == "flat":
            return 0.0

        # Funding times: 00:00, 08:00, 16:00 UTC
        hour = timestamp.hour
        minute = timestamp.minute

        if minute == 0 and hour in [0, 8, 16]:
            # Find funding rate
            mask = self.funding_rates["fundingTime"] <= timestamp
            if mask.any():
                rate = self.funding_rates.loc[mask, "fundingRate"].iloc[-1]

                # Calculate funding payment
                position_value = self.position_size * mark_price
                funding = position_value * rate

                # Long pays positive rate, short pays negative rate
                if self.position == "long":
                    return funding
                else:
                    return -funding

        return 0.0

    def _execute_entry(self, direction: str, price: float, timestamp: datetime):
        """Execute entry at given price."""
        # Apply slippage
        if direction == "long":
            exec_price = price * (1 + self.config.slippage_pct / 100)
        else:
            exec_price = price * (1 - self.config.slippage_pct / 100)

        # Calculate position size
        available_capital = self.capital * self.config.position_size_pct
        self.position_size = (available_capital * self.config.leverage) / exec_price

        # Deduct fees
        fee = available_capital * self.config.leverage * self.config.taker_fee
        self.capital -= fee
        self.fees_paid += fee

        # Set position
        self.position = direction
        self.entry_price = exec_price
        self.entry_time = timestamp
        self.current_trade_funding = 0.0

        # Set stop loss / take profit
        if self.config.use_stop_loss:
            if direction == "long":
                self.stop_loss_price = exec_price * (1 - self.config.stop_loss_pct / 100)
            else:
                self.stop_loss_price = exec_price * (1 + self.config.stop_loss_pct / 100)

        if self.config.use_take_profit:
            if direction == "long":
                self.take_profit_price = exec_price * (1 + self.config.take_profit_pct / 100)
            else:
                self.take_profit_price = exec_price * (1 - self.config.take_profit_pct / 100)

        # Initialize trailing stop tracking
        if self.config.use_trailing_stop:
            self.highest_price = exec_price
            self.lowest_price = exec_price

    def _execute_exit(self, price: float, timestamp: datetime, reason: str):
        """Execute exit at given price."""
        if self.position == "flat" or self.entry_price is None:
            return

        # Apply slippage
        if self.position == "long":
            exec_price = price * (1 - self.config.slippage_pct / 100)
        else:
            exec_price = price * (1 + self.config.slippage_pct / 100)

        # Calculate PnL
        if self.position == "long":
            pnl = (exec_price - self.entry_price) * self.position_size
        else:
            pnl = (self.entry_price - exec_price) * self.position_size

        # Deduct exit fees
        position_value = self.position_size * exec_price
        fee = position_value * self.config.taker_fee
        pnl -= fee
        self.fees_paid += fee

        # Deduct funding paid during trade
        pnl -= self.current_trade_funding

        # Calculate PnL percentage
        entry_value = self.position_size * self.entry_price / self.config.leverage
        pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0.0

        # Update capital
        self.capital += pnl

        # Record trade
        self.trades.append(Trade(
            entry_time=self.entry_time,
            exit_time=timestamp,
            direction=self.position,
            entry_price=self.entry_price,
            exit_price=exec_price,
            size=self.position_size,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            funding_paid=self.current_trade_funding,
            fees_paid=fee * 2,  # Entry + exit
        ))

        # Reset state
        self.position = "flat"
        self.position_size = 0.0
        self.entry_price = None
        self.entry_time = None
        self.stop_loss_price = None
        self.take_profit_price = None
        self.trailing_stop_price = None
        self.current_trade_funding = 0.0
        self.bars_in_position = 0
        self.cooldown_remaining = self.config.cooldown_bars

    def _check_stop_loss(self, bar: MTFBar) -> bool:
        """Check if stop loss hit."""
        if not self.config.use_stop_loss or self.stop_loss_price is None:
            return False

        if self.position == "long":
            return bar.low <= self.stop_loss_price
        else:
            return bar.high >= self.stop_loss_price

    def _check_take_profit(self, bar: MTFBar) -> bool:
        """Check if take profit hit."""
        if not self.config.use_take_profit or self.take_profit_price is None:
            return False

        if self.position == "long":
            return bar.high >= self.take_profit_price
        else:
            return bar.low <= self.take_profit_price

    def _update_trailing_stop(self, bar: MTFBar):
        """Update trailing stop price."""
        if not self.config.use_trailing_stop or self.position == "flat":
            return

        if self.position == "long":
            if bar.high > self.highest_price:
                self.highest_price = bar.high
                self.trailing_stop_price = self.highest_price * (1 - self.config.trailing_stop_pct / 100)
        else:
            if bar.low < self.lowest_price:
                self.lowest_price = bar.low
                self.trailing_stop_price = self.lowest_price * (1 + self.config.trailing_stop_pct / 100)

    def _check_trailing_stop(self, bar: MTFBar) -> bool:
        """Check if trailing stop hit."""
        if not self.config.use_trailing_stop or self.trailing_stop_price is None:
            return False

        if self.position == "long":
            return bar.low <= self.trailing_stop_price
        else:
            return bar.high >= self.trailing_stop_price

    def _check_liquidation(self, bar: MTFBar) -> bool:
        """Check if position liquidated."""
        if self.position == "flat" or self.entry_price is None:
            return False

        liq_price = self._calculate_liquidation_price(self.entry_price, self.position)

        if self.position == "long":
            return bar.low <= liq_price
        else:
            return bar.high >= liq_price

    def run(self, df: pd.DataFrame) -> dict:
        """
        Run backtest on 1-minute data.

        Args:
            df: DataFrame with 1m OHLCV data (columns: timestamp, open, high, low, close, volume)

        Returns:
            Backtest results dictionary
        """
        total_bars = len(df)
        logger.info(f"Running MTF backtest: {self.strategy.name}")
        logger.info(f"  Bars: {total_bars:,}")
        logger.info(f"  Leverage: {self.config.leverage}x")

        # Reset state
        self.capital = self.config.initial_capital
        self.position = "flat"
        self.trades = []
        self.equity_curve = []
        self.pending_signal = None
        self.bars_in_position = 0
        self.cooldown_remaining = 0

        # Progress tracking
        last_progress = 0

        for idx, row in df.iterrows():
            # Create 1m bar
            m1_bar = MTFBar(
                timestamp=row["timestamp"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                is_closed=True
            )

            # Build multi-timeframe bars
            mtf_bars = self.bar_builder.update(m1_bar)

            # Update indicators for closed bars
            self.indicator_calc.update("1m", m1_bar.close, m1_bar.high, m1_bar.low, m1_bar.volume)

            if mtf_bars.m5_closed:
                self.indicator_calc.update("5m", mtf_bars.m5_closed.close,
                                          mtf_bars.m5_closed.high, mtf_bars.m5_closed.low,
                                          mtf_bars.m5_closed.volume)
            if mtf_bars.m15_closed:
                self.indicator_calc.update("15m", mtf_bars.m15_closed.close,
                                          mtf_bars.m15_closed.high, mtf_bars.m15_closed.low,
                                          mtf_bars.m15_closed.volume)
            if mtf_bars.h1_closed:
                self.indicator_calc.update("1h", mtf_bars.h1_closed.close,
                                          mtf_bars.h1_closed.high, mtf_bars.h1_closed.low,
                                          mtf_bars.h1_closed.volume)
            if mtf_bars.h4_closed:
                self.indicator_calc.update("4h", mtf_bars.h4_closed.close,
                                          mtf_bars.h4_closed.high, mtf_bars.h4_closed.low,
                                          mtf_bars.h4_closed.volume)

            indicators = self.indicator_calc.get_indicators()

            # STEP 1: Execute pending signal at THIS bar's OPEN
            if self.pending_signal is not None:
                if self.pending_signal == "exit":
                    self._execute_exit(m1_bar.open, m1_bar.timestamp, "signal")
                elif self.pending_signal in ["long", "short"]:
                    if self.position != "flat":
                        self._execute_exit(m1_bar.open, m1_bar.timestamp, "signal")
                    self._execute_entry(self.pending_signal, m1_bar.open, m1_bar.timestamp)

                self.pending_signal = None

            # STEP 2: Check risk management (using bar's high/low)
            if self.position != "flat":
                # Check liquidation
                if self._check_liquidation(m1_bar):
                    liq_price = self._calculate_liquidation_price(self.entry_price, self.position)
                    self._execute_exit(liq_price, m1_bar.timestamp, "liquidation")
                # Check stop loss
                elif self._check_stop_loss(m1_bar):
                    self._execute_exit(self.stop_loss_price, m1_bar.timestamp, "stop_loss")
                # Check take profit
                elif self._check_take_profit(m1_bar):
                    self._execute_exit(self.take_profit_price, m1_bar.timestamp, "take_profit")
                else:
                    # Update and check trailing stop
                    self._update_trailing_stop(m1_bar)
                    if self._check_trailing_stop(m1_bar):
                        self._execute_exit(self.trailing_stop_price, m1_bar.timestamp, "trailing_stop")

            # STEP 3: Apply funding rate
            if self.position != "flat":
                funding = self._apply_funding_rate(m1_bar.timestamp, m1_bar.close)
                if funding != 0:
                    self.funding_paid += funding
                    self.current_trade_funding += funding

            # Update trade management counters
            if self.position != "flat":
                self.bars_in_position += 1
            if self.cooldown_remaining > 0:
                self.cooldown_remaining -= 1

            # STEP 4: Generate signal at bar CLOSE (for next bar)
            signal = self.strategy.on_bar(
                mtf_bars,
                indicators,
                self.position,
                self.entry_price
            )

            # Apply trade management rules
            if signal in ["long", "short", "exit"]:
                # For exits: check minimum holding period
                if signal == "exit":
                    if self.bars_in_position >= self.config.min_holding_bars:
                        self.pending_signal = signal
                # For entries: check cooldown
                elif signal in ["long", "short"]:
                    if self.cooldown_remaining == 0:
                        self.pending_signal = signal

            # Track equity
            unrealized_pnl = 0.0
            if self.position != "flat" and self.entry_price:
                if self.position == "long":
                    unrealized_pnl = (m1_bar.close - self.entry_price) * self.position_size
                else:
                    unrealized_pnl = (self.entry_price - m1_bar.close) * self.position_size

            # Record equity periodically (every hour)
            if m1_bar.timestamp.minute == 0:
                self.equity_curve.append({
                    "timestamp": m1_bar.timestamp,
                    "equity": self.capital + unrealized_pnl,
                    "capital": self.capital,
                    "position": self.position,
                })

            # Progress
            progress = int((idx + 1) / total_bars * 100)
            if progress >= last_progress + 10:
                logger.info(f"  Progress: {progress}% ({idx + 1:,}/{total_bars:,} bars)")
                last_progress = progress

        # Close any open position at end
        if self.position != "flat":
            last_row = df.iloc[-1]
            self._execute_exit(last_row["close"], last_row["timestamp"], "end_of_data")

        # Calculate results
        return self._calculate_results()

    def _calculate_results(self) -> dict:
        """Calculate backtest results."""
        if not self.trades:
            return {
                "strategy": self.strategy.name,
                "total_trades": 0,
                "final_capital": self.capital,
                "total_return_pct": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0,
            }

        # Basic stats
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]

        win_rate = len(winning_trades) / total_trades * 100

        total_profit = sum(t.pnl for t in winning_trades)
        total_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        avg_win = total_profit / len(winning_trades) if winning_trades else 0
        avg_loss = total_loss / len(losing_trades) if losing_trades else 0

        # Return
        total_return = self.capital - self.config.initial_capital
        total_return_pct = (total_return / self.config.initial_capital) * 100

        # Max drawdown
        max_equity = self.config.initial_capital
        max_drawdown = 0.0

        for eq in self.equity_curve:
            if eq["equity"] > max_equity:
                max_equity = eq["equity"]
            drawdown = (max_equity - eq["equity"]) / max_equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        max_drawdown_pct = max_drawdown * 100

        # Sharpe ratio (simplified)
        if len(self.equity_curve) > 1:
            returns = []
            for i in range(1, len(self.equity_curve)):
                ret = (self.equity_curve[i]["equity"] - self.equity_curve[i-1]["equity"]) / self.equity_curve[i-1]["equity"]
                returns.append(ret)

            if returns:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe = (avg_return / std_return) * np.sqrt(365 * 24) if std_return > 0 else 0
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        # Exit reason breakdown
        exit_reasons = {}
        for t in self.trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        # Long vs short performance
        long_trades = [t for t in self.trades if t.direction == "long"]
        short_trades = [t for t in self.trades if t.direction == "short"]

        long_pnl = sum(t.pnl for t in long_trades)
        short_pnl = sum(t.pnl for t in short_trades)

        return {
            "strategy": self.strategy.name,
            "leverage": self.config.leverage,
            "initial_capital": self.config.initial_capital,
            "final_capital": self.capital,
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_drawdown_pct": max_drawdown_pct,
            "sharpe_ratio": sharpe,
            "total_funding_paid": self.funding_paid,
            "total_fees_paid": self.fees_paid,
            "long_trades": len(long_trades),
            "short_trades": len(short_trades),
            "long_pnl": long_pnl,
            "short_pnl": short_pnl,
            "exit_reasons": exit_reasons,
        }


def run_mtf_backtest(
    symbol: str = "BTCUSDT",
    days: int = 365,
    leverages: list[int] = [1, 3, 5, 10],
    data_dir: str = "data/futures",
    output_dir: str = "data/futures/mtf_results",
) -> pd.DataFrame:
    """
    Run MTF backtest with all strategies.
    """
    from trader.logger_utils import setup_logging
    setup_logging(level="INFO")

    # Load data - check both path formats
    data_path = Path(data_dir) / "clean" / symbol / "ohlcv_1m.parquet"
    if not data_path.exists():
        # Try alternative path format
        data_path = Path(data_dir) / "clean" / f"{symbol}_1m.parquet"
    if not data_path.exists():
        logger.error(f"Data not found: {data_path}")
        logger.error("Please run: python main.py download-futures first")
        return pd.DataFrame()

    logger.info(f"Loading data: {data_path}")
    df = pd.read_parquet(data_path)

    # Normalize column names
    if "open_time" in df.columns and "timestamp" not in df.columns:
        df = df.rename(columns={"open_time": "timestamp"})

    # Filter to requested days
    if days < 1095:  # Less than 3 years
        cutoff = df["timestamp"].max() - pd.Timedelta(days=days)
        df = df[df["timestamp"] >= cutoff].reset_index(drop=True)

    logger.info(f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"Total bars: {len(df):,}")

    # Load funding rates - check both path formats
    funding_path = Path(data_dir) / "clean" / symbol / "funding_rate.parquet"
    if not funding_path.exists():
        funding_path = Path(data_dir) / "clean" / f"{symbol}_funding_rate.parquet"
    funding_df = None
    if funding_path.exists():
        funding_df = pd.read_parquet(funding_path)
        logger.info(f"Loaded funding rates: {len(funding_df):,} records")

    # Define strategies - focus on fewer, better tuned strategies
    strategies = [
        # Conservative trend following (fewer trades, higher quality)
        TrendFollowMTF(trend_adx_threshold=25.0, pullback_rsi_low=35.0, pullback_rsi_high=65.0),
        TrendFollowMTF(trend_adx_threshold=30.0, pullback_rsi_low=30.0, pullback_rsi_high=70.0),
        # Breakout with strict volume confirmation
        MomentumBreakoutMTF(bb_squeeze_threshold=0.015, volume_multiplier=2.0),
        # Divergence with extreme RSI only
        MACDDivergenceMTF(divergence_bars=15, rsi_oversold=25.0, rsi_overbought=75.0),
        # Mean reversion at extremes only
        RSIMeanReversionMTF(h1_rsi_oversold=20.0, h1_rsi_overbought=80.0, require_trend=True),
    ]

    # Run backtests
    results = []
    total_tests = len(strategies) * len(leverages)
    current_test = 0

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"Running {total_tests} MTF backtests...")
    logger.info("=" * 70)

    for strategy in strategies:
        for leverage in leverages:
            current_test += 1
            logger.info("")
            logger.info(f"[{current_test}/{total_tests}] {strategy.name} @ {leverage}x leverage")
            logger.info("-" * 50)

            # Adjust SL/TP based on leverage (higher leverage = tighter stops)
            base_sl = 3.0  # 3% base stop loss
            base_tp = 6.0  # 6% base take profit (2:1 RR)

            config = MTFBacktestConfig(
                leverage=leverage,
                use_stop_loss=True,
                stop_loss_pct=base_sl,
                use_take_profit=True,
                take_profit_pct=base_tp,
                use_trailing_stop=True,
                trailing_stop_pct=2.0,
            )

            # Create fresh strategy instance for each test
            strategy_copy = copy.deepcopy(strategy)

            backtester = MTFBacktester(
                config=config,
                strategy=strategy_copy,
                funding_rates=funding_df,
            )

            result = backtester.run(df.copy())
            results.append(result)

            logger.info(f"  Return: {result['total_return_pct']:.2f}%")
            logger.info(f"  Win Rate: {result['win_rate']:.1f}%")
            logger.info(f"  Max DD: {result['max_drawdown_pct']:.1f}%")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Sort by return
    results_df = results_df.sort_values("total_return_pct", ascending=False)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"mtf_results_{symbol}_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)

    # Print summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("MTF BACKTEST RESULTS - TOP 10")
    logger.info("=" * 70)

    top_10 = results_df.head(10)
    for idx, row in top_10.iterrows():
        logger.info(f"\n{row['strategy']} @ {row['leverage']}x")
        logger.info(f"  Return: {row['total_return_pct']:>8.2f}%")
        logger.info(f"  Win Rate: {row['win_rate']:>6.1f}%")
        logger.info(f"  Trades: {row['total_trades']:>5}")
        logger.info(f"  Max DD: {row['max_drawdown_pct']:>6.1f}%")
        logger.info(f"  Sharpe: {row['sharpe_ratio']:>6.2f}")
        logger.info(f"  PF: {row['profit_factor']:>6.2f}")

    logger.info("")
    logger.info(f"Results saved to: {results_file}")

    return results_df


if __name__ == "__main__":
    run_mtf_backtest(days=365)

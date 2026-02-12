from __future__ import annotations

from typing import Literal

import pandas as pd

from .base import Bar, Signal, Strategy, StrategyPosition

BollingerMode = Literal["mean_reversion", "breakout"]


class BollingerBandStrategy(Strategy):
    """
    Bollinger Band based trading strategy.

    Supports two modes:
    - Mean Reversion: Buy at lower band, sell at upper band (default)
    - Breakout: Buy when price breaks above upper band, sell when breaks below lower band
    """

    def __init__(
        self,
        period: int = 20,
        std_dev: float = 2.0,
        *,
        mode: BollingerMode = "mean_reversion",
        allow_short: bool = True,
        stop_loss_pct: float = 0.0,
        take_profit_pct: float = 0.0,
    ) -> None:
        """
        Initialize Bollinger Band strategy.

        Args:
            period: Moving average period (default: 20)
            std_dev: Standard deviation multiplier (default: 2.0)
            mode: Trading mode - "mean_reversion" or "breakout" (default: mean_reversion)
            allow_short: Whether to allow short positions (default: True)
            stop_loss_pct: Stop loss percentage (default: 0, disabled)
            take_profit_pct: Take profit percentage (default: 0, disabled)
        """
        if period < 2:
            raise ValueError("period must be at least 2")
        if std_dev <= 0:
            raise ValueError("std_dev must be positive")
        if mode not in ("mean_reversion", "breakout"):
            raise ValueError("mode must be 'mean_reversion' or 'breakout'")
        if stop_loss_pct < 0 or take_profit_pct < 0:
            raise ValueError("stop_loss_pct and take_profit_pct must be non-negative")

        self.period = period
        self.std_dev = std_dev
        self.mode = mode
        self.allow_short = allow_short
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self._closes: list[float] = []
        self._prev_close: float | None = None
        self._prev_upper: float | None = None
        self._prev_lower: float | None = None

    def _calculate_bands(self) -> tuple[float | None, float | None, float | None]:
        """
        Calculate Bollinger Bands.

        Returns:
            Tuple of (middle, upper, lower) bands or (None, None, None) if not enough data
        """
        if len(self._closes) < self.period:
            return None, None, None

        series = pd.Series(self._closes[-self.period:], dtype="float64")

        middle = series.mean()
        std = series.std()

        upper = middle + (self.std_dev * std)
        lower = middle - (self.std_dev * std)

        return float(middle), float(upper), float(lower)

    def _risk_exit(self, bar: Bar, position: StrategyPosition | None) -> bool:
        """Check if position should be exited due to risk limits."""
        if position is None or position.side == "flat" or position.entry_price <= 0:
            return False

        if position.side == "long":
            if self.stop_loss_pct > 0 and bar.close <= position.entry_price * (1 - self.stop_loss_pct):
                return True
            if self.take_profit_pct > 0 and bar.close >= position.entry_price * (1 + self.take_profit_pct):
                return True
        if position.side == "short":
            if self.stop_loss_pct > 0 and bar.close >= position.entry_price * (1 + self.stop_loss_pct):
                return True
            if self.take_profit_pct > 0 and bar.close <= position.entry_price * (1 - self.take_profit_pct):
                return True
        return False

    def _mean_reversion_signal(
        self,
        close: float,
        upper: float,
        lower: float,
        middle: float,
        position: StrategyPosition | None,
    ) -> Signal:
        """Generate signal for mean reversion mode."""
        if self._prev_close is None or self._prev_upper is None or self._prev_lower is None:
            return "hold"

        crossed_below_lower = self._prev_close >= self._prev_lower and close < lower
        crossed_above_upper = self._prev_close <= self._prev_upper and close > upper

        if crossed_below_lower:
            return "long"

        if crossed_above_upper:
            return "short" if self.allow_short else "exit"

        if position is not None:
            if position.side == "long" and close >= middle:
                return "exit"
            if position.side == "short" and close <= middle:
                return "exit"

        return "hold"

    def _breakout_signal(
        self,
        close: float,
        upper: float,
        lower: float,
    ) -> Signal:
        """Generate signal for breakout mode."""
        if self._prev_close is None or self._prev_upper is None or self._prev_lower is None:
            return "hold"

        broke_above_upper = self._prev_close <= self._prev_upper and close > upper
        broke_below_lower = self._prev_close >= self._prev_lower and close < lower

        if broke_above_upper:
            return "long"

        if broke_below_lower:
            return "short" if self.allow_short else "exit"

        return "hold"

    def on_bar(self, bar: Bar, position: StrategyPosition | None = None) -> Signal:
        """Process a bar and generate trading signal."""
        self._closes.append(bar.close)

        if self._risk_exit(bar, position):
            return "exit"

        middle, upper, lower = self._calculate_bands()
        if middle is None or upper is None or lower is None:
            return "hold"

        if self.mode == "mean_reversion":
            signal = self._mean_reversion_signal(bar.close, upper, lower, middle, position)
        else:
            signal = self._breakout_signal(bar.close, upper, lower)

        self._prev_close = bar.close
        self._prev_upper = upper
        self._prev_lower = lower
        return signal

    def get_state(self) -> dict:
        """Return current strategy state for persistence."""
        return {
            "closes_count": len(self._closes),
            "prev_close": self._prev_close,
            "prev_upper": self._prev_upper,
            "prev_lower": self._prev_lower,
            "mode": self.mode,
        }

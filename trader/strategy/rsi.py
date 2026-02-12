from __future__ import annotations

import pandas as pd

from .base import Bar, Signal, Strategy, StrategyPosition


class RSIStrategy(Strategy):
    """
    RSI (Relative Strength Index) based trading strategy.

    Generates signals based on RSI overbought/oversold levels:
    - Long: When RSI crosses below oversold level (potential reversal up)
    - Short/Exit: When RSI crosses above overbought level (potential reversal down)
    """

    def __init__(
        self,
        period: int = 14,
        overbought: float = 70.0,
        oversold: float = 30.0,
        *,
        allow_short: bool = True,
        stop_loss_pct: float = 0.0,
        take_profit_pct: float = 0.0,
    ) -> None:
        """
        Initialize RSI strategy.

        Args:
            period: RSI calculation period (default: 14)
            overbought: Overbought threshold (default: 70)
            oversold: Oversold threshold (default: 30)
            allow_short: Whether to allow short positions (default: True)
            stop_loss_pct: Stop loss percentage (default: 0, disabled)
            take_profit_pct: Take profit percentage (default: 0, disabled)
        """
        if period < 2:
            raise ValueError("period must be at least 2")
        if not (0 < oversold < overbought < 100):
            raise ValueError("oversold must be < overbought and both in (0, 100)")
        if stop_loss_pct < 0 or take_profit_pct < 0:
            raise ValueError("stop_loss_pct and take_profit_pct must be non-negative")

        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.allow_short = allow_short
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self._closes: list[float] = []
        self._prev_rsi: float | None = None

    def _calculate_rsi(self) -> float | None:
        """Calculate RSI from price history."""
        if len(self._closes) < self.period + 1:
            return None

        series = pd.Series(self._closes, dtype="float64")
        delta = series.diff()

        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(span=self.period, adjust=False).mean().iloc[-1]
        avg_loss = loss.ewm(span=self.period, adjust=False).mean().iloc[-1]

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

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

    def on_bar(self, bar: Bar, position: StrategyPosition | None = None) -> Signal:
        """Process a bar and generate trading signal."""
        self._closes.append(bar.close)

        if self._risk_exit(bar, position):
            return "exit"

        rsi = self._calculate_rsi()
        if rsi is None:
            return "hold"

        signal: Signal = "hold"

        if self._prev_rsi is not None:
            if self._prev_rsi >= self.oversold and rsi < self.oversold:
                signal = "long"
            elif self._prev_rsi <= self.overbought and rsi > self.overbought:
                signal = "short" if self.allow_short else "exit"

        self._prev_rsi = rsi
        return signal

    def get_state(self) -> dict:
        """Return current strategy state for persistence."""
        return {
            "closes_count": len(self._closes),
            "prev_rsi": self._prev_rsi,
        }

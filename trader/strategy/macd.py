from __future__ import annotations

import pandas as pd

from .base import Bar, Signal, Strategy, StrategyPosition


class MACDStrategy(Strategy):
    """
    MACD (Moving Average Convergence Divergence) based trading strategy.

    Generates signals based on MACD line crossing signal line:
    - Long: When MACD crosses above signal line (bullish crossover)
    - Short/Exit: When MACD crosses below signal line (bearish crossover)

    Optionally uses histogram for entry confirmation.
    """

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        *,
        allow_short: bool = True,
        use_histogram: bool = False,
        histogram_threshold: float = 0.0,
        stop_loss_pct: float = 0.0,
        take_profit_pct: float = 0.0,
    ) -> None:
        """
        Initialize MACD strategy.

        Args:
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line EMA period (default: 9)
            allow_short: Whether to allow short positions (default: True)
            use_histogram: Use histogram for entry confirmation (default: False)
            histogram_threshold: Minimum histogram value for entry (default: 0)
            stop_loss_pct: Stop loss percentage (default: 0, disabled)
            take_profit_pct: Take profit percentage (default: 0, disabled)
        """
        if fast_period < 1 or slow_period < 1 or signal_period < 1:
            raise ValueError("All periods must be at least 1")
        if fast_period >= slow_period:
            raise ValueError("fast_period must be less than slow_period")
        if stop_loss_pct < 0 or take_profit_pct < 0:
            raise ValueError("stop_loss_pct and take_profit_pct must be non-negative")

        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.allow_short = allow_short
        self.use_histogram = use_histogram
        self.histogram_threshold = histogram_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self._closes: list[float] = []
        self._prev_macd: float | None = None
        self._prev_signal: float | None = None
        self._prev_histogram: float | None = None

    def _calculate_macd(self) -> tuple[float | None, float | None, float | None]:
        """
        Calculate MACD, signal line, and histogram.

        Returns:
            Tuple of (macd, signal, histogram) or (None, None, None) if not enough data
        """
        min_required = self.slow_period + self.signal_period
        if len(self._closes) < min_required:
            return None, None, None

        series = pd.Series(self._closes, dtype="float64")

        fast_ema = series.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = series.ewm(span=self.slow_period, adjust=False).mean()

        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(histogram.iloc[-1])

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

        macd, signal_line, histogram = self._calculate_macd()
        if macd is None or signal_line is None:
            return "hold"

        signal: Signal = "hold"

        if self._prev_macd is not None and self._prev_signal is not None:
            macd_crossed_above = self._prev_macd <= self._prev_signal and macd > signal_line
            macd_crossed_below = self._prev_macd >= self._prev_signal and macd < signal_line

            if macd_crossed_above:
                if self.use_histogram and histogram is not None:
                    if histogram > self.histogram_threshold:
                        signal = "long"
                else:
                    signal = "long"
            elif macd_crossed_below:
                if self.use_histogram and histogram is not None:
                    if histogram < -self.histogram_threshold:
                        signal = "short" if self.allow_short else "exit"
                else:
                    signal = "short" if self.allow_short else "exit"

        self._prev_macd = macd
        self._prev_signal = signal_line
        self._prev_histogram = histogram
        return signal

    def get_state(self) -> dict:
        """Return current strategy state for persistence."""
        return {
            "closes_count": len(self._closes),
            "prev_macd": self._prev_macd,
            "prev_signal": self._prev_signal,
            "prev_histogram": self._prev_histogram,
        }

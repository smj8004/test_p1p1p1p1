from __future__ import annotations

import pandas as pd

from .base import Bar, Signal, Strategy, StrategyPosition


class EMACrossStrategy(Strategy):
    def __init__(
        self,
        short_window: int = 12,
        long_window: int = 26,
        *,
        allow_short: bool = True,
        stop_loss_pct: float = 0.0,
        take_profit_pct: float = 0.0,
    ) -> None:
        if long_window <= short_window:
            raise ValueError("long_window must be greater than short_window")
        if stop_loss_pct < 0 or take_profit_pct < 0:
            raise ValueError("stop_loss_pct and take_profit_pct must be non-negative")
        self.short_window = short_window
        self.long_window = long_window
        self.allow_short = allow_short
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self._closes: list[float] = []
        self._last_fast_over_slow: bool | None = None

    def _risk_exit(self, bar: Bar, position: StrategyPosition | None) -> bool:
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
        self._closes.append(bar.close)
        if self._risk_exit(bar, position):
            return "exit"

        if len(self._closes) < self.long_window:
            return "hold"

        series = pd.Series(self._closes, dtype="float64")
        fast_ema = series.ewm(span=self.short_window, adjust=False).mean().iloc[-1]
        slow_ema = series.ewm(span=self.long_window, adjust=False).mean().iloc[-1]
        fast_over_slow = bool(fast_ema > slow_ema)

        if self._last_fast_over_slow is None:
            self._last_fast_over_slow = fast_over_slow
            return "hold"

        signal: Signal = "hold"
        if not self._last_fast_over_slow and fast_over_slow:
            signal = "long"
        elif self._last_fast_over_slow and not fast_over_slow:
            signal = "short" if self.allow_short else "exit"

        self._last_fast_over_slow = fast_over_slow
        return signal

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

Signal = Literal["long", "short", "exit", "hold", "buy", "sell"]
PositionSide = Literal["flat", "long", "short"]


@dataclass(frozen=True)
class Bar:
    timestamp: datetime | str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class StrategyPosition:
    side: PositionSide = "flat"
    qty: float = 0.0
    entry_price: float = 0.0


class Strategy(ABC):
    @abstractmethod
    def on_bar(self, bar: Bar, position: StrategyPosition | None = None) -> Signal:
        raise NotImplementedError

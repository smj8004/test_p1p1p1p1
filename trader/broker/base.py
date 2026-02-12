from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

Side = Literal["BUY", "SELL", "buy", "sell"]
OrderType = Literal[
    "MARKET",
    "LIMIT",
    "STOP_MARKET",
    "TAKE_PROFIT_MARKET",
    "market",
    "limit",
    "stop_market",
    "take_profit_market",
]
OrderStatus = Literal["NEW", "FILLED", "REJECTED", "CANCELED"]
PositionSide = Literal["BOTH", "LONG", "SHORT"]


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: Side
    amount: float
    order_type: OrderType = "MARKET"
    price: float | None = None
    stop_price: float | None = None
    client_order_id: str | None = None
    reduce_only: bool = False
    time_in_force: str | None = None
    position_side: PositionSide = "BOTH"


@dataclass(frozen=True)
class OrderResult:
    order_id: str
    status: OrderStatus
    filled_qty: float
    avg_price: float
    fee: float = 0.0
    message: str = ""
    client_order_id: str | None = None


class Broker(ABC):
    @abstractmethod
    def place_order(self, request: OrderRequest) -> OrderResult:
        raise NotImplementedError

    @abstractmethod
    def get_balance(self) -> dict[str, float]:
        raise NotImplementedError

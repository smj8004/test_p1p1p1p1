from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import count
from typing import Literal

from .base import Broker, OrderRequest, OrderResult

Liquidity = Literal["taker", "maker"]
TriggerOrderType = Literal["STOP_MARKET", "TAKE_PROFIT_MARKET"]


@dataclass
class PaperPosition:
    qty: float = 0.0
    avg_entry_price: float = 0.0


@dataclass
class PaperOrderState:
    order_id: str
    client_order_id: str | None
    status: Literal["NEW", "FILLED", "REJECTED", "CANCELED"]
    filled_qty: float = 0.0
    avg_price: float = 0.0
    message: str = ""
    side: str = "BUY"
    order_type: str = "MARKET"
    qty: float = 0.0
    price: float | None = None
    stop_price: float | None = None
    reduce_only: bool = False
    position_side: str = "BOTH"


class PaperBroker(Broker):
    def __init__(
        self,
        *,
        starting_cash: float = 10_000.0,
        slippage_bps: float = 0.0,
        taker_fee_bps: float = 5.0,
        maker_fee_bps: float = 2.0,
    ) -> None:
        self.cash = starting_cash
        self.realized_pnl = 0.0
        self.fees_paid = 0.0
        self.positions: dict[str, PaperPosition] = {}
        self.last_price: dict[str, float] = {}
        self._order_seq = count(1)
        self._orders: dict[str, PaperOrderState] = {}
        self._client_to_order: dict[str, str] = {}
        self._pending_trigger_orders: dict[str, OrderRequest] = {}
        self._trigger_fill_events: list[tuple[OrderRequest, OrderResult]] = []
        self.slippage_bps = slippage_bps
        self.taker_fee_bps = taker_fee_bps
        self.maker_fee_bps = maker_fee_bps

    def _normalize_side(self, side: str) -> str:
        up = side.upper()
        if up not in {"BUY", "SELL"}:
            raise ValueError(f"Unsupported side: {side}")
        return up

    def _normalize_order_type(self, order_type: str) -> str:
        up = order_type.upper()
        aliases = {
            "STOP": "STOP_MARKET",
            "STOPMARKET": "STOP_MARKET",
            "TAKEPROFITMARKET": "TAKE_PROFIT_MARKET",
        }
        up = aliases.get(up, up)
        if up not in {"MARKET", "LIMIT", "STOP_MARKET", "TAKE_PROFIT_MARKET"}:
            raise ValueError(f"Unsupported order type: {order_type}")
        return up

    def update_market_price(self, symbol: str, price: float) -> None:
        self.last_price[symbol] = price
        self._trigger_pending_orders(symbol)

    def _next_order_id(self) -> str:
        return f"paper-{next(self._order_seq)}"

    def _fee_rate(self, liquidity: Liquidity) -> float:
        bps = self.taker_fee_bps if liquidity == "taker" else self.maker_fee_bps
        return bps / 10_000.0

    def _fill_price_for_market(self, side: str, market_price: float) -> float:
        slippage = self.slippage_bps / 10_000.0
        if side == "BUY":
            return market_price * (1 + slippage)
        return market_price * (1 - slippage)

    def _resolve_immediate_fill_price(self, request: OrderRequest) -> tuple[float | None, Liquidity]:
        market_price = self.last_price.get(request.symbol)
        order_type = self._normalize_order_type(request.order_type)
        side = self._normalize_side(request.side)

        if order_type == "MARKET":
            if market_price is None:
                return None, "taker"
            return self._fill_price_for_market(side, market_price), "taker"

        if order_type == "LIMIT":
            if request.price is None:
                return None, "maker"
            if market_price is None:
                return request.price, "maker"
            if side == "BUY" and market_price <= request.price:
                return request.price, "maker"
            if side == "SELL" and market_price >= request.price:
                return request.price, "maker"
            return None, "maker"

        return None, "taker"

    def _reduce_only_adjusted_request(self, request: OrderRequest) -> tuple[OrderRequest, str | None]:
        if not request.reduce_only:
            return request, None
        pos = self.positions.get(request.symbol, PaperPosition())
        side = self._normalize_side(request.side)
        if side == "BUY" and pos.qty >= 0:
            return request, "reduce_only BUY requires short position"
        if side == "SELL" and pos.qty <= 0:
            return request, "reduce_only SELL requires long position"
        clamped = min(abs(pos.qty), request.amount)
        if clamped <= 0:
            return request, "reduce_only amount resolved to zero"
        return replace(request, amount=clamped), None

    def _validate_stop_logic(self, request: OrderRequest) -> str | None:
        order_type = self._normalize_order_type(request.order_type)
        if order_type not in {"STOP_MARKET", "TAKE_PROFIT_MARKET"}:
            return None
        if request.stop_price is None:
            return "stop_price is required for trigger orders"
        market_price = self.last_price.get(request.symbol)
        if market_price is None:
            return "market price unavailable for trigger validation"

        side = self._normalize_side(request.side)
        stop = float(request.stop_price)
        if order_type == "STOP_MARKET":
            if side == "BUY" and stop <= market_price:
                return "BUY STOP_MARKET requires stop_price above current price"
            if side == "SELL" and stop >= market_price:
                return "SELL STOP_MARKET requires stop_price below current price"
        if order_type == "TAKE_PROFIT_MARKET":
            if side == "BUY" and stop >= market_price:
                return "BUY TAKE_PROFIT_MARKET requires stop_price below current price"
            if side == "SELL" and stop <= market_price:
                return "SELL TAKE_PROFIT_MARKET requires stop_price above current price"
        return None

    def _apply_fill(self, request: OrderRequest, fill_price: float, fee_rate: float) -> tuple[float, float]:
        side = self._normalize_side(request.side)
        signed_qty = request.amount if side == "BUY" else -request.amount
        pos = self.positions.setdefault(request.symbol, PaperPosition())

        if request.reduce_only and abs(pos.qty + signed_qty) > abs(pos.qty):
            raise ValueError("reduce_only order would increase position")

        realized = 0.0
        if pos.qty == 0 or (pos.qty > 0 and signed_qty > 0) or (pos.qty < 0 and signed_qty < 0):
            new_qty = pos.qty + signed_qty
            if pos.qty == 0:
                pos.avg_entry_price = fill_price
            else:
                total_notional = abs(pos.qty) * pos.avg_entry_price + abs(signed_qty) * fill_price
                pos.avg_entry_price = total_notional / abs(new_qty)
            pos.qty = new_qty
        else:
            closing_qty = min(abs(pos.qty), abs(signed_qty))
            position_sign = 1.0 if pos.qty > 0 else -1.0
            realized = closing_qty * (fill_price - pos.avg_entry_price) * position_sign
            pos.qty += signed_qty
            if abs(pos.qty) < 1e-12:
                pos.qty = 0.0
                pos.avg_entry_price = 0.0
            elif (position_sign > 0 and pos.qty < 0) or (position_sign < 0 and pos.qty > 0):
                pos.avg_entry_price = fill_price

        fee = abs(request.amount * fill_price) * fee_rate
        self.cash += realized
        self.cash -= fee
        self.realized_pnl += realized
        self.fees_paid += fee
        return realized, fee

    def _make_state(self, *, order_id: str, request: OrderRequest, status: str, message: str = "") -> PaperOrderState:
        return PaperOrderState(
            order_id=order_id,
            client_order_id=request.client_order_id,
            status=status,  # type: ignore[arg-type]
            message=message,
            side=self._normalize_side(request.side),
            order_type=self._normalize_order_type(request.order_type),
            qty=request.amount,
            price=request.price,
            stop_price=request.stop_price,
            reduce_only=request.reduce_only,
            position_side=request.position_side,
        )

    def _trigger_pending_orders(self, symbol: str) -> None:
        market_price = self.last_price.get(symbol)
        if market_price is None:
            return
        for order_id in list(self._pending_trigger_orders.keys()):
            req = self._pending_trigger_orders[order_id]
            if req.symbol != symbol:
                continue
            stop = req.stop_price
            if stop is None:
                continue
            side = self._normalize_side(req.side)
            order_type = self._normalize_order_type(req.order_type)

            triggered = False
            if order_type == "STOP_MARKET":
                triggered = (side == "BUY" and market_price >= stop) or (side == "SELL" and market_price <= stop)
            elif order_type == "TAKE_PROFIT_MARKET":
                triggered = (side == "BUY" and market_price <= stop) or (side == "SELL" and market_price >= stop)
            if not triggered:
                continue

            try:
                fill_price = self._fill_price_for_market(side, market_price)
                fee_rate = self._fee_rate("taker")
                self._apply_fill(req, fill_price, fee_rate)
                fee = abs(req.amount * fill_price) * fee_rate
                result = OrderResult(
                    order_id=order_id,
                    status="FILLED",
                    filled_qty=req.amount,
                    avg_price=fill_price,
                    fee=fee,
                    message=f"triggered {order_type}",
                    client_order_id=req.client_order_id,
                )
                state = self._orders[order_id]
                state.status = "FILLED"
                state.filled_qty = req.amount
                state.avg_price = fill_price
                state.message = result.message
                self._trigger_fill_events.append((req, result))
            except Exception as exc:
                state = self._orders[order_id]
                state.status = "REJECTED"
                state.message = str(exc)
            self._pending_trigger_orders.pop(order_id, None)

    def place_order(self, request: OrderRequest) -> OrderResult:
        if request.client_order_id and request.client_order_id in self._client_to_order:
            prev_order_id = self._client_to_order[request.client_order_id]
            prev = self._orders[prev_order_id]
            return OrderResult(
                order_id=prev.order_id,
                status=prev.status,
                filled_qty=prev.filled_qty,
                avg_price=prev.avg_price,
                fee=0.0,
                message="duplicate client_order_id ignored",
                client_order_id=prev.client_order_id,
            )

        adjusted_req, reduce_only_error = self._reduce_only_adjusted_request(request)
        order_id = self._next_order_id()
        state = self._make_state(order_id=order_id, request=adjusted_req, status="NEW")
        self._orders[order_id] = state
        if adjusted_req.client_order_id:
            self._client_to_order[adjusted_req.client_order_id] = order_id

        if reduce_only_error is not None:
            state.status = "REJECTED"
            state.message = reduce_only_error
            return OrderResult(
                order_id=order_id,
                status="REJECTED",
                filled_qty=0.0,
                avg_price=0.0,
                fee=0.0,
                message=state.message,
                client_order_id=adjusted_req.client_order_id,
            )

        stop_error = self._validate_stop_logic(adjusted_req)
        if stop_error is not None:
            state.status = "REJECTED"
            state.message = stop_error
            return OrderResult(
                order_id=order_id,
                status="REJECTED",
                filled_qty=0.0,
                avg_price=0.0,
                fee=0.0,
                message=stop_error,
                client_order_id=adjusted_req.client_order_id,
            )

        order_type = self._normalize_order_type(adjusted_req.order_type)
        if order_type in {"STOP_MARKET", "TAKE_PROFIT_MARKET"}:
            self._pending_trigger_orders[order_id] = adjusted_req
            return OrderResult(
                order_id=order_id,
                status="NEW",
                filled_qty=0.0,
                avg_price=0.0,
                fee=0.0,
                message="trigger order accepted",
                client_order_id=adjusted_req.client_order_id,
            )

        fill_price, liquidity = self._resolve_immediate_fill_price(adjusted_req)
        if fill_price is None:
            if order_type == "LIMIT":
                state.status = "CANCELED"
                state.message = "limit order not marketable in immediate-fill model"
            else:
                state.status = "REJECTED"
                state.message = "Market price unavailable or unsupported order type"
            return OrderResult(
                order_id=order_id,
                status=state.status,
                filled_qty=0.0,
                avg_price=0.0,
                fee=0.0,
                message=state.message,
                client_order_id=adjusted_req.client_order_id,
            )

        try:
            fee_rate = self._fee_rate(liquidity)
            self._apply_fill(adjusted_req, fill_price, fee_rate)
        except ValueError as exc:
            state.status = "REJECTED"
            state.message = str(exc)
            return OrderResult(
                order_id=order_id,
                status="REJECTED",
                filled_qty=0.0,
                avg_price=0.0,
                fee=0.0,
                message=state.message,
                client_order_id=adjusted_req.client_order_id,
            )

        state.status = "FILLED"
        state.filled_qty = adjusted_req.amount
        state.avg_price = fill_price
        fee = abs(adjusted_req.amount * fill_price) * self._fee_rate(liquidity)
        return OrderResult(
            order_id=order_id,
            status="FILLED",
            filled_qty=adjusted_req.amount,
            avg_price=fill_price,
            fee=fee,
            client_order_id=adjusted_req.client_order_id,
        )

    def poll_filled_orders(self) -> list[tuple[OrderRequest, OrderResult]]:
        out = list(self._trigger_fill_events)
        self._trigger_fill_events.clear()
        return out

    def cancel_order(self, order_id: str, *, symbol: str | None = None) -> bool:
        req = self._pending_trigger_orders.pop(order_id, None)
        if req is None:
            return False
        if symbol is not None and req.symbol != symbol:
            self._pending_trigger_orders[order_id] = req
            return False
        state = self._orders.get(order_id)
        if state is not None:
            state.status = "CANCELED"
            state.message = "canceled by runtime"
        return True

    def get_open_orders(self, symbol: str | None = None) -> dict[str, dict[str, object]]:
        out: dict[str, dict[str, object]] = {}
        for oid, req in self._pending_trigger_orders.items():
            if symbol is not None and req.symbol != symbol:
                continue
            out[oid] = {
                "symbol": req.symbol,
                "side": self._normalize_side(req.side),
                "order_type": self._normalize_order_type(req.order_type),
                "qty": req.amount,
                "stop_price": req.stop_price,
                "reduce_only": req.reduce_only,
                "client_order_id": req.client_order_id,
            }
        return out

    def restore_runtime_state(self, *, open_positions: dict[str, object], open_orders: dict[str, object]) -> None:
        pos_qty = float(open_positions.get("qty", 0.0))
        pos_entry = float(open_positions.get("entry_price", 0.0))
        if pos_qty != 0:
            self.positions[open_positions.get("symbol", "BTC/USDT")] = PaperPosition(
                qty=pos_qty,
                avg_entry_price=pos_entry,
            )
        for payload in open_orders.values():
            if not isinstance(payload, dict):
                continue
            client_id = payload.get("client_order_id")
            req = OrderRequest(
                symbol=str(payload.get("symbol", "BTC/USDT")),
                side=str(payload.get("side", "SELL")),
                amount=float(payload.get("qty", 0.0)),
                order_type=str(payload.get("order_type", "STOP_MARKET")),
                stop_price=float(payload.get("stop_price")) if payload.get("stop_price") is not None else None,
                reduce_only=bool(payload.get("reduce_only", False)),
                client_order_id=str(client_id) if client_id else None,
            )
            order_id = str(payload.get("order_id", self._next_order_id()))
            state = self._make_state(order_id=order_id, request=req, status="NEW")
            self._orders[order_id] = state
            if req.client_order_id:
                self._client_to_order[req.client_order_id] = order_id
            self._pending_trigger_orders[order_id] = req

    def get_balance(self) -> dict[str, float]:
        balance = {"cash": self.cash, "realized_pnl": self.realized_pnl, "fees_paid": self.fees_paid}
        for symbol, pos in self.positions.items():
            balance[f"position_qty:{symbol}"] = pos.qty
            balance[f"position_avg:{symbol}"] = pos.avg_entry_price
        return balance

    def get_position(self, symbol: str) -> PaperPosition:
        return self.positions.get(symbol, PaperPosition())

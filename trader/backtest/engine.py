from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

import pandas as pd

from trader.storage import SQLiteStorage
from trader.strategy.base import Bar, Strategy, StrategyPosition

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
OrderStatus = Literal["filled", "rejected", "open"]
Liquidity = Literal["taker", "maker"]
OrderSide = Literal["BUY", "SELL"]
PositionSide = Literal["flat", "long", "short"]
SizingMode = Literal["fixed_usdt", "percent_equity", "atr"]
ExecutionPriceSource = Literal["close", "next_open"]


@dataclass(frozen=True)
class BacktestConfig:
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    initial_equity: float = 10_000.0
    leverage: float = 3.0
    order_type: OrderType = "MARKET"
    execution_price_source: ExecutionPriceSource = "next_open"
    slippage_bps: float = 1.0
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 5.0
    default_liquidity: Liquidity = "taker"
    sizing_mode: SizingMode = "fixed_usdt"
    fixed_notional_usdt: float = 1_000.0
    equity_pct: float = 0.1
    atr_period: int = 14
    atr_risk_pct: float = 0.01
    atr_stop_multiple: float = 2.0
    enable_funding: bool = False
    strategy_name: str = ""
    strategy_params: dict[str, object] = field(default_factory=dict)
    notes: str = ""
    persist_to_db: bool = True
    db_path: Path = Path("data/trader.db")


@dataclass(frozen=True)
class Order:
    order_id: str
    run_id: str
    client_order_id: str
    ts: str
    signal: str
    side: OrderSide
    position_side: Literal["BOTH", "LONG", "SHORT"]
    reduce_only: bool
    order_type: OrderType
    qty: float
    requested_price: float | None
    stop_price: float | None
    time_in_force: str | None
    status: OrderStatus
    reason: str = ""


@dataclass(frozen=True)
class Fill:
    fill_id: str
    run_id: str
    order_id: str
    ts: str
    side: OrderSide
    qty: float
    price: float
    fee: float
    liquidity: Liquidity


@dataclass
class Position:
    side: PositionSide = "flat"
    qty: float = 0.0
    entry_price: float = 0.0
    entry_ts: str = ""
    leverage: float = 1.0
    entry_fee: float = 0.0
    funding_paid: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.side != "flat" and self.qty != 0.0

    def unrealized_pnl(self, mark_price: float) -> float:
        if not self.is_open:
            return 0.0
        return self.qty * (mark_price - self.entry_price)


@dataclass(frozen=True)
class Trade:
    trade_id: str
    run_id: str
    symbol: str
    side: Literal["long", "short"]
    entry_ts: str
    exit_ts: str
    qty: float
    entry_price: float
    exit_price: float
    gross_pnl: float
    fee_paid: float
    funding_paid: float
    net_pnl: float
    return_pct: float
    reason: str = ""


@dataclass
class BacktestResult:
    run_id: str
    summary: dict[str, float]
    initial_equity: float
    equity_curve: list[float]
    orders: list[Order] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    final_position: Position = field(default_factory=Position)


class BacktestEngine:
    def __init__(self, storage: SQLiteStorage | None = None) -> None:
        self.storage = storage

    def _fee_rate(self, *, order_type: OrderType, config: BacktestConfig) -> tuple[float, Liquidity]:
        if str(order_type).upper() == "MARKET" or config.default_liquidity == "taker":
            return config.taker_fee_bps / 10_000.0, "taker"
        return config.maker_fee_bps / 10_000.0, "maker"

    def _normalize_signal(self, signal: str) -> Literal["long", "short", "exit", "hold"]:
        normalized = signal.lower()
        if normalized == "buy":
            return "long"
        if normalized == "sell":
            return "exit"
        if normalized in {"long", "short", "exit", "hold"}:
            return normalized
        return "hold"

    def _compute_atr(self, candles: pd.DataFrame, period: int) -> pd.Series:
        prev_close = candles["close"].shift(1)
        tr_components = pd.concat(
            [
                candles["high"] - candles["low"],
                (candles["high"] - prev_close).abs(),
                (candles["low"] - prev_close).abs(),
            ],
            axis=1,
        )
        true_range = tr_components.max(axis=1)
        return true_range.rolling(window=period, min_periods=1).mean()

    def _execution_price(
        self,
        candles: pd.DataFrame,
        row_index: int,
        *,
        side: OrderSide,
        config: BacktestConfig,
    ) -> tuple[float, str]:
        if config.execution_price_source == "next_open" and row_index + 1 < len(candles):
            base_price = float(candles.iloc[row_index + 1]["open"])
            ts = str(candles.iloc[row_index + 1]["timestamp"])
        else:
            base_price = float(candles.iloc[row_index]["close"])
            ts = str(candles.iloc[row_index]["timestamp"])

        slippage = config.slippage_bps / 10_000.0
        price = base_price * (1 + slippage) if side == "BUY" else base_price * (1 - slippage)
        return price, ts

    def _target_notional(self, *, equity: float, mark_price: float, atr: float, config: BacktestConfig) -> float:
        safe_equity = max(equity, 0.0)
        if safe_equity <= 0:
            return 0.0

        if config.sizing_mode == "fixed_usdt":
            raw_notional = config.fixed_notional_usdt
        elif config.sizing_mode == "percent_equity":
            raw_notional = safe_equity * config.equity_pct * config.leverage
        else:
            if atr <= 0:
                raw_notional = 0.0
            else:
                risk_budget = safe_equity * config.atr_risk_pct
                qty = risk_budget / max(atr * config.atr_stop_multiple, 1e-9)
                raw_notional = qty * mark_price

        max_notional = safe_equity * config.leverage
        return max(0.0, min(raw_notional, max_notional))

    def _maybe_store_order(self, order: Order, storage: SQLiteStorage | None) -> None:
        if storage is not None:
            storage.save_order(order)

    def _maybe_store_fill(self, fill: Fill, storage: SQLiteStorage | None) -> None:
        if storage is not None:
            storage.save_fill(fill)

    def _maybe_store_trade(self, trade: Trade, storage: SQLiteStorage | None) -> None:
        if storage is not None:
            storage.save_trade(trade)

    def run(self, candles: pd.DataFrame, strategy: Strategy, config: BacktestConfig) -> BacktestResult:
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required.difference(candles.columns)
        if missing:
            raise ValueError(f"Missing candle columns: {sorted(missing)}")
        if candles.empty:
            raise ValueError("candles is empty")
        if str(config.order_type).upper() != "MARKET":
            raise NotImplementedError("Only market order execution is implemented in the engine")

        market = candles.reset_index(drop=True).copy()
        market["close"] = market["close"].astype(float)
        market["open"] = market["open"].astype(float)
        market["high"] = market["high"].astype(float)
        market["low"] = market["low"].astype(float)
        market["atr"] = self._compute_atr(market, config.atr_period)

        run_id = uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()

        created_storage = False
        storage = self.storage
        if storage is None and config.persist_to_db:
            storage = SQLiteStorage(config.db_path)
            created_storage = True
        if storage is not None:
            persisted_config = asdict(config)
            if not persisted_config.get("strategy_name"):
                persisted_config["strategy_name"] = strategy.__class__.__name__
            storage.start_backtest_run(
                run_id=run_id,
                created_at=created_at,
                symbol=config.symbol,
                timeframe=config.timeframe,
                initial_equity=config.initial_equity,
                config=persisted_config,
            )

        cash = config.initial_equity
        position = Position(leverage=config.leverage)
        equity_curve: list[float] = []
        orders: list[Order] = []
        fills: list[Fill] = []
        trades: list[Trade] = []
        order_seq = 0
        fill_seq = 0
        trade_seq = 0

        def next_order_id() -> str:
            nonlocal order_seq
            order_seq += 1
            return f"{run_id}-o{order_seq:05d}"

        def next_fill_id() -> str:
            nonlocal fill_seq
            fill_seq += 1
            return f"{run_id}-f{fill_seq:05d}"

        def next_trade_id() -> str:
            nonlocal trade_seq
            trade_seq += 1
            return f"{run_id}-t{trade_seq:05d}"

        def current_equity(mark_price: float) -> float:
            return cash + position.unrealized_pnl(mark_price)

        def close_position(row_index: int, reason_signal: str) -> None:
            nonlocal cash, position
            if not position.is_open:
                return

            side: OrderSide = "SELL" if position.side == "long" else "BUY"
            exec_price, exec_ts = self._execution_price(market, row_index, side=side, config=config)
            order_id = next_order_id()
            order = Order(
                order_id=order_id,
                run_id=run_id,
                client_order_id=f"cid-{order_id[-12:]}",
                ts=exec_ts,
                signal=reason_signal,
                side=side,
                position_side="LONG" if position.side == "long" else "SHORT",
                reduce_only=True,
                order_type=config.order_type,
                qty=abs(position.qty),
                requested_price=exec_price,
                stop_price=None,
                time_in_force=None,
                status="filled",
            )
            orders.append(order)
            self._maybe_store_order(order, storage)

            fee_rate, liquidity = self._fee_rate(order_type=config.order_type, config=config)
            exit_fee = abs(position.qty * exec_price) * fee_rate
            fill = Fill(
                fill_id=next_fill_id(),
                run_id=run_id,
                order_id=order.order_id,
                ts=exec_ts,
                side=side,
                qty=abs(position.qty),
                price=exec_price,
                fee=exit_fee,
                liquidity=liquidity,
            )
            fills.append(fill)
            self._maybe_store_fill(fill, storage)

            gross_pnl = position.qty * (exec_price - position.entry_price)
            fee_paid = position.entry_fee + exit_fee
            funding_paid = position.funding_paid
            net_pnl = gross_pnl - fee_paid - funding_paid
            notional_entry = abs(position.qty * position.entry_price)
            return_pct = (net_pnl / notional_entry) if notional_entry > 0 else 0.0

            cash += gross_pnl
            cash -= exit_fee

            trade = Trade(
                trade_id=next_trade_id(),
                run_id=run_id,
                symbol=config.symbol,
                side="long" if position.side == "long" else "short",
                entry_ts=position.entry_ts,
                exit_ts=exec_ts,
                qty=abs(position.qty),
                entry_price=position.entry_price,
                exit_price=exec_price,
                gross_pnl=gross_pnl,
                fee_paid=fee_paid,
                funding_paid=funding_paid,
                net_pnl=net_pnl,
                return_pct=return_pct,
                reason=reason_signal,
            )
            trades.append(trade)
            self._maybe_store_trade(trade, storage)

            position = Position(leverage=config.leverage)

        def open_position(row_index: int, desired: Literal["long", "short"], source_signal: str) -> None:
            nonlocal cash, position
            mark_price = float(market.iloc[row_index]["close"])
            atr = float(market.iloc[row_index]["atr"])
            equity = current_equity(mark_price)
            notional = self._target_notional(equity=equity, mark_price=mark_price, atr=atr, config=config)
            side: OrderSide = "BUY" if desired == "long" else "SELL"

            if notional <= 0:
                order_id = next_order_id()
                rejected = Order(
                    order_id=order_id,
                    run_id=run_id,
                    client_order_id=f"cid-{order_id[-12:]}",
                    ts=str(market.iloc[row_index]["timestamp"]),
                    signal=source_signal,
                    side=side,
                    position_side="LONG" if desired == "long" else "SHORT",
                    reduce_only=False,
                    order_type=config.order_type,
                    qty=0.0,
                    requested_price=None,
                    stop_price=None,
                    time_in_force=None,
                    status="rejected",
                    reason="Notional size resolved to zero",
                )
                orders.append(rejected)
                self._maybe_store_order(rejected, storage)
                return

            exec_price, exec_ts = self._execution_price(market, row_index, side=side, config=config)
            qty_abs = notional / exec_price
            qty = qty_abs if desired == "long" else -qty_abs
            order_id = next_order_id()

            order = Order(
                order_id=order_id,
                run_id=run_id,
                client_order_id=f"cid-{order_id[-12:]}",
                ts=exec_ts,
                signal=source_signal,
                side=side,
                position_side="LONG" if desired == "long" else "SHORT",
                reduce_only=False,
                order_type=config.order_type,
                qty=qty_abs,
                requested_price=exec_price,
                stop_price=None,
                time_in_force=None,
                status="filled",
            )
            orders.append(order)
            self._maybe_store_order(order, storage)

            fee_rate, liquidity = self._fee_rate(order_type=config.order_type, config=config)
            entry_fee = abs(qty * exec_price) * fee_rate
            fill = Fill(
                fill_id=next_fill_id(),
                run_id=run_id,
                order_id=order.order_id,
                ts=exec_ts,
                side=side,
                qty=qty_abs,
                price=exec_price,
                fee=entry_fee,
                liquidity=liquidity,
            )
            fills.append(fill)
            self._maybe_store_fill(fill, storage)

            cash -= entry_fee
            position = Position(
                side=desired,
                qty=qty,
                entry_price=exec_price,
                entry_ts=exec_ts,
                leverage=config.leverage,
                entry_fee=entry_fee,
                funding_paid=0.0,
            )

        try:
            for idx, row in market.iterrows():
                close_price = float(row["close"])
                strategy_position = StrategyPosition(
                    side=position.side,
                    qty=position.qty,
                    entry_price=position.entry_price,
                )
                bar = Bar(
                    timestamp=row["timestamp"],
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=close_price,
                    volume=float(row["volume"]),
                )
                signal = self._normalize_signal(strategy.on_bar(bar, strategy_position))

                if signal == "exit":
                    close_position(idx, reason_signal="exit")
                elif signal in {"long", "short"}:
                    if position.is_open and position.side != signal:
                        close_position(idx, reason_signal=signal)
                    if (not position.is_open) and signal in {"long", "short"}:
                        open_position(idx, desired=signal, source_signal=signal)

                if config.enable_funding and position.is_open and "funding_rate" in row.index:
                    funding_rate = row["funding_rate"]
                    if pd.notna(funding_rate):
                        funding_payment = position.qty * close_price * float(funding_rate)
                        cash -= funding_payment
                        position.funding_paid += funding_payment

                equity_curve.append(current_equity(close_price))

            if position.is_open:
                close_position(len(market) - 1, reason_signal="forced_exit")
                if equity_curve:
                    equity_curve[-1] = cash

            from trader.backtest.metrics import summarize_performance

            summary = summarize_performance(
                equity_curve=equity_curve,
                trades=trades,
                initial_equity=config.initial_equity,
            )
            if storage is not None:
                storage.finish_backtest_run(run_id, summary)
        finally:
            if created_storage and storage is not None:
                storage.close()

        return BacktestResult(
            run_id=run_id,
            summary=summary,
            initial_equity=config.initial_equity,
            equity_curve=equity_curve,
            orders=orders,
            fills=fills,
            trades=trades,
            final_position=position,
        )

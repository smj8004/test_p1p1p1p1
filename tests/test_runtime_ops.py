from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from trader.broker.base import Broker, OrderRequest, OrderResult
from trader.broker.live_binance import LiveBinanceBroker
from trader.broker.paper import PaperBroker
from trader.data.binance_live import LiveBar
from trader.risk.guards import RiskGuard
from trader.runtime import RuntimeConfig, RuntimeEngine
from trader.storage import SQLiteStorage
from trader.strategy.base import Bar, Strategy, StrategyPosition


class SequenceStrategy(Strategy):
    def __init__(self, signals: list[str]) -> None:
        self.signals = signals
        self.i = 0

    def on_bar(self, bar: Bar, position: StrategyPosition | None = None) -> str:
        if self.i >= len(self.signals):
            return "hold"
        out = self.signals[self.i]
        self.i += 1
        return out


class BoomStrategy(Strategy):
    def on_bar(self, bar: Bar, position: StrategyPosition | None = None) -> str:
        raise RuntimeError("boom")


@dataclass
class StaticFeed:
    bars: list[LiveBar]

    def iter_closed_bars(self, *, max_bars: int | None = None):  # type: ignore[override]
        emitted = 0
        for bar in self.bars:
            yield bar
            emitted += 1
            if max_bars is not None and emitted >= max_bars:
                return

    def close(self) -> None:
        return


class CountingBroker(Broker):
    def __init__(self) -> None:
        self.calls = 0

    def place_order(self, request: OrderRequest) -> OrderResult:
        self.calls += 1
        return OrderResult(
            order_id=f"c-{self.calls}",
            status="FILLED",
            filled_qty=request.amount,
            avg_price=100.0,
            fee=0.0,
            client_order_id=request.client_order_id,
        )

    def get_balance(self) -> dict[str, float]:
        return {"USDT": 1000.0}


class FakeLiveExchange:
    def __init__(self) -> None:
        self.create_calls = 0
        self.last_amount = 0.0

    def create_order(
        self,
        *,
        symbol: str,
        type: str,
        side: str,
        amount: float,
        price: float | None,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        self.create_calls += 1
        self.last_amount = amount
        return {"id": "x1", "status": "open", "filled": 0.0, "price": price or 100.0, "average": 0.0, "fee": {"cost": 0.0}}

    def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        return {"last": 100.0}

    def fetch_positions(self, symbols: list[str]) -> list[dict[str, Any]]:
        return []

    def fetch_order(self, order_id: str, symbol: str) -> dict[str, Any]:
        return {"id": order_id, "status": "open", "filled": 0.0, "price": 100.0, "average": 0.0, "fee": {"cost": 0.0}}

    def fetch_balance(self) -> dict[str, Any]:
        return {"total": {"USDT": 1000.0}}

    def price_to_precision(self, symbol: str, price: float) -> str:
        return f"{price:.2f}"

    def amount_to_precision(self, symbol: str, amount: float) -> str:
        return f"{amount:.3f}"

    def close(self) -> None:
        return


def _bars(prices: list[float]) -> list[LiveBar]:
    ts = pd.Timestamp("2025-01-01T00:00:00Z")
    out: list[LiveBar] = []
    for p in prices:
        out.append(LiveBar(timestamp=ts, open=p, high=p + 1, low=p - 1, close=p, volume=1000.0))
        ts += pd.Timedelta(minutes=1)
    return out


def _count_rows(db_path: Path, sql: str, args: tuple[Any, ...]) -> int:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(sql, args).fetchone()
        return int(row[0]) if row else 0
    finally:
        conn.close()


def _build_runtime(
    *,
    db_path: Path,
    bars: list[LiveBar],
    strategy: Strategy,
    broker: Broker,
    cfg: RuntimeConfig,
) -> tuple[RuntimeEngine, SQLiteStorage]:
    storage = SQLiteStorage(db_path)
    engine = RuntimeEngine(
        config=cfg,
        strategy=strategy,
        broker=broker,
        feed=StaticFeed(bars),
        storage=storage,
        risk_guard=RiskGuard(
            max_order_notional=1_000_000,
            max_position_notional=1_000_000,
            max_daily_loss=1_000_000,
            max_drawdown_pct=1.0,
            max_atr_pct=1.0,
        ),
        initial_equity=10_000.0,
    )
    return engine, storage


def test_reduce_only_cannot_increase_position_paper() -> None:
    broker = PaperBroker(starting_cash=10_000.0)
    broker.update_market_price("BTC/USDT", 100.0)
    broker.place_order(OrderRequest(symbol="BTC/USDT", side="BUY", amount=1.0, order_type="MARKET"))
    bad = broker.place_order(
        OrderRequest(symbol="BTC/USDT", side="BUY", amount=1.0, order_type="MARKET", reduce_only=True)
    )
    assert bad.status == "REJECTED"
    assert broker.get_position("BTC/USDT").qty == 1.0


def test_stop_take_profit_validation_for_long_side_paper() -> None:
    broker = PaperBroker(starting_cash=10_000.0)
    broker.update_market_price("BTC/USDT", 100.0)
    broker.place_order(OrderRequest(symbol="BTC/USDT", side="BUY", amount=1.0, order_type="MARKET"))

    bad_sl = broker.place_order(
        OrderRequest(
            symbol="BTC/USDT",
            side="SELL",
            amount=1.0,
            order_type="STOP_MARKET",
            stop_price=110.0,
            reduce_only=True,
        )
    )
    bad_tp = broker.place_order(
        OrderRequest(
            symbol="BTC/USDT",
            side="SELL",
            amount=1.0,
            order_type="TAKE_PROFIT_MARKET",
            stop_price=90.0,
            reduce_only=True,
        )
    )
    assert bad_sl.status == "REJECTED"
    assert bad_tp.status == "REJECTED"


def test_paper_stop_order_triggers_market_fill() -> None:
    broker = PaperBroker(starting_cash=10_000.0, slippage_bps=0.0)
    broker.update_market_price("BTC/USDT", 100.0)
    broker.place_order(OrderRequest(symbol="BTC/USDT", side="BUY", amount=1.0, order_type="MARKET"))
    stop = broker.place_order(
        OrderRequest(
            symbol="BTC/USDT",
            side="SELL",
            amount=1.0,
            order_type="STOP_MARKET",
            stop_price=95.0,
            reduce_only=True,
        )
    )
    assert stop.status == "NEW"
    broker.update_market_price("BTC/USDT", 94.0)
    fills = broker.poll_filled_orders()
    assert len(fills) == 1
    assert fills[0][1].status == "FILLED"
    assert broker.get_position("BTC/USDT").qty == 0.0


def test_live_reduce_only_is_clamped_and_invalid_stop_is_rejected() -> None:
    exchange = FakeLiveExchange()
    broker = LiveBinanceBroker(
        api_key="k",
        api_secret="s",
        live_trading=True,
        exchange=exchange,
        use_user_stream=False,
    )
    broker._positions["BTCUSDT"] = {"qty": 1.0, "entry_price": 100.0}  # type: ignore[attr-defined]
    clipped = broker.place_order(
        OrderRequest(symbol="BTC/USDT", side="SELL", amount=5.0, order_type="MARKET", reduce_only=True)
    )
    assert clipped.status == "NEW"
    assert exchange.last_amount == 1.0

    bad_sl = broker.place_order(
        OrderRequest(
            symbol="BTC/USDT",
            side="SELL",
            amount=1.0,
            order_type="STOP_MARKET",
            stop_price=110.0,
            reduce_only=True,
        )
    )
    assert bad_sl.status == "REJECTED"
    broker.close()


def test_protective_tp_fill_cancels_sl_in_runtime(tmp_path: Path) -> None:
    db_path = tmp_path / "tp_cancel.sqlite"
    broker = PaperBroker(starting_cash=10_000.0)
    engine, storage = _build_runtime(
        db_path=db_path,
        bars=_bars([100.0, 101.5, 101.0]),
        strategy=SequenceStrategy(["long", "hold", "hold"]),
        broker=broker,
        cfg=RuntimeConfig(
            mode="paper",
            symbol="BTC/USDT",
            timeframe="1m",
            fixed_notional_usdt=1_000.0,
            max_bars=3,
            enable_protective_orders=True,
            protective_stop_loss_pct=0.01,
            protective_take_profit_pct=0.01,
        ),
    )
    try:
        result = engine.run()
    finally:
        storage.close()
    assert result["halted"] is False
    tp_filled = _count_rows(
        db_path,
        "SELECT COUNT(*) FROM orders WHERE run_id = ? AND order_type = 'TAKE_PROFIT_MARKET' AND status = 'filled'",
        (result["run_id"],),
    )
    sl_canceled = _count_rows(
        db_path,
        "SELECT COUNT(*) FROM orders WHERE run_id = ? AND order_type = 'STOP_MARKET' AND status = 'canceled'",
        (result["run_id"],),
    )
    assert tp_filled >= 1
    assert sl_canceled >= 1


def test_dry_run_does_not_call_broker_send(tmp_path: Path) -> None:
    db_path = tmp_path / "dry.sqlite"
    broker = CountingBroker()
    engine, storage = _build_runtime(
        db_path=db_path,
        bars=_bars([100.0]),
        strategy=SequenceStrategy(["long"]),
        broker=broker,
        cfg=RuntimeConfig(mode="live", dry_run=True, one_shot=True, symbol="BTC/USDT", timeframe="1m"),
    )
    try:
        result = engine.run()
    finally:
        storage.close()
    assert result["processed_bars"] == 1
    assert broker.calls == 0


def test_one_shot_processes_single_bar(tmp_path: Path) -> None:
    db_path = tmp_path / "oneshot.sqlite"
    broker = CountingBroker()
    engine, storage = _build_runtime(
        db_path=db_path,
        bars=_bars([100.0, 101.0, 102.0]),
        strategy=SequenceStrategy(["hold", "hold", "hold"]),
        broker=broker,
        cfg=RuntimeConfig(mode="paper", one_shot=True, symbol="BTC/USDT", timeframe="1m"),
    )
    try:
        result = engine.run()
    finally:
        storage.close()
    assert result["processed_bars"] == 1


def test_halt_on_error_saves_state(tmp_path: Path) -> None:
    db_path = tmp_path / "halt.sqlite"
    broker = CountingBroker()
    engine, storage = _build_runtime(
        db_path=db_path,
        bars=_bars([100.0]),
        strategy=BoomStrategy(),
        broker=broker,
        cfg=RuntimeConfig(mode="paper", halt_on_error=True, symbol="BTC/USDT", timeframe="1m"),
    )
    try:
        result = engine.run()
        state = storage.load_runtime_state(result["run_id"])  # type: ignore[arg-type]
    finally:
        storage.close()
    assert result["halted"] is True
    assert state is not None


def test_resume_skips_processed_bars_and_keeps_idempotency(tmp_path: Path) -> None:
    db_path = tmp_path / "resume.sqlite"
    broker = PaperBroker(starting_cash=10_000.0)
    engine1, storage1 = _build_runtime(
        db_path=db_path,
        bars=_bars([100.0, 101.0]),
        strategy=SequenceStrategy(["long", "hold"]),
        broker=broker,
        cfg=RuntimeConfig(mode="paper", max_bars=1, symbol="BTC/USDT", timeframe="1m"),
    )
    try:
        first = engine1.run()
    finally:
        storage1.close()

    order_count_before = _count_rows(
        db_path,
        "SELECT COUNT(*) FROM orders WHERE run_id = ? AND signal = 'long'",
        (first["run_id"],),
    )
    broker2 = PaperBroker(starting_cash=10_000.0)
    engine2, storage2 = _build_runtime(
        db_path=db_path,
        bars=_bars([100.0, 101.0, 102.0]),
        strategy=SequenceStrategy(["long", "hold", "hold"]),
        broker=broker2,
        cfg=RuntimeConfig(
            mode="paper",
            resume=True,
            resume_run_id=str(first["run_id"]),
            symbol="BTC/USDT",
            timeframe="1m",
        ),
    )
    try:
        second = engine2.run()
    finally:
        storage2.close()

    order_count_after = _count_rows(
        db_path,
        "SELECT COUNT(*) FROM orders WHERE run_id = ? AND signal = 'long'",
        (first["run_id"],),
    )
    assert second["run_id"] == first["run_id"]
    assert second["processed_bars"] == 2
    assert order_count_after == order_count_before

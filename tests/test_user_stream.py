from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from trader.broker.base import OrderRequest
from trader.broker.live_binance import LiveBinanceBroker
from trader.data.binance_user_stream import BinanceFuturesListenKeyClient
from trader.storage import SQLiteStorage


class FakeExchange:
    def __init__(self, *, fetch_status: str = "open") -> None:
        self.fetch_status = fetch_status
        self.fetch_calls = 0

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
        return {
            "id": "9001",
            "status": "open",
            "filled": 0.0,
            "price": price or 100.0,
            "average": 0.0,
            "fee": {"cost": 0.0},
        }

    def fetch_order(self, order_id: str, symbol: str) -> dict[str, Any]:
        self.fetch_calls += 1
        return {
            "id": order_id,
            "status": self.fetch_status,
            "filled": 0.0,
            "price": 100.0,
            "average": 0.0,
            "fee": {"cost": 0.0},
        }

    def fetch_balance(self) -> dict[str, Any]:
        return {"total": {"USDT": 1000.0}}

    def close(self) -> None:
        return


class FakeUserStream:
    def __init__(self) -> None:
        self.callback = None
        self.started = False

    def start(self, on_event) -> None:  # type: ignore[no-untyped-def]
        self.callback = on_event
        self.started = True

    def stop(self) -> None:
        self.started = False


def _read_count(db_path: Path, table: str, *, run_id: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE run_id = ?", (run_id,)).fetchone()
        return int(row[0]) if row else 0
    finally:
        conn.close()


def _latest_order_status(db_path: Path, *, run_id: str, order_id: str) -> str | None:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT status
            FROM orders
            WHERE run_id = ? AND order_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (run_id, order_id),
        ).fetchone()
        return str(row[0]) if row else None
    finally:
        conn.close()


def _order_trade_update_event(
    *,
    status: str,
    exec_type: str,
    cumulative_qty: float,
    last_qty: float,
    trade_id: int,
) -> dict[str, Any]:
    return {
        "e": "ORDER_TRADE_UPDATE",
        "E": 1735689601000,
        "o": {
            "s": "BTCUSDT",
            "c": "cid-1",
            "S": "BUY",
            "o": "MARKET",
            "q": "0.5",
            "p": "0",
            "i": 777,
            "X": status,
            "x": exec_type,
            "z": str(cumulative_qty),
            "l": str(last_qty),
            "L": "100.0",
            "ap": "100.0",
            "n": "0.05",
            "N": "USDT",
            "t": trade_id,
            "T": 1735689601000,
            "m": False,
            "R": False,
            "ps": "LONG",
        },
    }


def test_user_stream_filled_event_persists_fill(tmp_path: Path) -> None:
    storage = SQLiteStorage(tmp_path / "trader.db")
    broker = LiveBinanceBroker(
        api_key="k",
        api_secret="s",
        live_trading=True,
        exchange=FakeExchange(),
        use_user_stream=False,
    )
    run_id = "run-filled"
    broker.attach_storage(storage=storage, run_id=run_id)

    broker.handle_user_stream_event(
        _order_trade_update_event(
            status="FILLED",
            exec_type="TRADE",
            cumulative_qty=0.5,
            last_qty=0.5,
            trade_id=1001,
        )
    )
    snapshot = broker.get_state_snapshot()
    storage.close()
    broker.close()

    assert _read_count(tmp_path / "trader.db", "fills", run_id=run_id) == 1
    assert _read_count(tmp_path / "trader.db", "orders", run_id=run_id) >= 1
    assert snapshot["open_orders"] == {}


def test_user_stream_partial_then_filled_accumulates_fills(tmp_path: Path) -> None:
    storage = SQLiteStorage(tmp_path / "trader.db")
    broker = LiveBinanceBroker(
        api_key="k",
        api_secret="s",
        live_trading=True,
        exchange=FakeExchange(),
        use_user_stream=False,
    )
    run_id = "run-partial"
    broker.attach_storage(storage=storage, run_id=run_id)

    broker.handle_user_stream_event(
        _order_trade_update_event(
            status="PARTIALLY_FILLED",
            exec_type="TRADE",
            cumulative_qty=0.2,
            last_qty=0.2,
            trade_id=2001,
        )
    )
    broker.handle_user_stream_event(
        _order_trade_update_event(
            status="FILLED",
            exec_type="TRADE",
            cumulative_qty=0.5,
            last_qty=0.3,
            trade_id=2002,
        )
    )
    storage.close()
    broker.close()

    assert _read_count(tmp_path / "trader.db", "fills", run_id=run_id) == 2
    assert _latest_order_status(tmp_path / "trader.db", run_id=run_id, order_id="777") == "filled"


def test_user_stream_dedupes_same_trade_id(tmp_path: Path) -> None:
    storage = SQLiteStorage(tmp_path / "trader.db")
    broker = LiveBinanceBroker(
        api_key="k",
        api_secret="s",
        live_trading=True,
        exchange=FakeExchange(),
        use_user_stream=False,
    )
    run_id = "run-dedupe"
    broker.attach_storage(storage=storage, run_id=run_id)
    event = _order_trade_update_event(
        status="FILLED",
        exec_type="TRADE",
        cumulative_qty=0.5,
        last_qty=0.5,
        trade_id=3001,
    )

    broker.handle_user_stream_event(event)
    broker.handle_user_stream_event(event)
    storage.close()
    broker.close()

    assert _read_count(tmp_path / "trader.db", "fills", run_id=run_id) == 1


def test_user_stream_timeout_uses_fallback_then_halts_if_inconclusive() -> None:
    fake_exchange = FakeExchange(fetch_status="open")
    fake_stream = FakeUserStream()
    broker = LiveBinanceBroker(
        api_key="k",
        api_secret="s",
        live_trading=True,
        exchange=fake_exchange,
        use_user_stream=True,
        user_stream=fake_stream,  # no events delivered
        ws_order_wait_sec=0.05,
        ws_poll_interval_sec=0.01,
    )
    assert fake_stream.started is True

    with pytest.raises(RuntimeError, match="No terminal order status"):
        broker.place_order(
            OrderRequest(
                symbol="BTC/USDT",
                side="buy",
                amount=0.1,
                order_type="market",
                client_order_id="cid-timeout",
            )
        )
    broker.close()
    assert fake_exchange.fetch_calls >= 1


def test_listen_key_client_lifecycle_calls_expected_endpoints() -> None:
    calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def requester(method: str, path: str, params: dict[str, Any] | None) -> dict[str, Any]:
        calls.append((method, path, params))
        if method == "POST":
            return {"listenKey": "lk-123"}
        return {}

    client = BinanceFuturesListenKeyClient(api_key="k", requester=requester)
    key = client.create_listen_key()
    client.keepalive_listen_key(key)
    client.close_listen_key(key)

    assert key == "lk-123"
    assert calls == [
        ("POST", "/fapi/v1/listenKey", None),
        ("PUT", "/fapi/v1/listenKey", {"listenKey": "lk-123"}),
        ("DELETE", "/fapi/v1/listenKey", {"listenKey": "lk-123"}),
    ]

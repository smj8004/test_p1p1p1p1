from __future__ import annotations

import json
from collections import deque
import queue
import threading
from typing import Any

import pandas as pd

import trader.data.binance_live as live_mod
from trader.data.binance_live import BinanceLiveFeed


def _kline_msg(ts_ms: int, close: float, *, closed: bool) -> dict:
    return {
        "e": "kline",
        "k": {
            "t": ts_ms,
            "o": str(close - 1.0),
            "h": str(close + 1.0),
            "l": str(close - 2.0),
            "c": str(close),
            "v": "100",
            "x": closed,
        },
    }


def _empty_fetcher(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])


class _FakeQueueWorker:
    def __init__(
        self,
        *,
        out_queue: queue.Queue[dict[str, Any]],
        payloads: list[dict[str, Any]],
    ) -> None:
        self._out_queue = out_queue
        self._payloads = payloads
        self.exception: Exception | None = None
        self._alive = False

    def start(self) -> None:
        self._alive = True
        self._out_queue.put({"kind": "event", "event_type": "ws_worker_started", "detail": {"mock": True}})
        for payload in self._payloads:
            self._out_queue.put({"kind": "payload", "payload": payload})
        self._out_queue.put({"kind": "stopped"})
        self._alive = False

    def stop(self) -> None:
        self._alive = False

    def join(self, timeout: float | None = None) -> None:
        self._alive = False

    def is_alive(self) -> bool:
        return self._alive


def test_ws_worker_queue_mock_emits_only_closed_kline(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    msgs = [
        _kline_msg(1735689600000, 100.0, closed=False),
        _kline_msg(1735689600000, 101.0, closed=True),
        _kline_msg(1735689660000, 102.0, closed=False),
    ]

    def fake_worker_factory(self, *, out_queue, stop_event):  # type: ignore[no-untyped-def]
        return _FakeQueueWorker(out_queue=out_queue, payloads=msgs)

    monkeypatch.setattr(BinanceLiveFeed, "_create_ws_worker", fake_worker_factory)

    feed = BinanceLiveFeed(
        symbol="BTC/USDT",
        timeframe="1m",
        mode="websocket",
        fetcher=_empty_fetcher,
        range_fetcher=lambda s, t, a, b: _empty_fetcher(s, t, 0),
        bootstrap_history_bars=0,
    )
    bars = list(feed.iter_closed_bars(max_bars=1))
    assert len(bars) == 1
    assert bars[0].timestamp == pd.Timestamp("2025-01-01T00:00:00Z")


def test_ws_worker_uses_asyncio_run(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    run_calls: list[object] = []
    original_run = live_mod.asyncio.run

    def wrapped_run(coro):  # type: ignore[no-untyped-def]
        run_calls.append(coro)
        return original_run(coro)

    async def fake_ws_loop(self):  # type: ignore[no-untyped-def]
        self._stop_event.set()
        return None

    monkeypatch.setattr(live_mod.asyncio, "run", wrapped_run)
    monkeypatch.setattr(live_mod.WSWorker, "_ws_loop", fake_ws_loop)

    out_q: queue.Queue[dict[str, Any]] = queue.Queue()
    stop_event = threading.Event()
    worker = live_mod.WSWorker(
        url="wss://example.invalid/ws",
        out_queue=out_q,
        stop_event=stop_event,
        ws_max_retries=1,
        ws_backoff_base_sec=0.01,
        ws_backoff_max_sec=0.01,
        ws_receive_timeout_sec=1.0,
    )
    worker.start()
    worker.join(timeout=2.0)

    assert run_calls, "WS worker must use asyncio.run from worker thread"


def test_ws_worker_events_forwarded_to_callback(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    msgs = [_kline_msg(1735689600000, 101.0, closed=True)]

    def fake_worker_factory(self, *, out_queue, stop_event):  # type: ignore[no-untyped-def]
        return _FakeQueueWorker(out_queue=out_queue, payloads=msgs)

    monkeypatch.setattr(BinanceLiveFeed, "_create_ws_worker", fake_worker_factory)
    events: list[str] = []

    feed = BinanceLiveFeed(
        symbol="BTC/USDT",
        timeframe="1m",
        mode="websocket",
        fetcher=_empty_fetcher,
        range_fetcher=lambda s, t, a, b: _empty_fetcher(s, t, 0),
        bootstrap_history_bars=0,
        ws_event_callback=lambda event_type, detail: events.append(event_type),
    )
    bars = list(feed.iter_closed_bars(max_bars=1))

    assert len(bars) == 1
    assert "ws_worker_started" in events


def test_ws_worker_invalid_json_does_not_kill_stream() -> None:
    events: list[str] = []

    async def source():  # type: ignore[no-untyped-def]
        yield b'{"broken":'
        yield json.dumps(_kline_msg(1735689600000, 101.0, closed=True))

    feed = BinanceLiveFeed(
        symbol="BTC/USDT",
        timeframe="1m",
        mode="websocket",
        fetcher=_empty_fetcher,
        range_fetcher=lambda s, t, a, b: _empty_fetcher(s, t, 0),
        bootstrap_history_bars=0,
        ws_worker_message_source=source,
        ws_event_callback=lambda event_type, detail: events.append(event_type),
        ws_max_retries=1,
    )

    bars = list(feed.iter_closed_bars(max_bars=1))
    assert len(bars) == 1
    assert bars[0].timestamp == pd.Timestamp("2025-01-01T00:00:00Z")
    assert "ws_worker_parse_error" in events


def test_ws_emits_only_closed_kline() -> None:
    msgs = deque(
        [
            _kline_msg(1735689600000, 100.0, closed=False),
            _kline_msg(1735689600000, 101.0, closed=True),
            _kline_msg(1735689660000, 102.0, closed=False),
        ]
    )
    feed = BinanceLiveFeed(
        symbol="BTC/USDT",
        timeframe="1m",
        mode="websocket",
        fetcher=_empty_fetcher,
        range_fetcher=lambda s, t, a, b: _empty_fetcher(s, t, 0),
        ws_message_provider=lambda: iter(msgs),
        bootstrap_history_bars=0,
    )
    bars = list(feed.iter_closed_bars(max_bars=1))
    assert len(bars) == 1
    assert bars[0].timestamp == pd.Timestamp("2025-01-01T00:00:00Z")


def test_ws_dedupe_same_timestamp() -> None:
    msgs = deque(
        [
            _kline_msg(1735689600000, 100.0, closed=True),
            _kline_msg(1735689600000, 101.0, closed=True),
            _kline_msg(1735689660000, 102.0, closed=True),
        ]
    )
    feed = BinanceLiveFeed(
        symbol="BTC/USDT",
        timeframe="1m",
        mode="websocket",
        fetcher=_empty_fetcher,
        range_fetcher=lambda s, t, a, b: _empty_fetcher(s, t, 0),
        ws_message_provider=lambda: iter(msgs),
        bootstrap_history_bars=0,
    )
    bars = list(feed.iter_closed_bars(max_bars=2))
    assert [b.timestamp for b in bars] == [
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2025-01-01T00:01:00Z"),
    ]


def test_ws_gap_triggers_backfill() -> None:
    msgs = deque(
        [
            _kline_msg(1735689600000, 100.0, closed=True),
            _kline_msg(1735689720000, 103.0, closed=True),  # missing 00:01
        ]
    )
    backfill_calls: list[tuple[str, str]] = []

    def backfill(symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
        backfill_calls.append((start, end))
        return pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2025-01-01T00:01:00Z")],
                "open": [101.0],
                "high": [102.0],
                "low": [100.0],
                "close": [102.0],
                "volume": [100.0],
            }
        )

    feed = BinanceLiveFeed(
        symbol="BTC/USDT",
        timeframe="1m",
        mode="websocket",
        fetcher=_empty_fetcher,
        range_fetcher=backfill,
        ws_message_provider=lambda: iter(msgs),
        bootstrap_history_bars=0,
    )
    bars = list(feed.iter_closed_bars(max_bars=3))
    assert [b.timestamp for b in bars] == [
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2025-01-01T00:01:00Z"),
        pd.Timestamp("2025-01-01T00:02:00Z"),
    ]
    assert len(backfill_calls) == 1

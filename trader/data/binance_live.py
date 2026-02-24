from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Iterator

import pandas as pd
import websockets

from trader.data.binance import BinanceDataClient
from trader.logger_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class LiveBar:
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str = ""
    is_backfill: bool = False


class WSWorker:
    def __init__(
        self,
        *,
        url: str,
        out_queue: queue.Queue[dict[str, Any]],
        stop_event: threading.Event,
        ws_max_retries: int,
        ws_backoff_base_sec: float,
        ws_backoff_max_sec: float,
        ws_receive_timeout_sec: float,
        message_source: Callable[[], AsyncIterator[str | bytes]] | None = None,
    ) -> None:
        self._url = url
        self._out_queue = out_queue
        self._stop_event = stop_event
        self._ws_max_retries = ws_max_retries
        self._ws_backoff_base_sec = ws_backoff_base_sec
        self._ws_backoff_max_sec = ws_backoff_max_sec
        self._ws_receive_timeout_sec = max(float(ws_receive_timeout_sec), 1.0)
        self._message_source = message_source
        self._last_stream_connected = False
        self._thread = threading.Thread(target=self._run, name="binance-ws-worker", daemon=True)
        self.exception: Exception | None = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _put(self, payload: dict[str, Any]) -> None:
        while True:
            if self._stop_event.is_set():
                try:
                    self._out_queue.put_nowait(payload)
                except queue.Full:
                    pass
                return
            try:
                self._out_queue.put(payload, timeout=0.2)
                return
            except queue.Full:
                continue

    def _emit_event(self, event_type: str, detail: dict[str, Any] | None = None) -> None:
        self._put({"kind": "event", "event_type": event_type, "detail": detail or {}})

    async def _sleep_with_stop(self, seconds: float) -> None:
        deadline = time.monotonic() + max(seconds, 0.0)
        while not self._stop_event.is_set():
            remain = deadline - time.monotonic()
            if remain <= 0:
                return
            await asyncio.sleep(min(remain, 0.2))

    def _decode_message(self, raw: str | bytes) -> str:
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return raw

    async def _consume_ws_stream(self) -> None:
        self._last_stream_connected = False
        last_recv = time.monotonic()
        if self._message_source is not None:
            self._last_stream_connected = True
            self._emit_event("ws_worker_connected", {"url": self._url, "source": "injected"})
            async for raw in self._message_source():
                if self._stop_event.is_set():
                    return
                text = self._decode_message(raw)
                try:
                    payload = json.loads(text)
                except Exception as exc:
                    self._emit_event(
                        "ws_worker_parse_error",
                        {
                            "error": str(exc),
                            "snippet": text[:200],
                        },
                    )
                    continue
                self._put({"kind": "payload", "payload": payload})
                last_recv = time.monotonic()
        else:
            async with websockets.connect(
                self._url,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=5,
                max_queue=1024,
                max_size=2**20,
            ) as ws:
                self._last_stream_connected = True
                self._emit_event("ws_worker_connected", {"url": self._url})
                async for raw in ws:
                    if self._stop_event.is_set():
                        return
                    text = self._decode_message(raw)
                    try:
                        payload = json.loads(text)
                    except Exception as exc:
                        self._emit_event(
                            "ws_worker_parse_error",
                            {
                                "error": str(exc),
                                "snippet": text[:200],
                            },
                        )
                        continue
                    self._put({"kind": "payload", "payload": payload})
                    last_recv = time.monotonic()
        if not self._stop_event.is_set():
            idle_sec = time.monotonic() - last_recv
            if idle_sec >= self._ws_receive_timeout_sec:
                raise TimeoutError(f"websocket receive timeout>{self._ws_receive_timeout_sec:.1f}s")
            raise ConnectionError("WebSocket stream ended unexpectedly")

    async def _ws_loop(self) -> None:
        attempt = 0
        while not self._stop_event.is_set():
            try:
                await self._consume_ws_stream()
                if self._stop_event.is_set():
                    return
                raise ConnectionError("WebSocket stream ended unexpectedly")
            except Exception as exc:
                if self._stop_event.is_set():
                    return
                if self._last_stream_connected:
                    attempt = 0
                attempt += 1
                if attempt > self._ws_max_retries:
                    self._emit_event(
                        "ws_worker_failed",
                        {
                            "attempt": attempt,
                            "max_retries": self._ws_max_retries,
                            "error": str(exc),
                        },
                    )
                    raise RuntimeError(f"WebSocket reconnect exhausted: {exc}") from exc
                delay = min(self._ws_backoff_base_sec * (2 ** (attempt - 1)), self._ws_backoff_max_sec)
                self._emit_event(
                    "ws_worker_reconnect",
                    {
                        "attempt": attempt,
                        "max_retries": self._ws_max_retries,
                        "delay_sec": delay,
                        "error": str(exc),
                    },
                )
                await self._sleep_with_stop(delay)

    def _run(self) -> None:
        self._emit_event("ws_worker_started", {"url": self._url})
        try:
            asyncio.run(self._ws_loop())
        except Exception as exc:
            self.exception = exc
            self._put({"kind": "error", "error": exc})
        finally:
            self._emit_event("ws_worker_stopped", {"error": str(self.exception) if self.exception else ""})
            self._put({"kind": "stopped"})


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    if timeframe.endswith("m"):
        return pd.Timedelta(minutes=int(timeframe[:-1]))
    if timeframe.endswith("h"):
        return pd.Timedelta(hours=int(timeframe[:-1]))
    if timeframe.endswith("d"):
        return pd.Timedelta(days=int(timeframe[:-1]))
    raise ValueError(f"Unsupported timeframe: {timeframe}")


class BinanceLiveFeed:
    def __init__(
        self,
        *,
        symbol: str,
        timeframe: str,
        mode: str = "rest",
        poll_interval_sec: float = 1.0,
        testnet: bool = False,
        fetcher: Callable[[str, str, int], pd.DataFrame] | None = None,
        range_fetcher: Callable[[str, str, str, str], pd.DataFrame] | None = None,
        ws_message_provider: Callable[[], Iterator[dict[str, Any]]] | None = None,
        ws_max_retries: int = 10,
        ws_backoff_base_sec: float = 1.0,
        ws_backoff_max_sec: float = 30.0,
        ws_receive_timeout_sec: float = 90.0,
        bootstrap_history_bars: int = 0,
        ws_event_callback: Callable[[str, dict[str, Any]], None] | None = None,
        ws_worker_message_source: Callable[[], AsyncIterator[str | bytes]] | None = None,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.mode = mode
        self.poll_interval_sec = poll_interval_sec
        self.ws_message_provider = ws_message_provider
        self.ws_max_retries = ws_max_retries
        self.ws_backoff_base_sec = ws_backoff_base_sec
        self.ws_backoff_max_sec = ws_backoff_max_sec
        self.ws_receive_timeout_sec = ws_receive_timeout_sec
        self.bootstrap_history_bars = bootstrap_history_bars
        self._ws_event_callback = ws_event_callback
        self._ws_worker_message_source = ws_worker_message_source

        self._client = None
        self._fetcher = fetcher
        self._range_fetcher = range_fetcher
        if self._fetcher is None or self._range_fetcher is None:
            self._client = BinanceDataClient(testnet=testnet)
            if self._fetcher is None:
                self._fetcher = self._default_fetcher
            if self._range_fetcher is None:
                self._range_fetcher = self._default_range_fetcher
        self._last_emitted_ts: pd.Timestamp | None = None
        self._delta = timeframe_to_timedelta(timeframe)
        self._testnet = testnet
        self._ws_worker: WSWorker | None = None
        self._ws_worker_stop_event: threading.Event | None = None

    def _default_fetcher(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        assert self._client is not None
        return self._client.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)

    def _default_range_fetcher(self, symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
        assert self._client is not None
        return self._client.fetch_ohlcv_range(symbol=symbol, timeframe=timeframe, start=start, end=end)

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        return out.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    def _to_bar(self, row: pd.Series, *, is_backfill: bool = False) -> LiveBar:
        return LiveBar(
            timestamp=pd.to_datetime(row["timestamp"], utc=True),
            symbol=self.symbol,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
            is_backfill=is_backfill,
        )

    def _emit_bar_if_new(self, bar: LiveBar) -> list[LiveBar]:
        if self._last_emitted_ts is not None and bar.timestamp <= self._last_emitted_ts:
            return []
        self._last_emitted_ts = bar.timestamp
        return [bar]

    def _backfill_missing(self, next_ts: pd.Timestamp) -> list[LiveBar]:
        if self._last_emitted_ts is None:
            return []
        missing_start = self._last_emitted_ts + self._delta
        if next_ts <= missing_start:
            return []

        assert self._range_fetcher is not None
        df = self._range_fetcher(
            self.symbol,
            self.timeframe,
            str(missing_start),
            str(next_ts),
        )
        normalized = self._normalize(df)
        emitted: list[LiveBar] = []
        for _, row in normalized.iterrows():
            bar = self._to_bar(row, is_backfill=True)
            if bar.timestamp >= next_ts:
                continue
            emitted.extend(self._emit_bar_if_new(bar))
        if emitted:
            logger.debug(f"[ws] gap detected -> backfilled {len(emitted)} bars up to {next_ts}")
        return emitted

    def _ws_url(self) -> str:
        market = self.symbol.replace("/", "").lower()
        if self._testnet:
            return f"wss://stream.binancefuture.com/ws/{market}@kline_{self.timeframe}"
        return f"wss://fstream.binance.com/ws/{market}@kline_{self.timeframe}"

    def _extract_closed_bar(self, payload: dict[str, Any]) -> LiveBar | None:
        data = payload.get("data", payload)
        if not isinstance(data, dict):
            return None
        kline = data.get("k")
        if not isinstance(kline, dict):
            return None
        if not bool(kline.get("x")):
            return None
        ts = pd.to_datetime(int(kline["t"]), unit="ms", utc=True)
        return LiveBar(
            timestamp=ts,
            symbol=self.symbol,
            open=float(kline["o"]),
            high=float(kline["h"]),
            low=float(kline["l"]),
            close=float(kline["c"]),
            volume=float(kline["v"]),
            is_backfill=False,
        )

    def set_event_callback(self, callback: Callable[[str, dict[str, Any]], None] | None) -> None:
        self._ws_event_callback = callback

    def _emit_ws_event(self, event_type: str, detail: dict[str, Any]) -> None:
        line = f"[ws] {event_type}"
        if detail:
            line = f"{line} {detail}"
        logger.debug(line)
        if self._ws_event_callback is not None:
            try:
                self._ws_event_callback(event_type, detail)
            except Exception:
                pass

    def _create_ws_worker(
        self,
        *,
        out_queue: queue.Queue[dict[str, Any]],
        stop_event: threading.Event,
    ) -> WSWorker:
        return WSWorker(
            url=self._ws_url(),
            out_queue=out_queue,
            stop_event=stop_event,
            ws_max_retries=self.ws_max_retries,
            ws_backoff_base_sec=self.ws_backoff_base_sec,
            ws_backoff_max_sec=self.ws_backoff_max_sec,
            ws_receive_timeout_sec=self.ws_receive_timeout_sec,
            message_source=self._ws_worker_message_source,
        )

    def _iter_ws_messages(self) -> Iterator[dict[str, Any]]:
        if self.ws_message_provider is not None:
            yield from self.ws_message_provider()
            return

        out_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=2048)
        stop_event = threading.Event()
        worker = self._create_ws_worker(out_queue=out_queue, stop_event=stop_event)
        self._ws_worker = worker
        self._ws_worker_stop_event = stop_event
        worker.start()
        try:
            while True:
                try:
                    item = out_queue.get(timeout=1.0)
                except queue.Empty:
                    if worker.exception is not None:
                        raise RuntimeError(f"WebSocket worker failed: {worker.exception}") from worker.exception
                    if not worker.is_alive():
                        return
                    continue

                kind = str(item.get("kind", ""))
                if kind == "payload":
                    payload = item.get("payload")
                    if isinstance(payload, dict):
                        yield payload
                    continue
                if kind == "event":
                    event_type = str(item.get("event_type", "ws_worker_event"))
                    detail = item.get("detail")
                    detail_payload = detail if isinstance(detail, dict) else {"detail": str(detail)}
                    self._emit_ws_event(event_type, detail_payload)
                    continue
                if kind == "error":
                    err = item.get("error")
                    if isinstance(err, Exception):
                        raise RuntimeError(f"WebSocket worker failed: {err}") from err
                    raise RuntimeError(f"WebSocket worker failed: {err}")
                if kind == "stopped":
                    if worker.exception is not None:
                        raise RuntimeError(f"WebSocket worker failed: {worker.exception}") from worker.exception
                    return
        finally:
            stop_event.set()
            worker.stop()
            worker.join(timeout=5.0)
            if worker.is_alive():
                self._emit_ws_event("ws_worker_join_timeout", {"timeout_sec": 5.0})
            self._ws_worker = None
            self._ws_worker_stop_event = None

    def _emit_bootstrap_bars(self, *, max_bars: int | None) -> list[LiveBar]:
        if self.bootstrap_history_bars <= 0:
            return []
        assert self._fetcher is not None
        limit = self.bootstrap_history_bars + 1
        df = self._normalize(self._fetcher(self.symbol, self.timeframe, limit))
        if df.empty:
            return []
        closed = df.iloc[:-1] if len(df) > 1 else df.iloc[0:0]
        bars: list[LiveBar] = []
        for _, row in closed.iterrows():
            bars.extend(self._emit_bar_if_new(self._to_bar(row, is_backfill=True)))
            if max_bars is not None and len(bars) >= max_bars:
                break
        if bars:
            logger.debug(f"[ws] bootstrap emitted {len(bars)} closed bars")
        return bars

    def _iter_closed_bars_websocket(self, *, max_bars: int | None) -> Iterator[LiveBar]:
        emitted = 0

        for bar in self._emit_bootstrap_bars(max_bars=max_bars):
            yield bar
            emitted += 1
            if max_bars is not None and emitted >= max_bars:
                return

        for payload in self._iter_ws_messages():
            bar = self._extract_closed_bar(payload)
            if bar is None:
                continue
            if self._last_emitted_ts is not None and bar.timestamp - self._last_emitted_ts > self._delta:
                for recovered in self._backfill_missing(bar.timestamp):
                    yield recovered
                    emitted += 1
                    if max_bars is not None and emitted >= max_bars:
                        return
            for emitted_bar in self._emit_bar_if_new(bar):
                yield emitted_bar
                emitted += 1
                if max_bars is not None and emitted >= max_bars:
                    return

    def _iter_closed_bars_rest(self, *, max_bars: int | None, history_limit: int) -> Iterator[LiveBar]:
        emitted = 0
        while True:
            assert self._fetcher is not None
            raw = self._fetcher(self.symbol, self.timeframe, history_limit)
            df = self._normalize(raw)
            if df.empty:
                time.sleep(self.poll_interval_sec)
                continue

            closed = df.iloc[:-1] if len(df) > 1 else df.iloc[0:0]
            for _, row in closed.iterrows():
                bar = self._to_bar(row)
                if self._last_emitted_ts is not None and bar.timestamp <= self._last_emitted_ts:
                    continue
                if self._last_emitted_ts is not None and bar.timestamp - self._last_emitted_ts > self._delta:
                    pass
                self._last_emitted_ts = bar.timestamp
                yield bar
                emitted += 1
                if max_bars is not None and emitted >= max_bars:
                    return
            time.sleep(self.poll_interval_sec)

    def iter_closed_bars(self, *, max_bars: int | None = None, history_limit: int = 500) -> Iterator[LiveBar]:
        try:
            if self.mode == "websocket":
                yield from self._iter_closed_bars_websocket(max_bars=max_bars)
                return
            if self.mode != "rest":
                raise ValueError(f"Unknown mode: {self.mode}")
            yield from self._iter_closed_bars_rest(max_bars=max_bars, history_limit=history_limit)
        except KeyboardInterrupt:
            logger.info("[feed] shutdown requested")
            return

    def close(self) -> None:
        if self._ws_worker_stop_event is not None:
            self._ws_worker_stop_event.set()
        if self._ws_worker is not None and self._ws_worker.is_alive():
            self._ws_worker.join(timeout=5.0)
            if self._ws_worker.is_alive():
                logger.warning("[ws] warning: worker thread did not stop within timeout; continuing shutdown")
            self._ws_worker = None
            self._ws_worker_stop_event = None
        if self._client is not None:
            self._client.close()

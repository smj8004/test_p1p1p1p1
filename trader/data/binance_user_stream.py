from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any, Callable, Iterator

import aiohttp
import httpx


class BinanceFuturesListenKeyClient:
    def __init__(
        self,
        *,
        api_key: str,
        testnet: bool = False,
        timeout_sec: float = 10.0,
        requester: Callable[[str, str, dict[str, Any] | None], dict[str, Any]] | None = None,
    ) -> None:
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        self.base_url = "https://testnet.binancefuture.com" if testnet else "https://fapi.binance.com"
        self._requester = requester

    def _request(self, method: str, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._requester is not None:
            return self._requester(method, path, params)
        url = f"{self.base_url}{path}"
        headers = {"X-MBX-APIKEY": self.api_key}
        with httpx.Client(timeout=self.timeout_sec) as client:
            response = client.request(method, url, params=params or {}, headers=headers)
            response.raise_for_status()
            if not response.content:
                return {}
            parsed = response.json()
            return parsed if isinstance(parsed, dict) else {}

    def create_listen_key(self) -> str:
        payload = self._request("POST", "/fapi/v1/listenKey")
        key = payload.get("listenKey")
        if not isinstance(key, str) or not key:
            raise RuntimeError(f"Invalid listenKey response: {payload}")
        return key

    def keepalive_listen_key(self, listen_key: str) -> None:
        self._request("PUT", "/fapi/v1/listenKey", {"listenKey": listen_key})

    def close_listen_key(self, listen_key: str) -> None:
        self._request("DELETE", "/fapi/v1/listenKey", {"listenKey": listen_key})


class _ListenKeyExpired(Exception):
    pass


class BinanceUserStream:
    def __init__(
        self,
        *,
        listen_key_client: BinanceFuturesListenKeyClient,
        testnet: bool = False,
        renew_secs: int = 1800,
        ws_receive_timeout_sec: float = 90.0,
        ws_max_retries: int = 20,
        ws_backoff_base_sec: float = 1.0,
        ws_backoff_max_sec: float = 30.0,
        ws_message_provider: Callable[[str], Iterator[dict[str, Any]]] | None = None,
    ) -> None:
        self.listen_key_client = listen_key_client
        self.testnet = testnet
        self.renew_secs = renew_secs
        self.ws_receive_timeout_sec = ws_receive_timeout_sec
        self.ws_max_retries = ws_max_retries
        self.ws_backoff_base_sec = ws_backoff_base_sec
        self.ws_backoff_max_sec = ws_backoff_max_sec
        self.ws_message_provider = ws_message_provider

        self._listen_key_lock = threading.Lock()
        self._listen_key: str | None = None
        self._stop_event = threading.Event()
        self._force_reconnect = threading.Event()
        self._ws_thread: threading.Thread | None = None
        self._renew_thread: threading.Thread | None = None
        self._on_event: Callable[[dict[str, Any]], None] | None = None

    def _ws_url(self, listen_key: str) -> str:
        if self.testnet:
            return f"wss://stream.binancefuture.com/ws/{listen_key}"
        return f"wss://fstream.binance.com/ws/{listen_key}"

    def _get_listen_key(self) -> str | None:
        with self._listen_key_lock:
            return self._listen_key

    def _set_listen_key(self, listen_key: str | None) -> None:
        with self._listen_key_lock:
            self._listen_key = listen_key

    def _iter_ws_messages_native(self, listen_key: str) -> Iterator[dict[str, Any]]:
        loop = asyncio.new_event_loop()
        session: aiohttp.ClientSession | None = None
        ws: aiohttp.ClientWebSocketResponse | None = None
        try:
            asyncio.set_event_loop(loop)
            session = aiohttp.ClientSession()
            ws = loop.run_until_complete(
                session.ws_connect(
                    self._ws_url(listen_key),
                    heartbeat=20.0,
                    autoclose=True,
                    autoping=True,
                )
            )
            while not self._stop_event.is_set() and not self._force_reconnect.is_set():
                try:
                    msg = loop.run_until_complete(ws.receive(timeout=self.ws_receive_timeout_sec))
                except TimeoutError as exc:
                    raise ConnectionError("user stream timeout") from exc
                if msg.type == aiohttp.WSMsgType.TEXT:
                    payload = json.loads(msg.data)
                    if isinstance(payload, dict):
                        yield payload
                    continue
                if msg.type in {aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR}:
                    raise ConnectionError(f"user stream closed: {msg.type}")
        finally:
            if ws is not None:
                loop.run_until_complete(ws.close())
            if session is not None:
                loop.run_until_complete(session.close())
            loop.close()

    def _iter_ws_messages(self, listen_key: str) -> Iterator[dict[str, Any]]:
        if self.ws_message_provider is not None:
            yield from self.ws_message_provider(listen_key)
            return
        yield from self._iter_ws_messages_native(listen_key)

    def _renew_loop(self) -> None:
        while not self._stop_event.wait(self.renew_secs):
            listen_key = self._get_listen_key()
            if not listen_key:
                continue
            try:
                self.listen_key_client.keepalive_listen_key(listen_key)
            except Exception as exc:
                print(f"[user-stream] keepalive failed: {exc}")
                self._force_reconnect.set()

    def _ensure_listen_key(self) -> str:
        current = self._get_listen_key()
        if current:
            return current
        listen_key = self.listen_key_client.create_listen_key()
        self._set_listen_key(listen_key)
        return listen_key

    def _safe_close_listen_key(self, listen_key: str | None) -> None:
        if not listen_key:
            return
        try:
            self.listen_key_client.close_listen_key(listen_key)
        except Exception:
            pass

    def _emit(self, payload: dict[str, Any]) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(payload)
        except Exception:
            pass

    def _ws_loop(self) -> None:
        attempt = 0
        while not self._stop_event.is_set():
            listen_key: str | None = None
            self._force_reconnect.clear()
            try:
                listen_key = self._ensure_listen_key()
                for payload in self._iter_ws_messages(listen_key):
                    if self._stop_event.is_set() or self._force_reconnect.is_set():
                        break
                    if payload.get("e") == "listenKeyExpired":
                        raise _ListenKeyExpired("listenKey expired")
                    self._emit(payload)
                if self._stop_event.is_set():
                    break
                raise ConnectionError("user stream disconnected")
            except _ListenKeyExpired:
                self._safe_close_listen_key(listen_key)
                self._set_listen_key(None)
                attempt = 0
                continue
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                self._safe_close_listen_key(listen_key)
                self._set_listen_key(None)
                attempt += 1
                if attempt > self.ws_max_retries:
                    print(f"[user-stream] reconnect exhausted: {exc}")
                    break
                delay = min(self.ws_backoff_base_sec * (2 ** (attempt - 1)), self.ws_backoff_max_sec)
                print(f"[user-stream] disconnected ({exc}); reconnect in {delay:.1f}s ({attempt}/{self.ws_max_retries})")
                self._stop_event.wait(delay)
                continue

    def start(self, on_event: Callable[[dict[str, Any]], None]) -> None:
        if self._ws_thread is not None and self._ws_thread.is_alive():
            return
        self._on_event = on_event
        self._stop_event.clear()
        self._force_reconnect.clear()
        self._renew_thread = threading.Thread(target=self._renew_loop, name="binance-user-stream-renew", daemon=True)
        self._ws_thread = threading.Thread(target=self._ws_loop, name="binance-user-stream-ws", daemon=True)
        self._renew_thread.start()
        self._ws_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._force_reconnect.set()
        if self._ws_thread is not None:
            self._ws_thread.join(timeout=5.0)
        if self._renew_thread is not None:
            self._renew_thread.join(timeout=5.0)
        listen_key = self._get_listen_key()
        self._safe_close_listen_key(listen_key)
        self._set_listen_key(None)

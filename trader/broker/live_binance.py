from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from threading import Lock
from typing import Any

import ccxt
import pandas as pd

from trader.data.binance_user_stream import BinanceFuturesListenKeyClient, BinanceUserStream

from .base import Broker, OrderRequest, OrderResult

TERMINAL_STATUSES = {"FILLED", "CANCELED", "REJECTED"}


class LiveBinanceBroker(Broker):
    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        live_trading: bool = False,
        request_timeout_ms: int = 10_000,
        max_retries: int = 3,
        retry_delay_sec: float = 1.0,
        use_user_stream: bool = False,
        listenkey_renew_secs: int = 1800,
        ws_order_wait_sec: float = 8.0,
        ws_poll_interval_sec: float = 0.1,
        exchange: Any | None = None,
        user_stream: BinanceUserStream | None = None,
    ) -> None:
        normalized_key = api_key.strip()
        normalized_secret = api_secret.strip()
        self._api_key_present = bool(normalized_key)
        self._api_secret_present = bool(normalized_secret)
        self._api_key_len = len(normalized_key)
        self._api_secret_len = len(normalized_secret)
        self.binance_env = "testnet" if testnet else "mainnet"
        if exchange is not None:
            self.exchange = exchange
        else:
            self.exchange = ccxt.binance(
                {
                    "apiKey": api_key,
                    "secret": api_secret,
                    "enableRateLimit": True,
                    "timeout": request_timeout_ms,
                    "options": {"defaultType": "future"},
                }
            )
            if testnet:
                self._configure_futures_testnet_urls(self.exchange)
        self.live_trading = live_trading
        self.max_retries = max_retries
        self.retry_delay_sec = retry_delay_sec
        self.ws_order_wait_sec = ws_order_wait_sec
        self.ws_poll_interval_sec = ws_poll_interval_sec
        self._client_results: dict[str, OrderResult] = {}
        self._lock = Lock()
        self._open_orders: dict[str, dict[str, Any]] = {}
        self._orders_by_id: dict[str, dict[str, Any]] = {}
        self._positions: dict[str, dict[str, float]] = {}
        self._seen_fill_keys: set[str] = set()
        self._last_user_stream_update: float | None = None
        self._storage: Any | None = None
        self._storage_run_id: str | None = None

        self.use_user_stream = bool(use_user_stream)
        self.handles_fill_persistence = self.use_user_stream
        self._user_stream = user_stream
        if self.use_user_stream and self._user_stream is None:
            listen_client = BinanceFuturesListenKeyClient(api_key=api_key, testnet=testnet)
            self._user_stream = BinanceUserStream(
                listen_key_client=listen_client,
                testnet=testnet,
                renew_secs=listenkey_renew_secs,
            )
        if self.use_user_stream and self._user_stream is not None:
            self._user_stream.start(self.handle_user_stream_event)

    def _configure_futures_testnet_urls(self, exchange: Any) -> None:
        base = "https://testnet.binancefuture.com"
        api = exchange.urls.get("api")
        if not isinstance(api, dict):
            return
        api["fapiPublic"] = f"{base}/fapi/v1"
        api["fapiPublicV2"] = f"{base}/fapi/v2"
        api["fapiPublicV3"] = f"{base}/fapi/v3"
        api["fapiPrivate"] = f"{base}/fapi/v1"
        api["fapiPrivateV2"] = f"{base}/fapi/v2"
        api["fapiPrivateV3"] = f"{base}/fapi/v3"
        api["fapiData"] = f"{base}/futures/data"

    def _futures_base_url(self) -> str:
        api = self.exchange.urls.get("api") if hasattr(self.exchange, "urls") else None
        if isinstance(api, dict):
            for key in ("fapiPrivateV3", "fapiPrivateV2", "fapiPrivate", "fapiPublicV3", "fapiPublicV2", "fapiPublic"):
                value = api.get(key)
                if isinstance(value, str) and value:
                    return value
        if self.binance_env == "testnet":
            return "https://testnet.binancefuture.com/fapi/v1"
        return "https://fapi.binance.com/fapi/v1"

    def _futures_ws_url(self) -> str:
        if self.binance_env == "testnet":
            return "wss://stream.binancefuture.com/ws"
        return "wss://fstream.binance.com/ws"

    def _last_request_url(self) -> str | None:
        for attr in ("last_request_url", "lastRequestUrl"):
            value = getattr(self.exchange, attr, None)
            if isinstance(value, str) and value:
                return value
        return None

    def _coerce_http_status(self, value: Any) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if text.isdigit():
                return int(text)
            m = re.match(r"(\d{3})", text)
            if m:
                return int(m.group(1))
        return None

    def _http_status_from_exchange(self) -> int | None:
        for attr in (
            "last_http_status",
            "lastHttpStatus",
            "last_response_status",
            "lastResponseStatus",
            "last_status_code",
            "lastStatusCode",
        ):
            status = self._coerce_http_status(getattr(self.exchange, attr, None))
            if status is not None:
                return status
        headers = getattr(self.exchange, "last_response_headers", None)
        if headers is None:
            headers = getattr(self.exchange, "lastResponseHeaders", None)
        if isinstance(headers, dict):
            for key in ("status", "Status", ":status"):
                status = self._coerce_http_status(headers.get(key))
                if status is not None:
                    return status
        return None

    def _http_status_from_exception(self, exc: Exception) -> int | None:
        for attr in ("http_status", "status_code", "status"):
            status = self._coerce_http_status(getattr(exc, attr, None))
            if status is not None:
                return status
        return None

    def _extract_error_code(self, exc: Exception) -> int | None:
        for attr in ("code", "error_code"):
            raw = getattr(exc, attr, None)
            if isinstance(raw, int):
                return raw
            if isinstance(raw, str) and raw.strip().lstrip("-").isdigit():
                return int(raw.strip())
        text = str(exc)
        patterns = [
            r'"code"\s*:\s*(-?\d+)',
            r"\bcode\s*[=:]\s*(-?\d+)",
        ]
        for pattern in patterns:
            m = re.search(pattern, text)
            if m:
                return int(m.group(1))
        return None

    def _http_status_or_placeholder(self, *, ok: bool, preferred: int | None = None) -> int | str:
        if preferred is not None:
            return preferred
        status = self._http_status_from_exchange()
        if status is not None:
            return status
        return "2xx" if ok else "unknown"

    def _auth_2015_guidance(self) -> list[str]:
        return [
            "Possible testnet/mainnet API key mix-up: check BINANCE_ENV and key issuance environment.",
            "Possible IP whitelist restriction: your current IP may not be allowed by the API key.",
            "Possible Futures permission disabled on the API key.",
            "Possible API key/secret mismatch (stale secret, partial rotation, or typo).",
        ]

    def _symbol_filter_snapshot(self, market: dict[str, Any]) -> dict[str, float | None]:
        filters = market.get("info", {}).get("filters", []) if isinstance(market.get("info"), dict) else []
        tick_size = None
        step_size = None
        min_notional = None
        if isinstance(filters, list):
            for filt in filters:
                if not isinstance(filt, dict):
                    continue
                ftype = str(filt.get("filterType", "")).upper()
                if ftype == "PRICE_FILTER":
                    tick_size = self._as_float(filt.get("tickSize"))
                elif ftype == "LOT_SIZE":
                    step_size = self._as_float(filt.get("stepSize"))
                elif ftype in {"MIN_NOTIONAL", "NOTIONAL"}:
                    min_notional = self._as_float(filt.get("notional", filt.get("minNotional")))
        return {"tick_size": tick_size, "step_size": step_size, "min_notional": min_notional}

    def _symbol_filter_snapshot_from_exchange_info(self, exchange_info: dict[str, Any], symbol: str) -> dict[str, float | None]:
        rows = exchange_info.get("symbols")
        if not isinstance(rows, list):
            raise RuntimeError("exchangeInfo has no symbols list")
        target = self._market_symbol_key(symbol)
        matched = None
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("symbol", "")).upper() == target:
                matched = row
                break
        if not isinstance(matched, dict):
            raise RuntimeError(f"symbol not found: {symbol}")
        if str(matched.get("status", "TRADING")).upper() != "TRADING":
            raise RuntimeError(f"symbol inactive: {symbol}")
        fake_market = {"info": {"filters": matched.get("filters", [])}}
        return self._symbol_filter_snapshot(fake_market)

    def _fetch_futures_balance_direct(self) -> tuple[bool, dict[str, Any]]:
        # Prefer USD-M futures private endpoints to avoid spot SAPI-dependent routes.
        method_names = (
            "fapiPrivateV2GetBalance",
            "fapiPrivateGetBalance",
            "fapiPrivateV3GetBalance",
            "fapiprivatev2GetBalance",
            "fapiprivateGetBalance",
            "fapiprivatev3GetBalance",
        )
        for method_name in method_names:
            method = getattr(self.exchange, method_name, None)
            if not callable(method):
                continue
            try:
                method()
                return True, {
                    "method": method_name,
                    "endpoint": self._last_request_url() or "GET /fapi/v2/balance",
                    "http_status": self._http_status_or_placeholder(ok=True),
                }
            except Exception as exc:
                return False, {
                    "method": method_name,
                    "endpoint": self._last_request_url() or "GET /fapi/v2/balance",
                    "http_status": self._http_status_or_placeholder(
                        ok=False,
                        preferred=self._http_status_from_exception(exc),
                    ),
                    "error_code": self._extract_error_code(exc),
                    "error": str(exc),
                }

        # Compatibility fallback for exchanges/fakes without direct fapi methods.
        try:
            try:
                self.exchange.fetch_balance(params={"type": "future"})
            except TypeError:
                self.exchange.fetch_balance()
            return True, {
                "method": "fetch_balance",
                "endpoint": self._last_request_url() or "GET /fapi/v2/balance (fetch_balance fallback)",
                "http_status": self._http_status_or_placeholder(ok=True),
            }
        except Exception as exc:
            return False, {
                "method": "fetch_balance",
                "endpoint": self._last_request_url() or "GET /fapi/v2/balance (fetch_balance fallback)",
                "http_status": self._http_status_or_placeholder(
                    ok=False,
                    preferred=self._http_status_from_exception(exc),
                ),
                "error_code": self._extract_error_code(exc),
                "error": str(exc),
            }

    def _fetch_exchange_info_public(self) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        method_names = (
            "fapiPublicGetExchangeInfo",
            "fapipublicGetExchangeinfo",
            "fapipublic_get_exchangeinfo",
        )
        for method_name in method_names:
            method = getattr(self.exchange, method_name, None)
            if not callable(method):
                continue
            try:
                data = method()
                if isinstance(data, dict):
                    return data, {
                        "method": method_name,
                        "endpoint": self._last_request_url() or "GET /fapi/v1/exchangeInfo",
                        "http_status": self._http_status_or_placeholder(ok=True),
                    }
                return None, {
                    "method": method_name,
                    "endpoint": self._last_request_url() or "GET /fapi/v1/exchangeInfo",
                    "http_status": self._http_status_or_placeholder(ok=False),
                    "error": "invalid exchangeInfo payload",
                }
            except Exception as exc:
                return None, {
                    "method": method_name,
                    "endpoint": self._last_request_url() or "GET /fapi/v1/exchangeInfo",
                    "http_status": self._http_status_or_placeholder(
                        ok=False,
                        preferred=self._http_status_from_exception(exc),
                    ),
                    "error_code": self._extract_error_code(exc),
                    "error": str(exc),
                }

        # Compatibility fallback for fakes that only expose load_markets.
        try:
            markets = self.exchange.load_markets()
            if isinstance(markets, dict):
                return {"_markets": markets}, {
                    "method": "load_markets",
                    "endpoint": self._last_request_url() or "GET /fapi/v1/exchangeInfo (load_markets fallback)",
                    "http_status": self._http_status_or_placeholder(ok=True),
                }
            return None, {
                "method": "load_markets",
                "endpoint": self._last_request_url() or "GET /fapi/v1/exchangeInfo (load_markets fallback)",
                "http_status": self._http_status_or_placeholder(ok=False),
                "error": "invalid load_markets payload",
            }
        except Exception as exc:
            return None, {
                "method": "load_markets",
                "endpoint": self._last_request_url() or "GET /fapi/v1/exchangeInfo (load_markets fallback)",
                "http_status": self._http_status_or_placeholder(
                    ok=False,
                    preferred=self._http_status_from_exception(exc),
                ),
                "error_code": self._extract_error_code(exc),
                "error": str(exc),
            }

    def _fetch_position_risk(self, symbol: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        market_symbol = self._market_symbol_key(symbol)
        last_error: dict[str, Any] | None = None
        for method_name in ("fapiPrivateGetPositionRisk", "fapiPrivate_get_positionrisk"):
            method = getattr(self.exchange, method_name, None)
            if not callable(method):
                continue
            try:
                rows = method({"symbol": market_symbol})
            except TypeError:
                try:
                    rows = method()
                except Exception as exc:
                    last_error = {
                        "attempted": True,
                        "ok": False,
                        "method": method_name,
                        "endpoint": self._last_request_url() or "GET /fapi/v2/positionRisk",
                        "http_status": self._http_status_or_placeholder(
                            ok=False,
                            preferred=self._http_status_from_exception(exc),
                        ),
                        "detail": str(exc),
                        "error_code": self._extract_error_code(exc),
                    }
                    continue
            except Exception as exc:
                last_error = {
                    "attempted": True,
                    "ok": False,
                    "method": method_name,
                    "endpoint": self._last_request_url() or "GET /fapi/v2/positionRisk",
                    "http_status": self._http_status_or_placeholder(
                        ok=False,
                        preferred=self._http_status_from_exception(exc),
                    ),
                    "detail": str(exc),
                    "error_code": self._extract_error_code(exc),
                }
                continue
            endpoint = self._last_request_url() or "GET /fapi/v2/positionRisk"
            success_meta = {
                "attempted": True,
                "ok": True,
                "method": method_name,
                "endpoint": endpoint,
                "http_status": self._http_status_or_placeholder(ok=True),
            }
            if isinstance(rows, list) and rows:
                row = rows[0]
                if isinstance(row, dict):
                    return row, success_meta
            if isinstance(rows, dict):
                return rows, success_meta
            return None, success_meta
        if last_error is not None:
            return None, last_error
        return (
            None,
            {
                "attempted": False,
                "ok": False,
                "endpoint": "GET /fapi/v2/positionRisk",
                "http_status": "unavailable",
                "detail": "positionRisk endpoint unavailable in exchange adapter",
            },
        )

    def preflight_check(
        self,
        *,
        symbol: str,
        max_time_drift_ms: int = 5_000,
        expected_leverage: float | None = None,
        expected_margin_mode: str | None = None,
        include_leverage_margin: bool = True,
    ) -> tuple[bool, list[dict[str, Any]]]:
        checks: list[dict[str, Any]] = []

        def _add(
            name: str,
            ok: bool,
            detail: str,
            *,
            required: bool = True,
            event_type: str = "preflight_check",
            extra: dict[str, Any] | None = None,
        ) -> None:
            row: dict[str, Any] = {
                "check": name,
                "ok": ok,
                "required": required,
                "detail": detail,
                "env": self.binance_env,
                "event_type": event_type,
            }
            if extra:
                row.update(extra)
            checks.append(row)

        def _add_endpoint(
            *,
            endpoint_name: str,
            ok: bool,
            endpoint_hint: str,
            detail: str = "",
            http_status: int | None = None,
            error_code: int | None = None,
        ) -> None:
            endpoint = self._last_request_url() or endpoint_hint
            payload = {
                "endpoint_name": endpoint_name,
                "endpoint": endpoint,
                "http_status": self._http_status_or_placeholder(ok=ok, preferred=http_status),
            }
            if error_code is not None:
                payload["error_code"] = error_code
            _add(
                "endpoint_call",
                ok,
                detail or f"{endpoint_name} endpoint call {'ok' if ok else 'failed'}",
                required=False,
                event_type="preflight_endpoint",
                extra=payload,
            )

        _add(
            "environment",
            True,
            "binance environment detected",
            required=False,
            event_type="preflight_environment",
            extra={
                "binance_env": self.binance_env,
                "base_url": self._futures_base_url(),
                "ws_url": self._futures_ws_url(),
            },
        )

        _add(
            "credentials_present",
            self._api_key_present and self._api_secret_present,
            "api key/secret present" if (self._api_key_present and self._api_secret_present) else "missing api key/secret",
        )
        _add(
            "credentials_meta",
            self._api_key_present and self._api_secret_present,
            "api key/secret presence + length snapshot",
            required=False,
            event_type="preflight_credentials",
            extra={
                "api_key_present": self._api_key_present,
                "api_key_len": self._api_key_len,
                "api_secret_present": self._api_secret_present,
                "api_secret_len": self._api_secret_len,
            },
        )

        futures_ok = True
        futures_detail = "futures account access ok"
        futures_endpoint_hint = "GET /fapi/v2/balance (fetch_balance type=future)"
        futures_ok, futures_meta = self._fetch_futures_balance_direct()
        futures_endpoint_hint = str(futures_meta.get("endpoint", futures_endpoint_hint))
        futures_method = str(futures_meta.get("method", "")).strip()
        futures_http_status = self._coerce_http_status(futures_meta.get("http_status"))
        futures_error_code_raw = futures_meta.get("error_code")
        futures_error_code = (
            int(futures_error_code_raw)
            if isinstance(futures_error_code_raw, str) and futures_error_code_raw.strip().lstrip("-").isdigit()
            else futures_error_code_raw
            if isinstance(futures_error_code_raw, int)
            else None
        )
        if futures_ok:
            success_detail = f"endpoint call ok via {futures_method}" if futures_method else ""
            _add_endpoint(
                endpoint_name="futures_permission",
                ok=True,
                endpoint_hint=futures_endpoint_hint,
                detail=success_detail,
                http_status=futures_http_status,
            )
        else:
            futures_error = str(futures_meta.get("error", "")).strip() or "unknown error"
            failure_detail = f"endpoint call failed: {futures_error}"
            if futures_method:
                failure_detail = f"{failure_detail} (method={futures_method})"
            futures_detail = f"futures account access failed: {futures_error}"
            _add_endpoint(
                endpoint_name="futures_permission",
                ok=False,
                endpoint_hint=futures_endpoint_hint,
                detail=failure_detail,
                http_status=futures_http_status,
                error_code=futures_error_code,
            )
            if futures_error_code == -2015:
                guidance = self._auth_2015_guidance()
                futures_detail = f"{futures_detail} (error_code=-2015)"
                _add(
                    "auth_error_guidance",
                    False,
                    "Binance -2015 guidance attached",
                    required=False,
                    event_type="preflight_auth_guidance",
                    extra={"error_code": -2015, "guide": guidance},
                )
        _add("futures_permission", futures_ok, futures_detail)

        time_ok = True
        time_detail = "server time check ok"
        time_endpoint_hint = "GET /fapi/v1/time (fetch_time)"
        try:
            server_ms = None
            fetch_time = getattr(self.exchange, "fetch_time", None)
            if callable(fetch_time):
                server_ms = fetch_time()
            if server_ms is None:
                raise RuntimeError("exchange.fetch_time unavailable")
            _add_endpoint(
                endpoint_name="server_time_sync",
                ok=True,
                endpoint_hint=time_endpoint_hint,
            )
            drift = abs(int(time.time() * 1000) - int(server_ms))
            if drift > max_time_drift_ms:
                time_ok = False
                time_detail = f"time drift too large: {drift}ms > {max_time_drift_ms}ms"
            else:
                time_detail = f"time drift {drift}ms"
        except Exception as exc:
            time_ok = False
            time_detail = f"time sync check failed: {exc}"
            _add_endpoint(
                endpoint_name="server_time_sync",
                ok=False,
                endpoint_hint=time_endpoint_hint,
                detail=f"endpoint call failed: {exc}",
                http_status=self._http_status_from_exception(exc),
                error_code=self._extract_error_code(exc),
            )
        _add("server_time_sync", time_ok, time_detail)

        market_ok = True
        market_detail = "symbol filters loaded"
        market_endpoint_hint = "GET /fapi/v1/exchangeInfo (load_markets)"
        market_http_status: int | None = None
        market_error_code: int | None = None
        try:
            exchange_info, market_meta = self._fetch_exchange_info_public()
            market_endpoint_hint = str(market_meta.get("endpoint", market_endpoint_hint))
            market_method = str(market_meta.get("method", "")).strip()
            market_http_status = self._coerce_http_status(market_meta.get("http_status"))
            market_error_code_raw = market_meta.get("error_code")
            market_error_code = (
                int(market_error_code_raw)
                if isinstance(market_error_code_raw, str) and market_error_code_raw.strip().lstrip("-").isdigit()
                else market_error_code_raw
                if isinstance(market_error_code_raw, int)
                else None
            )
            if exchange_info is None:
                market_error = str(market_meta.get("error", "")).strip() or "exchangeInfo unavailable"
                failure_detail = f"endpoint call failed: {market_error}"
                if market_method:
                    failure_detail = f"{failure_detail} (method={market_method})"
                raise RuntimeError(failure_detail)
            success_detail = f"endpoint call ok via {market_method}" if market_method else ""
            _add_endpoint(
                endpoint_name="symbol_filters",
                ok=True,
                endpoint_hint=market_endpoint_hint,
                detail=success_detail,
                http_status=market_http_status,
            )
            if "_markets" in exchange_info:
                markets = exchange_info.get("_markets")
                market = markets.get(symbol) if isinstance(markets, dict) else None
                if not isinstance(market, dict):
                    raise RuntimeError(f"symbol not found: {symbol}")
                if market.get("active") is False:
                    raise RuntimeError(f"symbol inactive: {symbol}")
                filters = self._symbol_filter_snapshot(market)
            else:
                filters = self._symbol_filter_snapshot_from_exchange_info(exchange_info, symbol)
            market_detail = (
                f"tick_size={filters['tick_size']} step_size={filters['step_size']} min_notional={filters['min_notional']}"
            )
        except Exception as exc:
            market_ok = False
            market_detail = f"symbol/filter check failed: {exc}"
            _add_endpoint(
                endpoint_name="symbol_filters",
                ok=False,
                endpoint_hint=market_endpoint_hint,
                detail=f"endpoint call failed: {exc}",
                http_status=market_http_status,
                error_code=market_error_code,
            )
        _add("symbol_filters", market_ok, market_detail)

        if include_leverage_margin:
            lev_margin_required = False
            lev_margin_ok = True
            lev_margin_detail = "leverage/margin check skipped (endpoint unavailable)"
            risk, risk_meta = self._fetch_position_risk(symbol)
            if bool(risk_meta.get("attempted", False)):
                _add(
                    "endpoint_call",
                    bool(risk_meta.get("ok", False)),
                    str(risk_meta.get("detail", "position risk endpoint call finished")),
                    required=False,
                    event_type="preflight_endpoint",
                    extra={
                        "endpoint_name": "leverage_margin_mode",
                        "endpoint": str(risk_meta.get("endpoint", "GET /fapi/v2/positionRisk")),
                        "http_status": risk_meta.get("http_status", "unknown"),
                        "method": risk_meta.get("method"),
                        "error_code": risk_meta.get("error_code"),
                    },
                )
            if risk is not None:
                lev_margin_required = True
                live_leverage = self._as_float(risk.get("leverage"))
                live_margin = str(risk.get("marginType", "")).lower()
                details = [f"live_leverage={live_leverage}", f"live_margin={live_margin or '-'}"]
                if expected_leverage is not None and expected_leverage > 0:
                    if abs(live_leverage - expected_leverage) > 1e-9:
                        lev_margin_ok = False
                        details.append(f"expected_leverage={expected_leverage}")
                if expected_margin_mode:
                    expected = expected_margin_mode.lower()
                    if live_margin and live_margin != expected:
                        lev_margin_ok = False
                        details.append(f"expected_margin={expected}")
                lev_margin_detail = ", ".join(details)
            _add("leverage_margin_mode", lev_margin_ok, lev_margin_detail, required=lev_margin_required)
        else:
            _add(
                "leverage_margin_mode",
                True,
                "leverage/margin check skipped by caller option",
                required=False,
            )

        required_failed = any((not bool(item["ok"])) and bool(item["required"]) for item in checks)
        return (not required_failed), checks

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
            raise ValueError(f"Unsupported order_type: {order_type}")
        return up

    def _to_exchange_order_type(self, order_type: str) -> str:
        if order_type == "MARKET":
            return "market"
        if order_type == "LIMIT":
            return "limit"
        return order_type

    def _market_symbol_key(self, symbol: str) -> str:
        return symbol.replace("/", "").upper()

    def _retry_create_order(
        self,
        *,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: float | None,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        attempt = 0
        while True:
            attempt += 1
            try:
                return self.exchange.create_order(
                    symbol=symbol,
                    type=order_type,
                    side=side.lower(),
                    amount=amount,
                    price=price,
                    params=params,
                )
            except (ccxt.NetworkError, ccxt.RequestTimeout) as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(f"create_order failed after retries: {exc}") from exc
                time.sleep(self.retry_delay_sec)

    def _retry_fetch_order(self, *, order_id: str, symbol: str) -> dict[str, Any] | None:
        attempt = 0
        while True:
            attempt += 1
            try:
                fetched = self.exchange.fetch_order(order_id, symbol=symbol)
                return fetched if isinstance(fetched, dict) else None
            except (ccxt.NetworkError, ccxt.RequestTimeout):
                if attempt >= self.max_retries:
                    return None
                time.sleep(self.retry_delay_sec)
            except Exception:
                return None

    def _retry_fetch_ticker_price(self, symbol: str) -> float | None:
        attempt = 0
        while True:
            attempt += 1
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                last = ticker.get("last")
                if last is not None:
                    return float(last)
                close = ticker.get("close")
                if close is not None:
                    return float(close)
                return None
            except (ccxt.NetworkError, ccxt.RequestTimeout):
                if attempt >= self.max_retries:
                    return None
                time.sleep(self.retry_delay_sec)
            except Exception:
                return None

    def _retry_fetch_positions(self, symbol: str) -> list[dict[str, Any]]:
        attempt = 0
        while True:
            attempt += 1
            try:
                fetch_positions = getattr(self.exchange, "fetch_positions", None)
                if not callable(fetch_positions):
                    return []
                rows = fetch_positions([symbol])
                return rows if isinstance(rows, list) else []
            except (ccxt.NetworkError, ccxt.RequestTimeout):
                if attempt >= self.max_retries:
                    return []
                time.sleep(self.retry_delay_sec)
            except Exception:
                return []

    def _round_price(self, symbol: str, price: float) -> float:
        try:
            if hasattr(self.exchange, "price_to_precision"):
                return float(self.exchange.price_to_precision(symbol, price))
        except Exception:
            pass
        return round(price, 8)

    def _round_amount(self, symbol: str, amount: float) -> float:
        try:
            if hasattr(self.exchange, "amount_to_precision"):
                return float(self.exchange.amount_to_precision(symbol, amount))
        except Exception:
            pass
        return round(amount, 8)

    def attach_storage(self, *, storage: Any, run_id: str) -> None:
        self._storage = storage
        self._storage_run_id = run_id

    def _write_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if self._storage is None:
            return
        row = {"run_id": self._storage_run_id, **payload}
        self._storage.write_event(datetime.now(timezone.utc).isoformat(), event_type, row)

    def _map_status(self, raw_status: str) -> str:
        status_map = {
            "NEW": "NEW",
            "OPEN": "NEW",
            "PARTIALLY_FILLED": "NEW",
            "FILLED": "FILLED",
            "CANCELED": "CANCELED",
            "CANCELLED": "CANCELED",
            "REJECTED": "REJECTED",
            "EXPIRED": "CANCELED",
        }
        return status_map.get(raw_status.upper(), "NEW")

    def _as_float(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _result(
        self,
        *,
        order_id: str,
        status: str,
        filled_qty: float = 0.0,
        avg_price: float = 0.0,
        fee: float = 0.0,
        message: str = "",
        client_order_id: str | None = None,
    ) -> OrderResult:
        return OrderResult(
            order_id=order_id,
            status=self._map_status(status),  # type: ignore[arg-type]
            filled_qty=filled_qty,
            avg_price=avg_price,
            fee=fee,
            message=message,
            client_order_id=client_order_id,
        )

    def _save_ws_order(self, payload: dict[str, Any]) -> None:
        if self._storage is None or self._storage_run_id is None:
            return
        order_data = payload.get("o")
        if not isinstance(order_data, dict):
            return
        side = self._normalize_side(str(order_data.get("S", "BUY")))
        status = self._map_status(str(order_data.get("X", "NEW")))
        order_id = str(order_data.get("i", ""))
        if not order_id:
            return
        self._storage.save_order(
            {
                "run_id": self._storage_run_id,
                "order_id": order_id,
                "client_order_id": order_data.get("c"),
                "ts": str(pd_timestamp_from_ms(order_data.get("T") or payload.get("E") or 0)),
                "signal": "ws_update",
                "side": side,
                "position_side": str(order_data.get("ps", "BOTH")),
                "reduce_only": bool(order_data.get("R", False)),
                "order_type": str(order_data.get("o", "MARKET")),
                "qty": self._as_float(order_data.get("q", 0.0)),
                "requested_price": self._as_float(order_data.get("p", 0.0)),
                "stop_price": self._as_float(order_data.get("sp", 0.0)),
                "time_in_force": order_data.get("f"),
                "status": status.lower(),
                "reason": str(order_data.get("x", "ORDER_TRADE_UPDATE")),
            }
        )

    def _save_ws_fill(
        self,
        *,
        payload: dict[str, Any],
        order_id: str,
        side: str,
        fill_qty: float,
        fill_price: float,
        fee: float,
        trade_key: str,
        liquidity: str,
    ) -> None:
        if self._storage is None or self._storage_run_id is None:
            return
        event_order = payload.get("o")
        event_time = None
        if isinstance(event_order, dict):
            event_time = event_order.get("T")
        self._storage.save_fill(
            {
                "run_id": self._storage_run_id,
                "fill_id": f"{self._storage_run_id}-{order_id}-{trade_key}",
                "order_id": order_id,
                "ts": str(pd_timestamp_from_ms(event_time or payload.get("E") or 0)),
                "side": side,
                "qty": fill_qty,
                "price": fill_price,
                "fee": fee,
                "liquidity": liquidity,
            }
        )

    def _handle_order_trade_update(self, payload: dict[str, Any]) -> None:
        order_data = payload.get("o")
        if not isinstance(order_data, dict):
            return
        client_order_id = str(order_data.get("c")) if order_data.get("c") else None
        order_id = str(order_data.get("i", ""))
        if not order_id:
            return
        side = self._normalize_side(str(order_data.get("S", "BUY")))
        raw_status = str(order_data.get("X", "NEW"))
        status = self._map_status(raw_status)
        filled_qty = self._as_float(order_data.get("z", 0.0))
        avg_price = self._as_float(order_data.get("ap", 0.0))
        fee = self._as_float(order_data.get("n", 0.0))
        result = self._result(
            order_id=order_id,
            status=status,
            filled_qty=filled_qty,
            avg_price=avg_price,
            fee=fee,
            message="user stream update",
            client_order_id=client_order_id,
        )

        with self._lock:
            cached = {
                "order_id": order_id,
                "client_order_id": client_order_id,
                "status": status,
                "filled_qty": filled_qty,
                "avg_price": avg_price,
                "fee": fee,
                "side": side,
                "order_type": str(order_data.get("o", "MARKET")).upper(),
            }
            self._orders_by_id[order_id] = cached
            if client_order_id:
                self._client_results[client_order_id] = result
                if status in TERMINAL_STATUSES:
                    self._open_orders.pop(client_order_id, None)
                else:
                    self._open_orders[client_order_id] = cached

        self._save_ws_order(payload)
        exec_type = str(order_data.get("x", "")).upper()
        fill_qty = self._as_float(order_data.get("l", 0.0))
        if exec_type != "TRADE" or fill_qty <= 0.0:
            return

        fill_price = self._as_float(order_data.get("L", 0.0))
        trade_id = order_data.get("t")
        trade_key = str(trade_id) if trade_id is not None and str(trade_id) != "-1" else ""
        if not trade_key:
            trade_time = order_data.get("T") or payload.get("E") or 0
            trade_key = f"{trade_time}:{fill_qty:.12f}:{fill_price:.12f}:{side}"

        with self._lock:
            if trade_key in self._seen_fill_keys:
                return
            self._seen_fill_keys.add(trade_key)

        liquidity = "maker" if bool(order_data.get("m", False)) else "taker"
        self._save_ws_fill(
            payload=payload,
            order_id=order_id,
            side=side,
            fill_qty=fill_qty,
            fill_price=fill_price,
            fee=fee,
            trade_key=trade_key,
            liquidity=liquidity,
        )

    def _handle_account_update(self, payload: dict[str, Any]) -> None:
        account = payload.get("a")
        if not isinstance(account, dict):
            return
        positions = account.get("P")
        if not isinstance(positions, list):
            return
        with self._lock:
            for pos in positions:
                if not isinstance(pos, dict):
                    continue
                symbol = str(pos.get("s", ""))
                if not symbol:
                    continue
                qty = self._as_float(pos.get("pa", 0.0))
                self._positions[symbol] = {
                    "qty": qty,
                    "entry_price": self._as_float(pos.get("ep", 0.0)),
                    "unrealized_pnl": self._as_float(pos.get("up", 0.0)),
                }

    def handle_user_stream_event(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._last_user_stream_update = time.time()
        event_type = str(payload.get("e", "UNKNOWN"))
        if event_type == "ORDER_TRADE_UPDATE":
            self._handle_order_trade_update(payload)
            self._write_event("user_stream_order_trade_update", payload)
            return
        if event_type == "ACCOUNT_UPDATE":
            self._handle_account_update(payload)
            self._write_event("user_stream_account_update", payload)
            return
        self._write_event("user_stream_event", payload)

    def _read_cached_terminal_result(self, client_order_id: str, order_id: str) -> OrderResult | None:
        with self._lock:
            cached = self._client_results.get(client_order_id)
            if cached is not None and cached.status in TERMINAL_STATUSES:
                return cached
            by_id = self._orders_by_id.get(order_id)
            if by_id is None:
                return None
            status = str(by_id.get("status", "NEW"))
            if status not in TERMINAL_STATUSES:
                return None
            return self._result(
                order_id=order_id,
                status=status,
                filled_qty=self._as_float(by_id.get("filled_qty", 0.0)),
                avg_price=self._as_float(by_id.get("avg_price", 0.0)),
                fee=self._as_float(by_id.get("fee", 0.0)),
                message="user stream terminal",
                client_order_id=client_order_id,
            )

    def _wait_for_terminal_ws_status(self, *, client_order_id: str, order_id: str) -> OrderResult | None:
        deadline = time.monotonic() + self.ws_order_wait_sec
        while time.monotonic() < deadline:
            cached = self._read_cached_terminal_result(client_order_id, order_id)
            if cached is not None:
                return cached
            time.sleep(self.ws_poll_interval_sec)
        return None

    def _fallback_fetch_terminal(self, *, order_id: str, symbol: str, client_order_id: str | None) -> OrderResult | None:
        fetched = self._retry_fetch_order(order_id=order_id, symbol=symbol)
        if fetched is None:
            return None
        raw_status = str(fetched.get("status", "open")).upper()
        mapped = self._map_status(raw_status)
        if mapped not in TERMINAL_STATUSES:
            return None
        avg_price = self._as_float(fetched.get("average") or fetched.get("price") or 0.0)
        filled_qty = self._as_float(fetched.get("filled") or 0.0)
        fee_obj = fetched.get("fee") or {}
        fee_paid = self._as_float(fee_obj.get("cost") if isinstance(fee_obj, dict) else 0.0)
        return self._result(
            order_id=str(fetched.get("id") or order_id),
            status=mapped,
            filled_qty=filled_qty,
            avg_price=avg_price,
            fee=fee_paid,
            message="REST fallback order sync",
            client_order_id=client_order_id,
        )

    def _position_qty_from_cache_or_rest(self, symbol: str) -> float:
        sym_key = self._market_symbol_key(symbol)
        with self._lock:
            if sym_key in self._positions:
                return self._as_float(self._positions[sym_key].get("qty", 0.0))
            if symbol in self._positions:
                return self._as_float(self._positions[symbol].get("qty", 0.0))
        for row in self._retry_fetch_positions(symbol):
            if not isinstance(row, dict):
                continue
            info = row.get("info") if isinstance(row.get("info"), dict) else {}
            row_symbol = str(info.get("symbol") or row.get("symbol") or "").upper().replace("/", "")
            if row_symbol != sym_key:
                continue
            qty = self._as_float(info.get("positionAmt", row.get("contracts", 0.0)))
            with self._lock:
                self._positions[sym_key] = {"qty": qty, "entry_price": self._as_float(info.get("entryPrice", 0.0))}
            return qty
        return 0.0

    def _validate_trigger_price(self, *, side: str, order_type: str, stop_price: float, mark_price: float) -> str | None:
        if order_type == "STOP_MARKET":
            if side == "BUY" and stop_price <= mark_price:
                return "BUY STOP_MARKET requires stop_price above current price"
            if side == "SELL" and stop_price >= mark_price:
                return "SELL STOP_MARKET requires stop_price below current price"
        if order_type == "TAKE_PROFIT_MARKET":
            if side == "BUY" and stop_price >= mark_price:
                return "BUY TAKE_PROFIT_MARKET requires stop_price below current price"
            if side == "SELL" and stop_price <= mark_price:
                return "SELL TAKE_PROFIT_MARKET requires stop_price above current price"
        return None

    def _prepare_order(self, request: OrderRequest) -> tuple[OrderRequest, OrderResult | None]:
        side = self._normalize_side(request.side)
        order_type = self._normalize_order_type(request.order_type)
        amount = max(float(request.amount), 0.0)
        stop_price = float(request.stop_price) if request.stop_price is not None else None
        price = float(request.price) if request.price is not None else None

        if amount <= 0:
            return request, self._result(
                order_id=f"reject-{int(time.time() * 1000)}",
                status="REJECTED",
                message="amount must be positive",
                client_order_id=request.client_order_id,
            )
        if order_type in {"STOP_MARKET", "TAKE_PROFIT_MARKET"} and stop_price is None:
            return request, self._result(
                order_id=f"reject-{int(time.time() * 1000)}",
                status="REJECTED",
                message="stop_price required for trigger order",
                client_order_id=request.client_order_id,
            )

        mark_price = self._retry_fetch_ticker_price(request.symbol)
        if order_type in {"STOP_MARKET", "TAKE_PROFIT_MARKET"} and mark_price is None:
            return request, self._result(
                order_id=f"reject-{int(time.time() * 1000)}",
                status="REJECTED",
                message="failed to fetch current price for stop validation",
                client_order_id=request.client_order_id,
            )
        if mark_price is not None and stop_price is not None:
            err = self._validate_trigger_price(side=side, order_type=order_type, stop_price=stop_price, mark_price=mark_price)
            if err is not None:
                return request, self._result(
                    order_id=f"reject-{int(time.time() * 1000)}",
                    status="REJECTED",
                    message=err,
                    client_order_id=request.client_order_id,
                )

        if request.reduce_only:
            pos_qty = self._position_qty_from_cache_or_rest(request.symbol)
            if side == "BUY" and pos_qty >= 0:
                return request, self._result(
                    order_id=f"reject-{int(time.time() * 1000)}",
                    status="REJECTED",
                    message="reduce_only BUY requires short position",
                    client_order_id=request.client_order_id,
                )
            if side == "SELL" and pos_qty <= 0:
                return request, self._result(
                    order_id=f"reject-{int(time.time() * 1000)}",
                    status="REJECTED",
                    message="reduce_only SELL requires long position",
                    client_order_id=request.client_order_id,
                )
            amount = min(amount, abs(pos_qty))
            if amount <= 0:
                return request, self._result(
                    order_id=f"reject-{int(time.time() * 1000)}",
                    status="REJECTED",
                    message="reduce_only clamped quantity is zero",
                    client_order_id=request.client_order_id,
                )

        rounded_amount = self._round_amount(request.symbol, amount)
        rounded_price = self._round_price(request.symbol, price) if price is not None else None
        rounded_stop = self._round_price(request.symbol, stop_price) if stop_price is not None else None
        prepared = OrderRequest(
            symbol=request.symbol,
            side=side,
            amount=rounded_amount,
            order_type=order_type,
            price=rounded_price,
            stop_price=rounded_stop,
            client_order_id=request.client_order_id,
            reduce_only=request.reduce_only,
            time_in_force=request.time_in_force,
            position_side=request.position_side,
        )
        return prepared, None

    def place_order(self, request: OrderRequest) -> OrderResult:
        if request.client_order_id and request.client_order_id in self._client_results:
            return self._client_results[request.client_order_id]

        if not self.live_trading:
            result = self._result(
                order_id=f"dryrun-{int(time.time() * 1000)}",
                status="REJECTED",
                message="LIVE_TRADING is disabled",
                client_order_id=request.client_order_id,
            )
            if request.client_order_id:
                self._client_results[request.client_order_id] = result
            return result

        prepared, reject_result = self._prepare_order(request)
        if reject_result is not None:
            if request.client_order_id:
                self._client_results[request.client_order_id] = reject_result
            return reject_result

        params: dict[str, Any] = {}
        if prepared.client_order_id:
            params["newClientOrderId"] = prepared.client_order_id
            params["clientOrderId"] = prepared.client_order_id
        if prepared.reduce_only:
            params["reduceOnly"] = True
        if prepared.stop_price is not None:
            params["stopPrice"] = prepared.stop_price
        if prepared.position_side:
            params["positionSide"] = prepared.position_side
        if prepared.time_in_force:
            params["timeInForce"] = prepared.time_in_force

        order = self._retry_create_order(
            symbol=prepared.symbol,
            order_type=self._to_exchange_order_type(self._normalize_order_type(prepared.order_type)),
            side=self._normalize_side(prepared.side),
            amount=prepared.amount,
            price=prepared.price,
            params=params,
        )
        avg_price = self._as_float(order.get("average") or order.get("price") or 0.0)
        filled_qty = self._as_float(order.get("filled") or 0.0)
        fee_obj = order.get("fee") or {}
        fee_paid = self._as_float(fee_obj.get("cost") if isinstance(fee_obj, dict) else 0.0)
        result = self._result(
            order_id=str(order.get("id", "unknown")),
            status=str(order.get("status", "open")),
            filled_qty=filled_qty,
            avg_price=avg_price,
            fee=fee_paid,
            message="Binance order submitted",
            client_order_id=prepared.client_order_id,
        )

        if self.use_user_stream and prepared.client_order_id:
            ws_result = self._wait_for_terminal_ws_status(
                client_order_id=prepared.client_order_id,
                order_id=result.order_id,
            )
            if ws_result is not None:
                result = ws_result
            else:
                fallback = self._fallback_fetch_terminal(
                    order_id=result.order_id,
                    symbol=prepared.symbol,
                    client_order_id=prepared.client_order_id,
                )
                if fallback is not None:
                    result = fallback
                else:
                    raise RuntimeError(
                        "No terminal order status from user stream and fallback REST check was inconclusive"
                    )

        if prepared.client_order_id:
            self._client_results[prepared.client_order_id] = result
        return result

    def cancel_order(self, order_id: str, *, symbol: str) -> bool:
        try:
            self.exchange.cancel_order(order_id, symbol=symbol)
            return True
        except Exception:
            return False

    def reconcile_runtime_state(
        self,
        *,
        symbol: str,
        open_positions: dict[str, Any],
        open_orders: dict[str, Any],
    ) -> tuple[bool, str]:
        expected_qty = self._as_float(open_positions.get("qty", 0.0))
        live_qty = self._position_qty_from_cache_or_rest(symbol)
        if abs(expected_qty - live_qty) > 1e-8:
            return False, f"position mismatch expected={expected_qty} live={live_qty}"
        try:
            fetch_open_orders = getattr(self.exchange, "fetch_open_orders", None)
            if callable(fetch_open_orders):
                live_open = fetch_open_orders(symbol)
                live_count = len(live_open) if isinstance(live_open, list) else 0
                if live_count != len(open_orders):
                    return False, f"open order mismatch expected={len(open_orders)} live={live_count}"
        except Exception as exc:
            return False, f"reconcile open orders failed: {exc}"
        return True, "ok"

    def restore_runtime_state(self, *, open_positions: dict[str, Any], open_orders: dict[str, Any]) -> None:
        qty = self._as_float(open_positions.get("qty", 0.0))
        symbol = str(open_positions.get("symbol", "BTCUSDT"))
        with self._lock:
            self._positions[self._market_symbol_key(symbol)] = {
                "qty": qty,
                "entry_price": self._as_float(open_positions.get("entry_price", 0.0)),
            }
            for payload in open_orders.values():
                if not isinstance(payload, dict):
                    continue
                cid = payload.get("client_order_id")
                if cid:
                    self._open_orders[str(cid)] = dict(payload)

    def get_balance(self) -> dict[str, float]:
        balance = self.exchange.fetch_balance()
        total = balance.get("total", {})
        return {k: float(v) for k, v in total.items() if isinstance(v, (int, float))}

    def get_state_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "open_orders": {k: dict(v) for k, v in self._open_orders.items()},
                "positions": {k: dict(v) for k, v in self._positions.items()},
                "last_user_stream_update": self._last_user_stream_update,
            }

    def close(self) -> None:
        if self._user_stream is not None:
            self._user_stream.stop()
        close_method = getattr(self.exchange, "close", None)
        if callable(close_method):
            close_method()


def pd_timestamp_from_ms(value: Any) -> pd.Timestamp:
    try:
        return pd.to_datetime(int(value), unit="ms", utc=True)
    except Exception:
        return pd.Timestamp(datetime.now(timezone.utc))

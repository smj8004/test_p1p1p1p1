from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import SecretStr
from typer.testing import CliRunner

from trader.broker.base import Broker, OrderRequest, OrderResult
from trader.broker.live_binance import LiveBinanceBroker
from trader.cli import app
from trader.config import AppConfig
from trader.data.binance_live import LiveBar
from trader.risk.guards import RiskGuard
from trader.runtime import RuntimeConfig, RuntimeEngine
from trader.storage import SQLiteStorage
from trader.strategy.base import Bar, Strategy, StrategyPosition


class FakeBinanceAuthError(Exception):
    def __init__(self, message: str, *, code: int, http_status: int) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status


class FakePreflightExchange:
    def __init__(self) -> None:
        self.urls = {"api": {"fapiPrivate": "https://fapi.binance.com/fapi/v1"}}
        self.last_request_url = ""
        self.last_response_headers: dict[str, str] = {}

    def fetch_balance(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.last_request_url = "https://fapi.binance.com/fapi/v2/balance"
        self.last_response_headers = {"status": "401"}
        raise FakeBinanceAuthError(
            'binance {"code":-2015,"msg":"Invalid API-key, IP, or permissions for action."}',
            code=-2015,
            http_status=401,
        )

    def fetch_time(self) -> int:
        self.last_request_url = "https://fapi.binance.com/fapi/v1/time"
        self.last_response_headers = {"status": "200"}
        return int(pd.Timestamp("2026-01-01T00:00:00Z").timestamp() * 1000)

    def load_markets(self) -> dict[str, Any]:
        self.last_request_url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
        self.last_response_headers = {"status": "200"}
        return {
            "BTC/USDT": {
                "active": True,
                "info": {
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.1"},
                        {"filterType": "LOT_SIZE", "stepSize": "0.001"},
                        {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
                    ]
                },
            }
        }

    def close(self) -> None:
        return


class FakeDirectFapiExchange:
    def __init__(self) -> None:
        self.urls = {"api": {"fapiPrivateV2": "https://fapi.binance.com/fapi/v2"}}
        self.last_request_url = ""
        self.last_response_headers: dict[str, str] = {}

    def fapiPrivateV2GetBalance(self) -> list[dict[str, str]]:
        self.last_request_url = "https://fapi.binance.com/fapi/v2/balance"
        self.last_response_headers = {"status": "200"}
        return [{"asset": "USDT", "balance": "1000"}]

    def fapiPublicGetExchangeInfo(self) -> dict[str, Any]:
        self.last_request_url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
        self.last_response_headers = {"status": "200"}
        return {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.1"},
                        {"filterType": "LOT_SIZE", "stepSize": "0.001"},
                        {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
                    ],
                }
            ]
        }

    def fetch_time(self) -> int:
        self.last_request_url = "https://fapi.binance.com/fapi/v1/time"
        self.last_response_headers = {"status": "200"}
        return int(pd.Timestamp("2026-01-01T00:00:00Z").timestamp() * 1000)

    def fetch_balance(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        raise AssertionError("fetch_balance fallback must not be used when fapi direct method is available")

    def load_markets(self) -> dict[str, Any]:
        raise AssertionError("load_markets fallback must not be used when fapi direct method is available")

    def close(self) -> None:
        return


def test_preflight_prefers_direct_fapi_endpoints_when_available() -> None:
    broker = LiveBinanceBroker(
        api_key="abc123",
        api_secret="secret987",
        testnet=False,
        live_trading=False,
        exchange=FakeDirectFapiExchange(),
        use_user_stream=False,
    )
    try:
        ok, checks = broker.preflight_check(
            symbol="BTC/USDT",
            max_time_drift_ms=10_000_000_000,
            include_leverage_margin=False,
        )
    finally:
        broker.close()

    assert ok is True
    endpoint_rows = [row for row in checks if row.get("event_type") == "preflight_endpoint"]
    futures_row = next(row for row in endpoint_rows if row.get("endpoint_name") == "futures_permission")
    market_row = next(row for row in endpoint_rows if row.get("endpoint_name") == "symbol_filters")
    assert "fapi.binance.com/fapi/v2/balance" in str(futures_row.get("endpoint", ""))
    assert "fapi.binance.com/fapi/v1/exchangeInfo" in str(market_row.get("endpoint", ""))
    assert futures_row.get("http_status") in {200, "2xx"}
    assert market_row.get("http_status") in {200, "2xx"}


def test_preflight_emits_env_credentials_endpoint_and_2015_guidance() -> None:
    broker = LiveBinanceBroker(
        api_key="abc123",
        api_secret="secret987",
        testnet=False,
        live_trading=False,
        exchange=FakePreflightExchange(),
        use_user_stream=False,
    )
    try:
        ok, checks = broker.preflight_check(
            symbol="BTC/USDT",
            max_time_drift_ms=10_000_000_000,
            include_leverage_margin=False,
        )
    finally:
        broker.close()

    assert ok is False
    env_row = next(row for row in checks if row.get("event_type") == "preflight_environment")
    assert env_row["binance_env"] == "mainnet"
    assert str(env_row["base_url"]).startswith("https://fapi.binance.com")
    assert str(env_row["ws_url"]).startswith("wss://fstream.binance.com")

    cred_row = next(row for row in checks if row.get("event_type") == "preflight_credentials")
    assert cred_row["api_key_present"] is True
    assert cred_row["api_secret_present"] is True
    assert cred_row["api_key_len"] == 6
    assert cred_row["api_secret_len"] == 9

    endpoint_rows = [row for row in checks if row.get("event_type") == "preflight_endpoint"]
    endpoint_names = {str(row.get("endpoint_name")) for row in endpoint_rows}
    assert {"futures_permission", "server_time_sync", "symbol_filters"}.issubset(endpoint_names)
    assert any(str(row.get("http_status")) == "401" for row in endpoint_rows if row.get("endpoint_name") == "futures_permission")

    guidance_row = next(row for row in checks if row.get("event_type") == "preflight_auth_guidance")
    guide = guidance_row.get("guide")
    assert isinstance(guide, list)
    assert any("testnet/mainnet" in str(item) for item in guide)
    assert any("IP whitelist" in str(item) for item in guide)
    assert any("Futures permission" in str(item) for item in guide)
    assert any("key/secret mismatch" in str(item) for item in guide)


class HoldStrategy(Strategy):
    def on_bar(self, bar: Bar, position: StrategyPosition | None = None) -> str:
        return "hold"


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


class EventTypeBroker(Broker):
    def preflight_check(self, **kwargs):  # type: ignore[no-untyped-def]
        return (
            True,
            [
                {
                    "event_type": "preflight_environment",
                    "check": "environment",
                    "ok": True,
                    "required": False,
                    "detail": "env row",
                }
            ],
        )

    def place_order(self, request: OrderRequest) -> OrderResult:
        return OrderResult(order_id="x", status="FILLED", filled_qty=request.amount, avg_price=100.0)

    def get_balance(self) -> dict[str, float]:
        return {"USDT": 1000.0}


def _risk_guard() -> RiskGuard:
    return RiskGuard(
        max_order_notional=1_000_000,
        max_position_notional=1_000_000,
        max_daily_loss=1_000_000,
        max_drawdown_pct=1.0,
        max_atr_pct=1.0,
    )


def test_runtime_uses_preflight_event_type_when_present(tmp_path: Path) -> None:
    storage = SQLiteStorage(tmp_path / "preflight_event_type.db")
    bar = LiveBar(
        timestamp=pd.Timestamp("2026-01-01T00:00:00Z"),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=1000.0,
    )
    engine = RuntimeEngine(
        config=RuntimeConfig(mode="live", symbol="BTC/USDT", timeframe="1m", one_shot=True),
        strategy=HoldStrategy(),
        broker=EventTypeBroker(),
        feed=StaticFeed([bar]),
        storage=storage,
        risk_guard=_risk_guard(),
    )
    try:
        result = engine.run()
        events = storage.list_recent_events_for_run(str(result["run_id"]), limit=20)
    finally:
        storage.close()
    assert any(row["event_type"] == "preflight_environment" for row in events)


def test_doctor_command_runs_preflight_without_leverage_check(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    cfg = AppConfig(
        symbol="BTC/USDT",
        binance_env="testnet",
        binance_testnet=True,
        binance_api_key=SecretStr("k12345"),
        binance_api_secret=SecretStr("s67890"),
    )
    monkeypatch.setattr(AppConfig, "from_env", classmethod(lambda cls: cfg))

    class FakeDoctorBroker:
        called: dict[str, Any] = {}

        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            self.called["testnet"] = kwargs.get("testnet")

        def preflight_check(self, **kwargs):  # type: ignore[no-untyped-def]
            self.called["include_leverage_margin"] = kwargs.get("include_leverage_margin")
            self.called["symbol"] = kwargs.get("symbol")
            return True, [
                {
                    "event_type": "preflight_environment",
                    "check": "environment",
                    "ok": True,
                    "required": False,
                    "binance_env": "mainnet",
                    "base_url": "https://fapi.binance.com/fapi/v1",
                    "ws_url": "wss://fstream.binance.com/ws",
                },
                {
                    "event_type": "preflight_credentials",
                    "check": "credentials_meta",
                    "ok": True,
                    "required": False,
                    "api_key_present": True,
                    "api_key_len": 6,
                    "api_secret_present": True,
                    "api_secret_len": 6,
                },
                {
                    "event_type": "preflight_endpoint",
                    "check": "endpoint_call",
                    "ok": True,
                    "required": False,
                    "endpoint_name": "server_time_sync",
                    "endpoint": "https://fapi.binance.com/fapi/v1/time",
                    "http_status": 200,
                    "detail": "ok",
                },
                {"event_type": "preflight_check", "check": "server_time_sync", "ok": True, "required": True, "detail": "ok"},
            ]

        def close(self) -> None:
            return

    monkeypatch.setattr("trader.cli.LiveBinanceBroker", FakeDoctorBroker)

    runner = CliRunner()
    result = runner.invoke(app, ["doctor", "--env", "mainnet"])
    assert result.exit_code == 0
    assert "No orders were sent" in result.stdout
    assert FakeDoctorBroker.called["testnet"] is False
    assert FakeDoctorBroker.called["include_leverage_margin"] is False
    assert FakeDoctorBroker.called["symbol"] == "BTC/USDT"

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import pandas as pd
from typer.testing import CliRunner

from trader.broker.base import Broker, OrderRequest, OrderResult
from trader.cli import app
from trader.risk.guards import RiskGuard
from trader.runtime import RuntimeConfig, RuntimeEngine
from trader.storage import SQLiteStorage
from trader.strategy.base import Bar, Strategy, StrategyPosition


class HoldStrategy(Strategy):
    def on_bar(self, bar: Bar, position: StrategyPosition | None = None) -> str:
        return "hold"


@dataclass
class StaticFeed:
    bars: list

    def iter_closed_bars(self, *, max_bars: int | None = None):  # type: ignore[override]
        emitted = 0
        for bar in self.bars:
            yield bar
            emitted += 1
            if max_bars is not None and emitted >= max_bars:
                return

    def close(self) -> None:
        return


@dataclass
class DelayedFeed:
    bars: list
    delays: list[float]

    def iter_closed_bars(self, *, max_bars: int | None = None):  # type: ignore[override]
        emitted = 0
        for idx, bar in enumerate(self.bars):
            if idx < len(self.delays) and self.delays[idx] > 0:
                time.sleep(self.delays[idx])
            yield bar
            emitted += 1
            if max_bars is not None and emitted >= max_bars:
                return

    def close(self) -> None:
        return


class PreflightFailBroker(Broker):
    def preflight_check(self, **kwargs):  # type: ignore[no-untyped-def]
        return False, [{"check": "credentials", "ok": False, "required": True, "detail": "missing key"}]

    def place_order(self, request: OrderRequest) -> OrderResult:
        return OrderResult(order_id="x", status="REJECTED", filled_qty=0.0, avg_price=0.0, message="blocked")

    def get_balance(self) -> dict[str, float]:
        return {"USDT": 0.0}


class PreflightPassBroker(Broker):
    def preflight_check(self, **kwargs):  # type: ignore[no-untyped-def]
        return True, [{"check": "ok", "ok": True, "required": True, "detail": "ok"}]

    def place_order(self, request: OrderRequest) -> OrderResult:
        return OrderResult(order_id="x", status="FILLED", filled_qty=request.amount, avg_price=100.0, fee=0.0)

    def get_balance(self) -> dict[str, float]:
        return {"USDT": 1000.0}


def _bar(ts: str, price: float = 100.0, *, is_backfill: bool = False):  # type: ignore[no-untyped-def]
    from trader.data.binance_live import LiveBar

    return LiveBar(
        timestamp=pd.Timestamp(ts),
        open=price,
        high=price + 1.0,
        low=price - 1.0,
        close=price,
        volume=1000.0,
        is_backfill=is_backfill,
    )


def _risk_guard() -> RiskGuard:
    return RiskGuard(
        max_order_notional=1_000_000,
        max_position_notional=1_000_000,
        max_daily_loss=1_000_000,
        max_drawdown_pct=1.0,
        max_atr_pct=1.0,
    )


def test_preflight_failure_halts_runtime(tmp_path: Path) -> None:
    storage = SQLiteStorage(tmp_path / "preflight.db")
    engine = RuntimeEngine(
        config=RuntimeConfig(mode="live", symbol="BTC/USDT", timeframe="1m"),
        strategy=HoldStrategy(),
        broker=PreflightFailBroker(),
        feed=StaticFeed([_bar("2025-01-01T00:00:00Z")]),
        storage=storage,
        risk_guard=_risk_guard(),
    )
    try:
        result = engine.run()
        events = storage.list_recent_events_for_run(str(result["run_id"]), limit=20)
    finally:
        storage.close()
    assert result["halted"] is True
    assert any(row["event_type"] == "preflight_failed" for row in events)


def test_feed_stall_detection_halts_runtime(tmp_path: Path) -> None:
    storage = SQLiteStorage(tmp_path / "stall.db")
    engine = RuntimeEngine(
        config=RuntimeConfig(
            mode="live",
            symbol="BTC/USDT",
            timeframe="1m",
            feed_stall_timeout_sec=0.05,
            max_bars=2,
        ),
        strategy=HoldStrategy(),
        broker=PreflightPassBroker(),
        feed=DelayedFeed(
            bars=[
                _bar("2020-01-01T00:00:00Z"),
                _bar("2020-01-01T00:01:00Z"),
            ],
            delays=[0.0, 0.12],
        ),
        storage=storage,
        risk_guard=_risk_guard(),
    )
    try:
        result = engine.run()
        events = storage.list_recent_events_for_run(str(result["run_id"]), limit=20)
    finally:
        storage.close()
    assert result["halted"] is True
    assert any(row["event_type"] == "feed_stall_detected" for row in events)


def test_status_command_prints_run_snapshot(tmp_path: Path) -> None:
    db_path = tmp_path / "status.db"
    storage = SQLiteStorage(db_path)
    run_id = "run-status-1"
    storage.save_runtime_state(
        run_id=run_id,
        last_bar_ts="2025-01-01T00:00:00Z",
        open_positions={"symbol": "BTC/USDT", "qty": 0.1, "entry_price": 100.0},
        open_orders={},
        strategy_state={},
        risk_state={"halted": False, "halt_reason": ""},
        updated_at="2025-01-01T00:01:00Z",
    )
    storage.write_event("2025-01-01T00:01:01Z", "runtime_started", {"run_id": run_id, "mode": "paper"})
    storage.save_trade(
        {
            "run_id": run_id,
            "trade_id": "t1",
            "symbol": "BTC/USDT",
            "side": "long",
            "entry_ts": "2025-01-01T00:00:00Z",
            "exit_ts": "2025-01-01T00:10:00Z",
            "qty": 0.1,
            "entry_price": 100.0,
            "exit_price": 101.0,
            "gross_pnl": 0.1,
            "fee_paid": 0.01,
            "funding_paid": 0.0,
            "net_pnl": 0.09,
            "return_pct": 0.009,
            "reason": "test",
        }
    )
    storage.close()

    runner = CliRunner()
    result = runner.invoke(app, ["status", "--run-id", run_id, "--db-path", str(db_path)])
    assert result.exit_code == 0
    assert run_id in result.stdout
    assert "Runtime Status" in result.stdout

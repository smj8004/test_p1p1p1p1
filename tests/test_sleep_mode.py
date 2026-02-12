from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import pandas as pd

from trader.broker.base import Broker, OrderRequest, OrderResult
from trader.config import AppConfig
from trader.data.binance_live import LiveBar
from trader.risk.guards import RiskGuard
from trader.runtime import RuntimeConfig, RuntimeEngine
from trader.storage import SQLiteStorage
from trader.strategy.base import Bar, Strategy, StrategyPosition


class LongStrategy(Strategy):
    def on_bar(self, bar: Bar, position: StrategyPosition | None = None) -> str:
        return "long"


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


@dataclass
class DelayedFeed:
    bars: list[LiveBar]
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


class CountingBroker(Broker):
    def __init__(self) -> None:
        self.calls = 0

    def place_order(self, request: OrderRequest) -> OrderResult:
        self.calls += 1
        return OrderResult(
            order_id=f"ord-{self.calls}",
            status="FILLED",
            filled_qty=request.amount,
            avg_price=100.0,
            fee=0.0,
            client_order_id=request.client_order_id,
        )

    def get_balance(self) -> dict[str, float]:
        return {"USDT": 1000.0}


class ErrorBroker(Broker):
    def place_order(self, request: OrderRequest) -> OrderResult:
        raise RuntimeError("api boom")

    def get_balance(self) -> dict[str, float]:
        return {"USDT": 1000.0}


def _bar(ts: str, price: float = 100.0, *, is_backfill: bool = False) -> LiveBar:
    return LiveBar(
        timestamp=pd.Timestamp(ts),
        open=price,
        high=price + 1.0,
        low=price - 1.0,
        close=price,
        volume=1000.0,
        is_backfill=is_backfill,
    )


def _risk_guard(**kwargs) -> RiskGuard:  # type: ignore[no-untyped-def]
    base = dict(
        max_order_notional=1_000_000,
        max_position_notional=1_000_000,
        max_daily_loss=1_000_000,
        max_drawdown_pct=1.0,
        max_atr_pct=1.0,
        account_allocation_pct=1.0,
        risk_per_trade_pct=0.0,
        daily_loss_limit_pct=0.0,
    )
    base.update(kwargs)
    return RiskGuard(**base)


def test_preset_loading_and_env_override(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("LEVERAGE", "1.5")
    cfg = AppConfig.from_env(preset="sleep_mode")
    assert cfg.preset_name is not None
    assert "sleep_mode" in cfg.preset_name
    assert abs(cfg.account_allocation_pct - 0.15) < 1e-12
    assert abs(cfg.leverage - 1.5) < 1e-12


def test_budget_clamp_respects_allocation_and_position_cap() -> None:
    guard = _risk_guard(account_allocation_pct=0.1, max_position_notional=1200)
    allowed, reason = guard.suggest_entry_notional(
        equity=10_000.0,
        current_position_notional=0.0,
        requested_order_notional=5_000.0,
        realized_pnl_today=0.0,
    )
    assert abs(allowed - 1000.0) < 1e-9
    assert "clamped" in reason


def test_quiet_hours_blocks_new_entries(tmp_path: Path) -> None:
    db_path = tmp_path / "quiet.sqlite"
    storage = SQLiteStorage(db_path)
    broker = CountingBroker()
    engine = RuntimeEngine(
        config=RuntimeConfig(
            mode="paper",
            symbol="BTC/USDT",
            timeframe="1m",
            one_shot=True,
            quiet_hours="00:00-23:59 UTC",
        ),
        strategy=LongStrategy(),
        broker=broker,
        feed=StaticFeed([_bar("2026-01-01T12:00:00Z")]),
        storage=storage,
        risk_guard=_risk_guard(quiet_hours="00:00-23:59 UTC"),
    )
    try:
        result = engine.run()
        events = storage.list_recent_events_for_run(str(result["run_id"]), limit=20)
    finally:
        storage.close()
    assert result["halted"] is False
    assert broker.calls == 0
    assert any(row["event_type"] == "quiet_hours_entry_blocked" for row in events)


def test_protective_missing_halt_mode(tmp_path: Path) -> None:
    db_path = tmp_path / "protective.sqlite"
    storage = SQLiteStorage(db_path)
    broker = CountingBroker()
    engine = RuntimeEngine(
        config=RuntimeConfig(
            mode="paper",
            symbol="BTC/USDT",
            timeframe="1m",
            max_bars=1,
            one_shot=True,
            enable_protective_orders=True,
            require_protective_orders=True,
            protective_missing_policy="halt",
            protective_stop_loss_pct=0.01,
        ),
        strategy=HoldStrategy(),
        broker=broker,
        feed=StaticFeed([_bar("2026-01-01T00:00:00Z")]),
        storage=storage,
        risk_guard=_risk_guard(),
    )
    engine.position_qty = 1.0
    engine.position_entry_price = 100.0
    engine.position_entry_ts = "2026-01-01T00:00:00Z"
    try:
        result = engine.run()
        events = storage.list_recent_events_for_run(str(result["run_id"]), limit=20)
    finally:
        storage.close()
    assert result["halted"] is True
    assert any(row["event_type"] == "protective_orders_halt" for row in events)


def test_feed_stall_halts_runtime(tmp_path: Path) -> None:
    db_path = tmp_path / "stall.sqlite"
    storage = SQLiteStorage(db_path)
    engine = RuntimeEngine(
        config=RuntimeConfig(
            mode="paper",
            symbol="BTC/USDT",
            timeframe="1m",
            feed_stall_timeout_sec=0.05,
            max_bars=2,
        ),
        strategy=HoldStrategy(),
        broker=CountingBroker(),
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


def test_backfill_like_old_bars_do_not_trigger_recv_stall(tmp_path: Path) -> None:
    db_path = tmp_path / "backfill_no_stall.sqlite"
    storage = SQLiteStorage(db_path)
    bars = [_bar(f"2020-01-01T00:{i:02d}:00Z", is_backfill=True) for i in range(30)]
    engine = RuntimeEngine(
        config=RuntimeConfig(
            mode="paper",
            symbol="BTC/USDT",
            timeframe="1m",
            feed_stall_timeout_sec=0.05,
            max_bars=30,
        ),
        strategy=HoldStrategy(),
        broker=CountingBroker(),
        feed=StaticFeed(bars),
        storage=storage,
        risk_guard=_risk_guard(),
    )
    try:
        result = engine.run()
        events = storage.list_recent_events_for_run(str(result["run_id"]), limit=50)
    finally:
        storage.close()
    assert result["halted"] is False
    assert result["processed_bars"] == 30
    assert not any(row["event_type"] == "feed_stall_detected" for row in events)


def test_api_error_threshold_halts_runtime(tmp_path: Path) -> None:
    db_path = tmp_path / "api_halt.sqlite"
    storage = SQLiteStorage(db_path)
    engine = RuntimeEngine(
        config=RuntimeConfig(
            mode="live",
            symbol="BTC/USDT",
            timeframe="1m",
            one_shot=True,
            api_error_halt_threshold=1,
        ),
        strategy=LongStrategy(),
        broker=ErrorBroker(),
        feed=StaticFeed([_bar("2026-01-01T00:00:00Z")]),
        storage=storage,
        risk_guard=_risk_guard(),
    )
    try:
        result = engine.run()
        events = storage.list_recent_events_for_run(str(result["run_id"]), limit=30)
    finally:
        storage.close()
    assert result["halted"] is True
    assert any(row["event_type"] == "api_error_halt" for row in events)

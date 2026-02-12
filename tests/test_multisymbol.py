from __future__ import annotations

import sqlite3
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pandas as pd

from trader.broker.paper import PaperBroker
from trader.data.binance_live import BinanceLiveFeed, LiveBar
from trader.risk.guards import RiskGuard
from trader.runtime import RuntimeConfig, RuntimeEngine, RuntimeOrchestrator
from trader.storage import SQLiteStorage
from trader.strategy.base import Bar, Strategy, StrategyPosition


class LongOnceStrategy(Strategy):
    def __init__(self) -> None:
        self.calls = 0

    def on_bar(self, bar: Bar, position: StrategyPosition | None = None) -> str:
        self.calls += 1
        return "long" if self.calls == 1 else "hold"


class HoldStrategy(Strategy):
    def __init__(self) -> None:
        self.calls = 0

    def on_bar(self, bar: Bar, position: StrategyPosition | None = None) -> str:
        self.calls += 1
        return "hold"


class SequenceStrategy(Strategy):
    def __init__(self, signals: list[str]) -> None:
        self.signals = signals
        self.calls = 0

    def on_bar(self, bar: Bar, position: StrategyPosition | None = None) -> str:
        self.calls += 1
        idx = self.calls - 1
        if idx < len(self.signals):
            return self.signals[idx]
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


def _bar(symbol: str, ts: str, price: float = 100.0) -> LiveBar:
    return LiveBar(
        timestamp=pd.Timestamp(ts),
        symbol=symbol,
        open=price,
        high=price + 1.0,
        low=price - 1.0,
        close=price,
        volume=1000.0,
    )


def _risk_guard() -> RiskGuard:
    return RiskGuard(
        max_order_notional=1_000_000,
        max_position_notional=1_000_000,
        max_daily_loss=1_000_000,
        max_drawdown_pct=1.0,
        max_atr_pct=1.0,
    )


def test_multisymbol_orders_and_state_are_isolated(tmp_path: Path) -> None:
    db_path = tmp_path / "multisymbol.sqlite"
    storage = SQLiteStorage(db_path)
    broker = PaperBroker(starting_cash=10_000.0)
    run_id = uuid4().hex

    btc_strategy = LongOnceStrategy()
    eth_strategy = HoldStrategy()

    btc_feed = StaticFeed([_bar("BTC/USDT", "2026-01-01T00:00:00Z", 100.0)])
    eth_feed = StaticFeed([_bar("ETH/USDT", "2026-01-01T00:00:00Z", 200.0)])

    cfg_btc = RuntimeConfig(mode="paper", symbol="BTC/USDT", timeframe="1m")
    cfg_eth = RuntimeConfig(mode="paper", symbol="ETH/USDT", timeframe="1m")

    btc_engine = RuntimeEngine(
        config=cfg_btc,
        strategy=btc_strategy,
        broker=broker,
        feed=btc_feed,  # unused by orchestrator worker dispatch
        storage=storage,
        risk_guard=_risk_guard(),
        initial_equity=10_000.0,
        run_id=run_id,
    )
    eth_engine = RuntimeEngine(
        config=cfg_eth,
        strategy=eth_strategy,
        broker=broker,
        feed=eth_feed,  # unused by orchestrator worker dispatch
        storage=storage,
        risk_guard=_risk_guard(),
        initial_equity=10_000.0,
        run_id=run_id,
    )

    orchestrator = RuntimeOrchestrator(
        engines={"BTC/USDT": btc_engine, "ETH/USDT": eth_engine},
        feeds={"BTC/USDT": btc_feed, "ETH/USDT": eth_feed},
        max_bars=1,
    )
    try:
        result = orchestrator.run()
        summary = storage.get_run_status(run_id)
    finally:
        storage.close()

    assert result["halted"] is False
    assert set(result["symbols"].keys()) == {"BTC/USDT", "ETH/USDT"}
    assert btc_strategy.calls == 1
    assert eth_strategy.calls == 1
    assert broker.get_position("BTC/USDT").qty > 0
    assert broker.get_position("ETH/USDT").qty == 0

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT symbol, COUNT(*) AS c FROM orders GROUP BY symbol").fetchall()
    finally:
        conn.close()
    counts = {str(sym): int(c) for sym, c in rows}
    assert counts.get("BTC/USDT", 0) >= 1
    assert counts.get("ETH/USDT", 0) == 0

    pos_state = summary.get("open_positions", {})
    assert isinstance(pos_state, dict)
    assert "BTC/USDT" in pos_state
    assert "ETH/USDT" in pos_state


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


def test_multisymbol_feed_dedupe_and_backfill_are_independent() -> None:
    btc_msgs = deque(
        [
            _kline_msg(1735689600000, 100.0, closed=True),
            _kline_msg(1735689600000, 101.0, closed=True),  # duplicate ts
            _kline_msg(1735689660000, 102.0, closed=True),
        ]
    )
    eth_msgs = deque(
        [
            _kline_msg(1735689600000, 200.0, closed=True),
            _kline_msg(1735689720000, 203.0, closed=True),  # missing 00:01
        ]
    )

    def eth_backfill(symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2025-01-01T00:01:00Z")],
                "open": [201.0],
                "high": [202.0],
                "low": [200.0],
                "close": [202.0],
                "volume": [100.0],
            }
        )

    btc_feed = BinanceLiveFeed(
        symbol="BTC/USDT",
        timeframe="1m",
        mode="websocket",
        fetcher=_empty_fetcher,
        range_fetcher=lambda s, t, a, b: _empty_fetcher(s, t, 0),
        ws_message_provider=lambda: iter(btc_msgs),
        bootstrap_history_bars=0,
    )
    eth_feed = BinanceLiveFeed(
        symbol="ETH/USDT",
        timeframe="1m",
        mode="websocket",
        fetcher=_empty_fetcher,
        range_fetcher=eth_backfill,
        ws_message_provider=lambda: iter(eth_msgs),
        bootstrap_history_bars=0,
    )

    btc_bars = list(btc_feed.iter_closed_bars(max_bars=3))
    eth_bars = list(eth_feed.iter_closed_bars(max_bars=3))

    assert [b.timestamp for b in btc_bars] == [
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2025-01-01T00:01:00Z"),
    ]
    assert [b.timestamp for b in eth_bars] == [
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2025-01-01T00:01:00Z"),
        pd.Timestamp("2025-01-01T00:02:00Z"),
    ]


def test_multisymbol_one_symbol_stop_does_not_stop_others(tmp_path: Path) -> None:
    db_path = tmp_path / "multisymbol_independent_stop.sqlite"
    storage = SQLiteStorage(db_path)
    broker = PaperBroker(starting_cash=10_000.0)
    run_id = uuid4().hex

    btc_engine = RuntimeEngine(
        config=RuntimeConfig(mode="paper", symbol="BTC/USDT", timeframe="1m", one_shot=True),
        strategy=HoldStrategy(),
        broker=broker,
        feed=StaticFeed(
            [
                _bar("BTC/USDT", "2026-01-01T00:00:00Z", 100.0),
                _bar("BTC/USDT", "2026-01-01T00:01:00Z", 101.0),
                _bar("BTC/USDT", "2026-01-01T00:02:00Z", 102.0),
            ]
        ),
        storage=storage,
        risk_guard=_risk_guard(),
        initial_equity=10_000.0,
        run_id=run_id,
    )
    eth_engine = RuntimeEngine(
        config=RuntimeConfig(mode="paper", symbol="ETH/USDT", timeframe="1m"),
        strategy=HoldStrategy(),
        broker=broker,
        feed=StaticFeed(
            [
                _bar("ETH/USDT", "2026-01-01T00:00:00Z", 200.0),
                _bar("ETH/USDT", "2026-01-01T00:01:00Z", 201.0),
                _bar("ETH/USDT", "2026-01-01T00:02:00Z", 202.0),
            ]
        ),
        storage=storage,
        risk_guard=_risk_guard(),
        initial_equity=10_000.0,
        run_id=run_id,
    )

    orchestrator = RuntimeOrchestrator(
        engines={"BTC/USDT": btc_engine, "ETH/USDT": eth_engine},
        feeds={"BTC/USDT": btc_engine.feed, "ETH/USDT": eth_engine.feed},
    )
    try:
        result = orchestrator.run()
    finally:
        storage.close()

    assert result["halted"] is False
    assert int(result["symbols"]["BTC/USDT"]["processed_bars"]) == 1
    assert int(result["symbols"]["ETH/USDT"]["processed_bars"]) == 3


def test_multisymbol_account_daily_loss_halts_all(tmp_path: Path) -> None:
    db_path = tmp_path / "multisymbol_account_halt.sqlite"
    storage = SQLiteStorage(db_path)
    broker = PaperBroker(starting_cash=10_000.0)
    run_id = uuid4().hex
    tight_account_guard = RiskGuard(
        max_order_notional=1_000_000,
        max_position_notional=1_000_000,
        max_daily_loss=50.0,
        max_drawdown_pct=1.0,
        max_atr_pct=1.0,
    )

    btc_engine = RuntimeEngine(
        config=RuntimeConfig(mode="paper", symbol="BTC/USDT", timeframe="1m", fixed_notional_usdt=1_000.0),
        strategy=SequenceStrategy(["long", "exit", "hold"]),
        broker=broker,
        feed=StaticFeed(
            [
                _bar("BTC/USDT", "2026-01-01T00:00:00Z", 100.0),
                _bar("BTC/USDT", "2026-01-01T00:01:00Z", 90.0),
                _bar("BTC/USDT", "2026-01-01T00:02:00Z", 91.0),
            ]
        ),
        storage=storage,
        risk_guard=_risk_guard(),
        initial_equity=10_000.0,
        run_id=run_id,
    )
    eth_engine = RuntimeEngine(
        config=RuntimeConfig(mode="paper", symbol="ETH/USDT", timeframe="1m"),
        strategy=HoldStrategy(),
        broker=broker,
        feed=StaticFeed(
            [
                _bar("ETH/USDT", "2026-01-01T00:00:00Z", 200.0),
                _bar("ETH/USDT", "2026-01-01T00:01:00Z", 201.0),
                _bar("ETH/USDT", "2026-01-01T00:02:00Z", 202.0),
                _bar("ETH/USDT", "2026-01-01T00:03:00Z", 203.0),
            ]
        ),
        storage=storage,
        risk_guard=_risk_guard(),
        initial_equity=10_000.0,
        run_id=run_id,
    )

    orchestrator = RuntimeOrchestrator(
        engines={"BTC/USDT": btc_engine, "ETH/USDT": eth_engine},
        feeds={"BTC/USDT": btc_engine.feed, "ETH/USDT": eth_engine.feed},
        account_risk_guard=tight_account_guard,
        account_initial_equity=10_000.0,
    )
    try:
        result = orchestrator.run()
    finally:
        storage.close()

    assert result["halted"] is True
    assert "daily loss" in str(result["halt_reason"]).lower()
    assert int(result["symbols"]["ETH/USDT"]["processed_bars"]) < 4

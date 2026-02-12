from __future__ import annotations

from collections import deque

import pandas as pd
import pytest

from trader.broker.base import OrderRequest
from trader.broker.paper import PaperBroker
from trader.data.binance_live import BinanceLiveFeed


def _frame(ts: list[str], prices: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(ts, utc=True),
            "open": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "volume": [1000.0 for _ in prices],
        }
    )


def test_paper_broker_applies_slippage_and_fee() -> None:
    broker = PaperBroker(starting_cash=10_000.0, slippage_bps=10.0, taker_fee_bps=10.0)
    broker.update_market_price("BTC/USDT", 100.0)

    buy = broker.place_order(
        OrderRequest(
            symbol="BTC/USDT",
            side="buy",
            amount=1.0,
            order_type="market",
            client_order_id="cid-1",
        )
    )
    assert buy.status == "FILLED"
    assert buy.avg_price == pytest.approx(100.1)
    assert buy.fee == pytest.approx(0.1001)

    broker.update_market_price("BTC/USDT", 100.0)
    sell = broker.place_order(
        OrderRequest(
            symbol="BTC/USDT",
            side="sell",
            amount=1.0,
            order_type="market",
            client_order_id="cid-2",
            reduce_only=True,
        )
    )
    assert sell.status == "FILLED"
    assert sell.avg_price == pytest.approx(99.9)
    assert sell.fee == pytest.approx(0.0999)
    assert broker.cash == pytest.approx(9999.6)


def test_paper_broker_idempotency_same_client_order_id_only_once() -> None:
    broker = PaperBroker(starting_cash=10_000.0)
    broker.update_market_price("BTC/USDT", 100.0)
    req = OrderRequest(
        symbol="BTC/USDT",
        side="buy",
        amount=1.0,
        order_type="market",
        client_order_id="dup-1",
    )
    first = broker.place_order(req)
    second = broker.place_order(req)
    assert first.order_id == second.order_id
    assert second.message.startswith("duplicate")
    assert broker.get_position("BTC/USDT").qty == pytest.approx(1.0)


def test_live_feed_dedupes_bars_by_timestamp() -> None:
    frames = deque(
        [
            _frame(
                ["2025-01-01T00:00:00Z", "2025-01-01T00:01:00Z", "2025-01-01T00:02:00Z"],
                [100.0, 101.0, 102.0],
            ),
            _frame(
                ["2025-01-01T00:01:00Z", "2025-01-01T00:02:00Z", "2025-01-01T00:03:00Z"],
                [101.0, 102.0, 103.0],
            ),
        ]
    )

    def fetcher(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        if frames:
            return frames.popleft()
        return _frame(
            ["2025-01-01T00:02:00Z", "2025-01-01T00:03:00Z", "2025-01-01T00:04:00Z"],
            [102.0, 103.0, 104.0],
        )

    feed = BinanceLiveFeed(
        symbol="BTC/USDT",
        timeframe="1m",
        mode="rest",
        poll_interval_sec=0.0,
        fetcher=fetcher,
    )
    bars = list(feed.iter_closed_bars(max_bars=3, history_limit=3))
    timestamps = [b.timestamp.isoformat() for b in bars]
    assert timestamps == [
        "2025-01-01T00:00:00+00:00",
        "2025-01-01T00:01:00+00:00",
        "2025-01-01T00:02:00+00:00",
    ]

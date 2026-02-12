from __future__ import annotations

from typing import Any

import ccxt
import pandas as pd


class BinanceDataClient:
    def __init__(self, *, testnet: bool = False) -> None:
        self.exchange = ccxt.binance(
            {
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            }
        )
        if testnet:
            self._configure_futures_testnet_urls()

    def _configure_futures_testnet_urls(self) -> None:
        base = "https://testnet.binancefuture.com"
        api = self.exchange.urls.get("api")
        if not isinstance(api, dict):
            return
        api["fapiPublic"] = f"{base}/fapi/v1"
        api["fapiPublicV2"] = f"{base}/fapi/v2"
        api["fapiPublicV3"] = f"{base}/fapi/v3"
        api["fapiPrivate"] = f"{base}/fapi/v1"
        api["fapiPrivateV2"] = f"{base}/fapi/v2"
        api["fapiPrivateV3"] = f"{base}/fapi/v3"
        api["fapiData"] = f"{base}/futures/data"

    def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1h", limit: int = 300
    ) -> pd.DataFrame:
        rows = self.exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
        frame = pd.DataFrame(
            rows,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        return frame

    def fetch_ohlcv_range(
        self,
        *,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
        limit: int = 1000,
    ) -> pd.DataFrame:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        else:
            end_ts = end_ts.tz_convert("UTC")
        start_ms = int(start_ts.timestamp() * 1000)
        end_ms = int(end_ts.timestamp() * 1000)
        timeframe_ms = int(self.exchange.parse_timeframe(timeframe) * 1000)
        since = start_ms
        rows: list[list[float]] = []

        while since < end_ms:
            chunk = self.exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, since=since, limit=limit)
            if not chunk:
                break
            rows.extend([r for r in chunk if r[0] < end_ms])
            last_ts = int(chunk[-1][0])
            next_since = last_ts + timeframe_ms
            if next_since <= since:
                break
            since = next_since
            if last_ts >= end_ms:
                break

        if not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        frame = frame.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        return frame

    def fetch_funding_rate(self, symbol: str) -> dict[str, Any]:
        if not self.exchange.has.get("fetchFundingRate"):
            return {}
        return self.exchange.fetch_funding_rate(symbol)

    def fetch_trading_fee(self, symbol: str) -> dict[str, Any]:
        if not self.exchange.has.get("fetchTradingFee"):
            return {}
        return self.exchange.fetch_trading_fee(symbol)

    def close(self) -> None:
        close_method = getattr(self.exchange, "close", None)
        if callable(close_method):
            close_method()

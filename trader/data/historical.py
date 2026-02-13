"""
Historical Data Downloader for Binance

Downloads and caches historical kline (candlestick) data from Binance API.
Supports incremental updates and efficient local storage.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterator

import pandas as pd
import requests

from trader.logging import get_logger

logger = get_logger(__name__)

# Binance API endpoints
BINANCE_API_URL = "https://api.binance.com/api/v3/klines"
BINANCE_TESTNET_API_URL = "https://testnet.binance.vision/api/v3/klines"

# Timeframe to milliseconds mapping
TIMEFRAME_MS = {
    "1m": 60 * 1000,
    "3m": 3 * 60 * 1000,
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "2h": 2 * 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "6h": 6 * 60 * 60 * 1000,
    "8h": 8 * 60 * 60 * 1000,
    "12h": 12 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
    "3d": 3 * 24 * 60 * 60 * 1000,
    "1w": 7 * 24 * 60 * 60 * 1000,
    "1M": 30 * 24 * 60 * 60 * 1000,
}


class HistoricalDataDownloader:
    """
    Downloads historical kline data from Binance API.

    Features:
    - Automatic pagination (Binance limits 1000 candles per request)
    - Local caching to avoid re-downloading
    - Incremental updates
    - Rate limiting to respect API limits
    """

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        cache_dir: Path | str = "data/historical",
        testnet: bool = False,
    ) -> None:
        self.symbol = symbol
        self.binance_symbol = symbol.replace("/", "")  # BTC/USDT -> BTCUSDT
        self.timeframe = timeframe
        self.cache_dir = Path(cache_dir)
        self.testnet = testnet
        self.api_url = BINANCE_TESTNET_API_URL if testnet else BINANCE_API_URL

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cache_file(self) -> Path:
        """Get cache file path for current symbol/timeframe."""
        return self.cache_dir / f"{self.binance_symbol}_{self.timeframe}.csv"

    def _fetch_klines(
        self,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 1000,
    ) -> list[list]:
        """Fetch klines from Binance API."""
        params = {
            "symbol": self.binance_symbol,
            "interval": self.timeframe,
            "limit": limit,
        }

        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        response = requests.get(self.api_url, params=params, timeout=30)
        response.raise_for_status()

        return response.json()

    def _klines_to_dataframe(self, klines: list[list]) -> pd.DataFrame:
        """Convert Binance klines to DataFrame."""
        if not klines:
            return pd.DataFrame()

        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])

        # Convert types
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = df[col].astype(float)

        df["trades"] = df["trades"].astype(int)

        # Keep only essential columns
        df = df[["open_time", "open", "high", "low", "close", "volume", "close_time"]]
        df = df.rename(columns={"open_time": "timestamp"})

        return df

    def download(
        self,
        days: int = 365,
        end_date: datetime | None = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Download historical data.

        Args:
            days: Number of days of data to download
            end_date: End date (default: now)
            show_progress: Show download progress

        Returns:
            DataFrame with OHLCV data
        """
        if end_date is None:
            end_date = datetime.now(timezone.utc)

        start_date = end_date - timedelta(days=days)

        start_ms = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)

        timeframe_ms = TIMEFRAME_MS.get(self.timeframe, 60000)
        expected_candles = (end_ms - start_ms) // timeframe_ms

        logger.info(f"Downloading {self.symbol} {self.timeframe} data...")
        logger.info(f"  Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"  Expected candles: ~{expected_candles:,}")

        all_klines = []
        current_start = start_ms
        request_count = 0

        while current_start < end_ms:
            try:
                klines = self._fetch_klines(
                    start_time=current_start,
                    end_time=end_ms,
                    limit=1000,
                )

                if not klines:
                    break

                all_klines.extend(klines)
                request_count += 1

                # Move to next batch
                last_close_time = klines[-1][6]  # close_time is at index 6
                current_start = last_close_time + 1

                if show_progress and request_count % 10 == 0:
                    progress = len(all_klines) / expected_candles * 100
                    logger.info(f"  Progress: {len(all_klines):,} candles ({progress:.1f}%)")

                # Rate limiting (Binance allows 1200 requests/min)
                time.sleep(0.05)

            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed: {e}, retrying in 5 seconds...")
                time.sleep(5)
                continue

        df = self._klines_to_dataframe(all_klines)

        if not df.empty:
            # Remove duplicates and sort
            df = df.drop_duplicates(subset=["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)

            logger.info(f"  Downloaded {len(df):,} candles")
            logger.info(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        return df

    def download_and_cache(
        self,
        days: int = 365,
        update_existing: bool = True,
    ) -> pd.DataFrame:
        """
        Download data and save to cache.

        Args:
            days: Number of days to download
            update_existing: If True, update existing cache with new data

        Returns:
            DataFrame with all cached data
        """
        existing_df = None

        # Load existing cache
        if self.cache_file.exists() and update_existing:
            logger.info(f"Loading existing cache: {self.cache_file}")
            existing_df = pd.read_csv(self.cache_file, parse_dates=["timestamp"])
            existing_df["timestamp"] = pd.to_datetime(existing_df["timestamp"], utc=True)
            logger.info(f"  Existing data: {len(existing_df):,} candles")

            # Calculate what we need to download
            last_timestamp = existing_df["timestamp"].max()
            now = datetime.now(timezone.utc)
            days_to_download = (now - last_timestamp).days + 1

            if days_to_download <= 0:
                logger.info("  Cache is up to date!")
                return existing_df

            logger.info(f"  Downloading {days_to_download} days of new data...")
            new_df = self.download(days=days_to_download)

            if not new_df.empty:
                # Merge with existing data
                df = pd.concat([existing_df, new_df], ignore_index=True)
                df = df.drop_duplicates(subset=["timestamp"])
                df = df.sort_values("timestamp").reset_index(drop=True)
            else:
                df = existing_df
        else:
            # Download all requested data
            df = self.download(days=days)

        # Save to cache
        if not df.empty:
            df.to_csv(self.cache_file, index=False)
            logger.info(f"Saved to cache: {self.cache_file}")
            logger.info(f"  Total candles: {len(df):,}")

        return df

    def load_cache(self) -> pd.DataFrame | None:
        """Load data from cache without downloading."""
        if not self.cache_file.exists():
            return None

        df = pd.read_csv(self.cache_file, parse_dates=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    def iter_bars(self, df: pd.DataFrame) -> Iterator[dict]:
        """Iterate over DataFrame rows as bar dictionaries."""
        for _, row in df.iterrows():
            yield {
                "timestamp": row["timestamp"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
            }


def download_historical_data(
    symbol: str = "BTC/USDT",
    timeframe: str = "1m",
    days: int = 365,
    cache_dir: str = "data/historical",
) -> pd.DataFrame:
    """
    Convenience function to download historical data.

    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        timeframe: Candlestick timeframe (e.g., "1m", "1h", "1d")
        days: Number of days to download
        cache_dir: Directory to cache data

    Returns:
        DataFrame with OHLCV data
    """
    downloader = HistoricalDataDownloader(
        symbol=symbol,
        timeframe=timeframe,
        cache_dir=Path(cache_dir),
    )

    return downloader.download_and_cache(days=days)

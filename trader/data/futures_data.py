"""
Binance USDT-M Futures Data Downloader

Comprehensive data collection for realistic futures backtesting:
- OHLCV (1m klines)
- Funding Rate (8h)
- Mark Price Klines
- Index Price Klines
- Exchange Info (symbol filters)
- Open Interest History
- Long/Short Ratio
- Aggregated Trades (optional, heavy)

All data uses FAPI endpoints (USDT-M Futures), not SAPI.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

import pandas as pd
import requests

from trader.logging import get_logger, setup_logging

logger = get_logger(__name__)

# Binance FAPI endpoints (USDT-M Futures)
FAPI_BASE_URL = "https://fapi.binance.com"
FAPI_TESTNET_URL = "https://testnet.binancefuture.com"

# Timeframe to milliseconds
TIMEFRAME_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


@dataclass
class FuturesDataConfig:
    """Configuration for futures data download."""
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    days: int = 365
    base_dir: Path = field(default_factory=lambda: Path("data/futures"))

    # Data types to download
    download_ohlcv: bool = True
    download_funding: bool = True
    download_mark_price: bool = True
    download_index_price: bool = True
    download_exchange_info: bool = True
    download_open_interest: bool = True
    download_long_short_ratio: bool = True
    download_agg_trades: bool = False  # Heavy, optional

    # Force re-download even if cache exists
    force_download: bool = False

    # Rate limiting (Binance limit: 1200 req/min, but be conservative)
    request_delay: float = 0.25  # 250ms between requests (safe)
    retry_delay: float = 10.0   # Wait longer on retry
    max_retries: int = 5


class FuturesDataDownloader:
    """
    Downloads and manages USDT-M Futures data from Binance FAPI.

    Data Structure:
    data/futures/
    ├── raw/                      # 원천 데이터 (API 응답 그대로)
    │   ├── BTCUSDT/
    │   │   ├── klines_1m.csv
    │   │   ├── funding_rate.csv
    │   │   ├── mark_price_1m.csv
    │   │   ├── index_price_1m.csv
    │   │   ├── open_interest_5m.csv
    │   │   ├── long_short_ratio_5m.csv
    │   │   └── agg_trades.csv (optional)
    │   └── ETHUSDT/
    │       └── ...
    ├── clean/                    # 정제 데이터 (결측 처리, UTC 정렬)
    │   ├── BTCUSDT/
    │   │   ├── ohlcv_1m.parquet
    │   │   ├── ohlcv_5m.parquet  (리샘플)
    │   │   ├── ohlcv_15m.parquet
    │   │   ├── ohlcv_1h.parquet
    │   │   └── funding_rate.parquet
    │   └── ...
    ├── meta/                     # 메타데이터
    │   ├── exchange_info.json
    │   ├── symbol_filters.json
    │   └── download_manifest.json
    └── features/ (optional)      # 파생 피처
    """

    def __init__(self, config: FuturesDataConfig | None = None) -> None:
        self.config = config or FuturesDataConfig()
        self.base_url = FAPI_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) BinanceTrader/1.0"
        })

        # Create directory structure
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Create directory structure."""
        for subdir in ["raw", "clean", "meta", "features"]:
            (self.config.base_dir / subdir).mkdir(parents=True, exist_ok=True)

        for symbol in self.config.symbols:
            (self.config.base_dir / "raw" / symbol).mkdir(parents=True, exist_ok=True)
            (self.config.base_dir / "clean" / symbol).mkdir(parents=True, exist_ok=True)

    def _request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Make API request with retry logic and adaptive backoff."""
        url = f"{self.base_url}{endpoint}"

        for attempt in range(self.config.max_retries):
            try:
                response = self.session.get(url, params=params, timeout=30)

                # Handle rate limiting (429)
                if response.status_code == 429:
                    wait_time = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Rate limited (429). Waiting {wait_time:.0f}s...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()

                # Rate limiting between requests
                time.sleep(self.config.request_delay)

                return response.json()

            except requests.exceptions.RequestException as e:
                wait_time = self.config.retry_delay * (2 ** attempt)
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    logger.info(f"Retrying in {wait_time:.0f}s...")
                    time.sleep(wait_time)
                else:
                    raise

        return None

    # =========================================================================
    # A. OHLCV Klines (1m)
    # =========================================================================

    def download_klines(
        self,
        symbol: str,
        interval: str = "1m",
        days: int | None = None,
    ) -> pd.DataFrame:
        """
        Download futures OHLCV klines.

        Uses /fapi/v1/klines endpoint.
        """
        days = days or self.config.days
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        interval_ms = TIMEFRAME_MS.get(interval, 60_000)
        expected_candles = (end_ms - start_ms) // interval_ms

        logger.info(f"[{symbol}] Downloading {interval} klines...")
        logger.info(f"  Period: {start_time.date()} to {end_time.date()}")
        logger.info(f"  Expected: ~{expected_candles:,} candles")

        all_klines = []
        current_start = start_ms
        request_count = 0

        while current_start < end_ms:
            klines = self._request("/fapi/v1/klines", {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_ms,
                "limit": 1500,  # Max for futures
            })

            if not klines:
                break

            all_klines.extend(klines)
            request_count += 1

            # Progress
            if request_count % 20 == 0:
                progress = len(all_klines) / expected_candles * 100
                logger.info(f"  Progress: {len(all_klines):,} candles ({progress:.1f}%)")

            # Move to next batch
            last_close_time = klines[-1][6]
            current_start = last_close_time + 1

        if not all_klines:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])

        # Type conversion
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

        for col in ["open", "high", "low", "close", "volume", "quote_volume",
                    "taker_buy_base", "taker_buy_quote"]:
            df[col] = df[col].astype(float)

        df["trades"] = df["trades"].astype(int)

        # Clean up
        df = df.drop(columns=["ignore"])
        df = df.drop_duplicates(subset=["open_time"])
        df = df.sort_values("open_time").reset_index(drop=True)

        logger.info(f"  Downloaded: {len(df):,} candles")

        return df

    # =========================================================================
    # B. Funding Rate
    # =========================================================================

    def download_funding_rate(
        self,
        symbol: str,
        days: int | None = None,
    ) -> pd.DataFrame:
        """
        Download funding rate history.

        Uses /fapi/v1/fundingRate endpoint.
        Funding rate is applied every 8 hours (00:00, 08:00, 16:00 UTC).
        """
        days = days or self.config.days
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        logger.info(f"[{symbol}] Downloading funding rates...")

        all_funding = []
        current_start = start_ms

        while current_start < end_ms:
            funding = self._request("/fapi/v1/fundingRate", {
                "symbol": symbol,
                "startTime": current_start,
                "endTime": end_ms,
                "limit": 1000,
            })

            if not funding:
                break

            all_funding.extend(funding)

            # Move to next batch
            last_time = funding[-1]["fundingTime"]
            current_start = last_time + 1

        if not all_funding:
            return pd.DataFrame()

        df = pd.DataFrame(all_funding)
        df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")

        # Handle markPrice which may be empty string for older records
        if "markPrice" in df.columns:
            df["markPrice"] = df["markPrice"].replace("", pd.NA)
            df["markPrice"] = pd.to_numeric(df["markPrice"], errors="coerce")

        df = df.drop_duplicates(subset=["fundingTime"])
        df = df.sort_values("fundingTime").reset_index(drop=True)

        logger.info(f"  Downloaded: {len(df):,} funding rate records")

        return df

    # =========================================================================
    # C. Mark Price Klines
    # =========================================================================

    def download_mark_price_klines(
        self,
        symbol: str,
        interval: str = "1m",
        days: int | None = None,
    ) -> pd.DataFrame:
        """
        Download mark price klines.

        Uses /fapi/v1/markPriceKlines endpoint.
        Mark price is used for liquidation and PnL calculation.
        """
        days = days or self.config.days
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        logger.info(f"[{symbol}] Downloading mark price {interval} klines...")

        all_klines = []
        current_start = start_ms

        while current_start < end_ms:
            klines = self._request("/fapi/v1/markPriceKlines", {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_ms,
                "limit": 1500,
            })

            if not klines:
                break

            all_klines.extend(klines)

            last_close_time = klines[-1][6]
            current_start = last_close_time + 1

        if not all_klines:
            return pd.DataFrame()

        df = pd.DataFrame(all_klines, columns=[
            "open_time", "open", "high", "low", "close",
            "ignore1", "close_time", "ignore2", "ignore3",
            "ignore4", "ignore5", "ignore6"
        ])

        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)

        df = df[["open_time", "open", "high", "low", "close", "close_time"]]
        df = df.drop_duplicates(subset=["open_time"])
        df = df.sort_values("open_time").reset_index(drop=True)

        logger.info(f"  Downloaded: {len(df):,} mark price candles")

        return df

    # =========================================================================
    # D. Index Price Klines
    # =========================================================================

    def download_index_price_klines(
        self,
        symbol: str,
        interval: str = "1m",
        days: int | None = None,
    ) -> pd.DataFrame:
        """
        Download index price klines.

        Uses /fapi/v1/indexPriceKlines endpoint.
        Index price is the weighted average of spot prices.
        """
        days = days or self.config.days
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        # Index symbol format: e.g., BTCUSDT -> BTCUSD
        index_symbol = symbol.replace("USDT", "USD")

        logger.info(f"[{symbol}] Downloading index price {interval} klines...")

        all_klines = []
        current_start = start_ms

        while current_start < end_ms:
            try:
                klines = self._request("/fapi/v1/indexPriceKlines", {
                    "pair": index_symbol,
                    "interval": interval,
                    "startTime": current_start,
                    "endTime": end_ms,
                    "limit": 1500,
                })
            except Exception as e:
                logger.warning(f"  Index price not available: {e}")
                return pd.DataFrame()

            if not klines:
                break

            all_klines.extend(klines)

            last_close_time = klines[-1][6]
            current_start = last_close_time + 1

        if not all_klines:
            return pd.DataFrame()

        df = pd.DataFrame(all_klines, columns=[
            "open_time", "open", "high", "low", "close",
            "ignore1", "close_time", "ignore2", "ignore3",
            "ignore4", "ignore5", "ignore6"
        ])

        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)

        df = df[["open_time", "open", "high", "low", "close", "close_time"]]
        df = df.drop_duplicates(subset=["open_time"])
        df = df.sort_values("open_time").reset_index(drop=True)

        logger.info(f"  Downloaded: {len(df):,} index price candles")

        return df

    # =========================================================================
    # E. Exchange Info (Symbol Filters)
    # =========================================================================

    def download_exchange_info(self) -> dict[str, Any]:
        """
        Download exchange info including symbol filters.

        Uses /fapi/v1/exchangeInfo endpoint.
        Saves: tickSize, stepSize, minNotional, price/qty precision.
        """
        logger.info("Downloading exchange info...")

        info = self._request("/fapi/v1/exchangeInfo")

        if not info:
            return {}

        # Extract relevant symbol filters
        symbol_filters = {}
        for sym_info in info.get("symbols", []):
            symbol = sym_info["symbol"]
            if symbol not in self.config.symbols:
                continue

            filters = {}
            for f in sym_info.get("filters", []):
                filter_type = f["filterType"]
                if filter_type == "PRICE_FILTER":
                    filters["tickSize"] = f["tickSize"]
                    filters["minPrice"] = f["minPrice"]
                    filters["maxPrice"] = f["maxPrice"]
                elif filter_type == "LOT_SIZE":
                    filters["stepSize"] = f["stepSize"]
                    filters["minQty"] = f["minQty"]
                    filters["maxQty"] = f["maxQty"]
                elif filter_type == "MIN_NOTIONAL":
                    filters["minNotional"] = f.get("notional", f.get("minNotional"))
                elif filter_type == "MARKET_LOT_SIZE":
                    filters["marketStepSize"] = f["stepSize"]
                    filters["marketMinQty"] = f["minQty"]
                    filters["marketMaxQty"] = f["maxQty"]

            symbol_filters[symbol] = {
                "symbol": symbol,
                "pair": sym_info.get("pair"),
                "contractType": sym_info.get("contractType"),
                "pricePrecision": sym_info.get("pricePrecision"),
                "quantityPrecision": sym_info.get("quantityPrecision"),
                "baseAsset": sym_info.get("baseAsset"),
                "quoteAsset": sym_info.get("quoteAsset"),
                "marginAsset": sym_info.get("marginAsset"),
                "filters": filters,
            }

        logger.info(f"  Downloaded info for {len(symbol_filters)} symbols")

        return {
            "serverTime": info.get("serverTime"),
            "timezone": info.get("timezone"),
            "downloadedAt": datetime.now(timezone.utc).isoformat(),
            "symbols": symbol_filters,
        }

    # =========================================================================
    # F. Open Interest History
    # =========================================================================

    def download_open_interest(
        self,
        symbol: str,
        period: str = "5m",
        days: int | None = None,
    ) -> pd.DataFrame:
        """
        Download historical open interest.

        Uses /futures/data/openInterestHist endpoint.
        Period: 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d
        """
        days = days or min(self.config.days, 30)  # API limits to ~30 days
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        logger.info(f"[{symbol}] Downloading open interest ({period})...")

        all_oi = []
        current_start = start_ms
        request_count = 0

        while current_start < end_ms:
            try:
                oi = self._request("/futures/data/openInterestHist", {
                    "symbol": symbol,
                    "period": period,
                    "startTime": current_start,
                    "endTime": end_ms,
                    "limit": 500,
                })
            except Exception as e:
                logger.warning(f"  Open interest not available: {e}")
                break

            if not oi:
                break

            all_oi.extend(oi)
            request_count += 1

            if request_count % 5 == 0:
                logger.info(f"  Progress: {len(all_oi):,} OI records...")

            last_time = oi[-1]["timestamp"]
            current_start = last_time + 1

        if not all_oi:
            return pd.DataFrame()

        df = pd.DataFrame(all_oi)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["sumOpenInterest"] = df["sumOpenInterest"].astype(float)
        df["sumOpenInterestValue"] = df["sumOpenInterestValue"].astype(float)

        df = df.drop_duplicates(subset=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(f"  Downloaded: {len(df):,} OI records")

        return df

    # =========================================================================
    # G. Long/Short Ratio
    # =========================================================================

    def download_long_short_ratio(
        self,
        symbol: str,
        period: str = "5m",
        days: int | None = None,
    ) -> pd.DataFrame:
        """
        Download global long/short account ratio.

        Uses /futures/data/globalLongShortAccountRatio endpoint.
        """
        days = days or min(self.config.days, 30)
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        logger.info(f"[{symbol}] Downloading long/short ratio ({period})...")

        all_ratio = []
        current_start = start_ms
        request_count = 0

        while current_start < end_ms:
            try:
                ratio = self._request("/futures/data/globalLongShortAccountRatio", {
                    "symbol": symbol,
                    "period": period,
                    "startTime": current_start,
                    "endTime": end_ms,
                    "limit": 500,
                })
            except Exception as e:
                logger.warning(f"  Long/short ratio not available: {e}")
                break

            if not ratio:
                break

            all_ratio.extend(ratio)
            request_count += 1

            if request_count % 5 == 0:
                logger.info(f"  Progress: {len(all_ratio):,} L/S ratio records...")

            last_time = ratio[-1]["timestamp"]
            current_start = last_time + 1

        if not all_ratio:
            return pd.DataFrame()

        df = pd.DataFrame(all_ratio)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["longShortRatio"] = df["longShortRatio"].astype(float)
        df["longAccount"] = df["longAccount"].astype(float)
        df["shortAccount"] = df["shortAccount"].astype(float)

        df = df.drop_duplicates(subset=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(f"  Downloaded: {len(df):,} L/S ratio records")

        return df

    # =========================================================================
    # H. Aggregated Trades (Optional, Heavy)
    # =========================================================================

    def download_agg_trades(
        self,
        symbol: str,
        days: int = 1,  # Default to just 1 day (very heavy)
    ) -> pd.DataFrame:
        """
        Download aggregated trades.

        Uses /fapi/v1/aggTrades endpoint.
        WARNING: This is very data-heavy. 1 day can be millions of records.
        """
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        logger.info(f"[{symbol}] Downloading aggregated trades ({days} days)...")
        logger.warning("  This may take a long time and generate large files!")

        all_trades = []
        current_start = start_ms
        request_count = 0

        while current_start < end_ms:
            trades = self._request("/fapi/v1/aggTrades", {
                "symbol": symbol,
                "startTime": current_start,
                "endTime": end_ms,
                "limit": 1000,
            })

            if not trades:
                break

            all_trades.extend(trades)
            request_count += 1

            if request_count % 100 == 0:
                logger.info(f"  Progress: {len(all_trades):,} trades...")

            last_time = trades[-1]["T"]
            current_start = last_time + 1

        if not all_trades:
            return pd.DataFrame()

        df = pd.DataFrame(all_trades)
        df = df.rename(columns={
            "a": "agg_trade_id",
            "p": "price",
            "q": "quantity",
            "f": "first_trade_id",
            "l": "last_trade_id",
            "T": "timestamp",
            "m": "is_buyer_maker",
        })

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["price"] = df["price"].astype(float)
        df["quantity"] = df["quantity"].astype(float)

        df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(f"  Downloaded: {len(df):,} aggregated trades")

        return df

    # =========================================================================
    # Data Processing (Clean & Resample)
    # =========================================================================

    def resample_ohlcv(
        self,
        df: pd.DataFrame,
        target_interval: str,
    ) -> pd.DataFrame:
        """
        Resample 1m OHLCV to larger timeframes.

        Properly handles OHLCV aggregation:
        - open: first
        - high: max
        - low: min
        - close: last
        - volume: sum
        """
        if df.empty:
            return df

        df = df.set_index("open_time")

        # Pandas offset aliases
        offset_map = {
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "1h",
            "2h": "2h",
            "4h": "4h",
            "6h": "6h",
            "12h": "12h",
            "1d": "1D",
        }

        offset = offset_map.get(target_interval, "5min")

        resampled = df.resample(offset, label="left", closed="left").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "quote_volume": "sum",
            "trades": "sum",
            "taker_buy_base": "sum",
            "taker_buy_quote": "sum",
        }).dropna()

        resampled = resampled.reset_index()
        resampled = resampled.rename(columns={"open_time": "open_time"})

        return resampled

    def validate_data(self, df: pd.DataFrame, interval: str = "1m") -> dict[str, Any]:
        """
        Validate data quality.

        Checks:
        - Missing bars
        - Duplicates
        - Timezone consistency
        - Price anomalies
        """
        if df.empty:
            return {"valid": False, "error": "Empty DataFrame"}

        issues = []

        # Check for duplicates
        duplicates = df.duplicated(subset=["open_time"]).sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate timestamps")

        # Check for missing bars
        interval_ms = TIMEFRAME_MS.get(interval, 60_000)
        expected_bars = (df["open_time"].max() - df["open_time"].min()).total_seconds() * 1000 / interval_ms
        actual_bars = len(df)
        missing_pct = (expected_bars - actual_bars) / expected_bars * 100 if expected_bars > 0 else 0

        if missing_pct > 1:
            issues.append(f"Missing {missing_pct:.2f}% of expected bars")

        # Check timezone
        if df["open_time"].dt.tz is None:
            issues.append("Timestamps missing timezone")
        elif str(df["open_time"].dt.tz) != "UTC":
            issues.append(f"Timezone is {df['open_time'].dt.tz}, expected UTC")

        # Check for price anomalies (OHLC consistency)
        if "high" in df.columns and "low" in df.columns:
            invalid_hl = (df["high"] < df["low"]).sum()
            if invalid_hl > 0:
                issues.append(f"Found {invalid_hl} bars where high < low")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "stats": {
                "total_bars": len(df),
                "date_range": f"{df['open_time'].min()} to {df['open_time'].max()}",
                "missing_pct": missing_pct,
                "duplicates": duplicates,
            }
        }

    # =========================================================================
    # Save/Load Methods
    # =========================================================================

    def save_raw(self, df: pd.DataFrame, symbol: str, data_type: str) -> Path:
        """Save raw data to CSV."""
        path = self.config.base_dir / "raw" / symbol / f"{data_type}.csv"
        df.to_csv(path, index=False)
        logger.info(f"  Saved raw: {path}")
        return path

    def save_clean(self, df: pd.DataFrame, symbol: str, data_type: str) -> Path:
        """Save clean data to Parquet."""
        path = self.config.base_dir / "clean" / symbol / f"{data_type}.parquet"
        try:
            df.to_parquet(path, index=False)
        except ImportError:
            # Fallback to CSV if pyarrow not available
            path = path.with_suffix(".csv")
            df.to_csv(path, index=False)
        logger.info(f"  Saved clean: {path}")
        return path

    def save_meta(self, data: dict[str, Any], filename: str) -> Path:
        """Save metadata to JSON."""
        path = self.config.base_dir / "meta" / filename
        path.write_text(json.dumps(data, indent=2, default=str))
        logger.info(f"  Saved meta: {path}")
        return path

    def load_clean(self, symbol: str, data_type: str) -> pd.DataFrame | None:
        """Load clean data from cache."""
        parquet_path = self.config.base_dir / "clean" / symbol / f"{data_type}.parquet"
        csv_path = self.config.base_dir / "clean" / symbol / f"{data_type}.csv"

        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path)
            # Convert timestamp columns
            for col in df.columns:
                if "time" in col.lower() or col == "timestamp":
                    df[col] = pd.to_datetime(df[col], utc=True)
            return df

        return None

    def has_cache(self, symbol: str, data_type: str) -> bool:
        """Check if clean cache exists for given data type."""
        parquet_path = self.config.base_dir / "clean" / symbol / f"{data_type}.parquet"
        csv_path = self.config.base_dir / "clean" / symbol / f"{data_type}.csv"
        return parquet_path.exists() or csv_path.exists()

    def get_cache_info(self, symbol: str, data_type: str) -> dict[str, Any] | None:
        """Get info about cached data."""
        df = self.load_clean(symbol, data_type)
        if df is None:
            return None

        time_col = None
        for col in ["open_time", "timestamp", "fundingTime"]:
            if col in df.columns:
                time_col = col
                break

        if time_col:
            return {
                "records": len(df),
                "start": df[time_col].min(),
                "end": df[time_col].max(),
            }
        return {"records": len(df)}

    # =========================================================================
    # Main Download Orchestrator
    # =========================================================================

    def download_all(self) -> dict[str, Any]:
        """
        Download all configured data types for all symbols.

        Returns manifest with download status.
        """
        setup_logging(level="INFO")

        manifest = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "config": {
                "symbols": self.config.symbols,
                "days": self.config.days,
                "base_dir": str(self.config.base_dir),
            },
            "downloads": {},
        }

        logger.info("")
        logger.info("=" * 70)
        logger.info("BINANCE USDT-M FUTURES DATA DOWNLOADER")
        logger.info("=" * 70)
        logger.info(f"  Symbols:    {', '.join(self.config.symbols)}")
        logger.info(f"  Days:       {self.config.days}")
        logger.info(f"  Output:     {self.config.base_dir}")
        logger.info("=" * 70)

        # Download exchange info first (static data)
        if self.config.download_exchange_info:
            logger.info("")
            logger.info("-" * 50)
            logger.info("Downloading Exchange Info...")
            logger.info("-" * 50)

            exchange_info = self.download_exchange_info()
            self.save_meta(exchange_info, "exchange_info.json")
            self.save_meta(exchange_info.get("symbols", {}), "symbol_filters.json")
            manifest["downloads"]["exchange_info"] = True

        # Download per-symbol data
        for symbol in self.config.symbols:
            logger.info("")
            logger.info("=" * 50)
            logger.info(f"SYMBOL: {symbol}")
            logger.info("=" * 50)

            symbol_manifest = {}

            # A. OHLCV Klines
            if self.config.download_ohlcv:
                logger.info("")
                cache_info = self.get_cache_info(symbol, "ohlcv_1m")
                if cache_info and not self.config.force_download:
                    logger.info(f"[{symbol}] OHLCV already cached: {cache_info['records']:,} records")
                    logger.info(f"  Range: {cache_info.get('start')} to {cache_info.get('end')}")
                    symbol_manifest["ohlcv"] = {"records": cache_info["records"], "cached": True}
                else:
                    df = self.download_klines(symbol, "1m")
                    if not df.empty:
                        self.save_raw(df, symbol, "klines_1m")

                        # Validate and save clean
                        validation = self.validate_data(df, "1m")
                        if validation["valid"]:
                            self.save_clean(df, symbol, "ohlcv_1m")

                            # Resample to larger timeframes
                            for tf in ["5m", "15m", "1h", "4h"]:
                                resampled = self.resample_ohlcv(df, tf)
                                self.save_clean(resampled, symbol, f"ohlcv_{tf}")

                        symbol_manifest["ohlcv"] = {
                            "records": len(df),
                            "validation": validation,
                        }

            # B. Funding Rate
            if self.config.download_funding:
                logger.info("")
                cache_info = self.get_cache_info(symbol, "funding_rate")
                if cache_info and not self.config.force_download:
                    logger.info(f"[{symbol}] Funding rate already cached: {cache_info['records']:,} records")
                    symbol_manifest["funding_rate"] = {"records": cache_info["records"], "cached": True}
                else:
                    df = self.download_funding_rate(symbol)
                    if not df.empty:
                        self.save_raw(df, symbol, "funding_rate")
                        self.save_clean(df, symbol, "funding_rate")
                        symbol_manifest["funding_rate"] = {"records": len(df)}

            # C. Mark Price
            if self.config.download_mark_price:
                logger.info("")
                cache_info = self.get_cache_info(symbol, "mark_price_1m")
                if cache_info and not self.config.force_download:
                    logger.info(f"[{symbol}] Mark price already cached: {cache_info['records']:,} records")
                    symbol_manifest["mark_price"] = {"records": cache_info["records"], "cached": True}
                else:
                    df = self.download_mark_price_klines(symbol, "1m")
                    if not df.empty:
                        self.save_raw(df, symbol, "mark_price_1m")
                        self.save_clean(df, symbol, "mark_price_1m")
                        symbol_manifest["mark_price"] = {"records": len(df)}

            # D. Index Price
            if self.config.download_index_price:
                logger.info("")
                cache_info = self.get_cache_info(symbol, "index_price_1m")
                if cache_info and not self.config.force_download:
                    logger.info(f"[{symbol}] Index price already cached: {cache_info['records']:,} records")
                    symbol_manifest["index_price"] = {"records": cache_info["records"], "cached": True}
                else:
                    df = self.download_index_price_klines(symbol, "1m")
                    if not df.empty:
                        self.save_raw(df, symbol, "index_price_1m")
                        self.save_clean(df, symbol, "index_price_1m")
                        symbol_manifest["index_price"] = {"records": len(df)}

            # E. Open Interest (always re-download, only recent 30 days available)
            if self.config.download_open_interest:
                logger.info("")
                df = self.download_open_interest(symbol, "5m")
                if not df.empty:
                    self.save_raw(df, symbol, "open_interest_5m")
                    self.save_clean(df, symbol, "open_interest_5m")
                    symbol_manifest["open_interest"] = {"records": len(df)}

            # F. Long/Short Ratio (always re-download, only recent 30 days available)
            if self.config.download_long_short_ratio:
                logger.info("")
                df = self.download_long_short_ratio(symbol, "5m")
                if not df.empty:
                    self.save_raw(df, symbol, "long_short_ratio_5m")
                    self.save_clean(df, symbol, "long_short_ratio_5m")
                    symbol_manifest["long_short_ratio"] = {"records": len(df)}

            # G. Aggregated Trades (optional, heavy)
            if self.config.download_agg_trades:
                logger.info("")
                cache_info = self.get_cache_info(symbol, "agg_trades")
                if cache_info and not self.config.force_download:
                    logger.info(f"[{symbol}] Agg trades already cached: {cache_info['records']:,} records")
                    symbol_manifest["agg_trades"] = {"records": cache_info["records"], "cached": True}
                else:
                    df = self.download_agg_trades(symbol, days=1)
                    if not df.empty:
                        self.save_raw(df, symbol, "agg_trades")
                        symbol_manifest["agg_trades"] = {"records": len(df)}

            manifest["downloads"][symbol] = symbol_manifest

        # Save manifest
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        self.save_meta(manifest, "download_manifest.json")

        # Print summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("DOWNLOAD COMPLETE")
        logger.info("=" * 70)
        logger.info(f"  Output directory: {self.config.base_dir}")
        logger.info("")
        logger.info("  Data structure:")
        logger.info("  ├── raw/       (원천 CSV)")
        logger.info("  ├── clean/     (정제 Parquet/CSV)")
        logger.info("  ├── meta/      (메타데이터 JSON)")
        logger.info("  └── features/  (파생 피처, 추후 생성)")
        logger.info("=" * 70)

        return manifest


def download_futures_data(
    symbols: list[str] | None = None,
    days: int = 365,
    base_dir: str = "data/futures",
    include_trades: bool = False,
) -> dict[str, Any]:
    """
    Convenience function to download futures data.

    Args:
        symbols: List of symbols (default: BTCUSDT, ETHUSDT)
        days: Number of days to download
        base_dir: Output directory
        include_trades: Include aggregated trades (heavy)

    Returns:
        Download manifest
    """
    config = FuturesDataConfig(
        symbols=symbols or ["BTCUSDT", "ETHUSDT"],
        days=days,
        base_dir=Path(base_dir),
        download_agg_trades=include_trades,
    )

    downloader = FuturesDataDownloader(config)
    return downloader.download_all()

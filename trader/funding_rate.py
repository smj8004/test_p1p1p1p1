"""
Funding Rate Arbitrage Strategy

원리:
- 현물 BTC 매수 + 선물 BTC 숏 = Delta Neutral (가격 중립)
- Funding Rate가 양수일 때: 롱이 숏에게 지불 → 숏 포지션 수익
- Funding Rate가 음수일 때: 숏이 롱에게 지불 → 숏 포지션 손실

수익 구조:
- 8시간마다 funding rate 정산 (하루 3회: 00:00, 08:00, 16:00 UTC)
- 연환산 수익률 = 평균 funding rate * 3 * 365 * 100%
- 예: 0.01% * 3 * 365 = 10.95% 연간

리스크:
- Funding rate가 음수로 전환되면 손실
- 거래소 리스크 (해킹, 파산)
- 청산 리스크 (레버리지 사용 시)
- 현물/선물 가격 괴리 (basis risk)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import aiohttp
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FundingConfig:
    """Funding Rate 수집 설정"""
    data_dir: Path = field(default_factory=lambda: Path("data/futures/funding"))
    symbols: list[str] = field(default_factory=lambda: [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
        "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"
    ])
    base_url: str = "https://fapi.binance.com"
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class FundingArbitrageConfig:
    """Funding Rate 차익거래 설정"""
    initial_capital: float = 10000.0  # 초기 자본
    spot_allocation: float = 0.5  # 현물에 할당 비율 (50%)
    futures_allocation: float = 0.5  # 선물에 할당 비율 (50%)
    futures_leverage: int = 1  # 선물 레버리지 (1x = 청산 위험 없음)
    spot_fee: float = 0.001  # 현물 거래 수수료 (0.1%)
    futures_fee: float = 0.0004  # 선물 거래 수수료 (0.04%)
    min_funding_rate: float = 0.0001  # 최소 진입 funding rate (0.01%)
    exit_funding_rate: float = -0.0001  # 청산 기준 (-0.01%)
    rebalance_threshold: float = 0.05  # 리밸런싱 기준 (5% 괴리)


class FundingRateCollector:
    """Funding Rate 데이터 수집기"""

    def __init__(self, config: FundingConfig | None = None):
        self.config = config or FundingConfig()
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

    def _get_data_path(self, symbol: str) -> Path:
        """심볼별 데이터 파일 경로"""
        return self.config.data_dir / f"{symbol}_funding.parquet"

    def _get_existing_data(self, symbol: str) -> pd.DataFrame | None:
        """기존 저장된 데이터 로드"""
        path = self._get_data_path(symbol)
        if path.exists():
            try:
                df = pd.read_parquet(path)
                logger.info(f"Loaded existing {symbol} data: {len(df)} records")
                return df
            except Exception as e:
                logger.warning(f"Failed to load {symbol} data: {e}")
        return None

    async def _fetch_funding_rate(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 1000
    ) -> list[dict]:
        """Binance API에서 funding rate 조회"""
        url = f"{self.config.base_url}/fapi/v1/fundingRate"
        params = {"symbol": symbol, "limit": limit}

        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        for attempt in range(self.config.max_retries):
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:
                        # Rate limit
                        retry_after = int(response.headers.get("Retry-After", 60))
                        logger.warning(f"Rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                    else:
                        text = await response.text()
                        logger.error(f"API error {response.status}: {text}")
                        await asyncio.sleep(self.config.retry_delay * (attempt + 1))
            except Exception as e:
                logger.error(f"Request failed: {e}")
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))

        return []

    async def _fetch_all_funding_rates(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        start_time: datetime | None = None
    ) -> pd.DataFrame:
        """특정 심볼의 모든 funding rate 조회 (페이지네이션)"""
        all_data = []

        # 시작 시간 설정 (기본: 30일 전)
        if start_time is None:
            start_time = datetime.now(timezone.utc) - timedelta(days=30)

        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

        current_start = start_ts

        while current_start < end_ts:
            data = await self._fetch_funding_rate(
                session, symbol,
                start_time=current_start,
                end_time=end_ts,
                limit=1000
            )

            if not data:
                break

            all_data.extend(data)

            # 다음 페이지 시작점
            last_time = max(d["fundingTime"] for d in data)
            if last_time <= current_start:
                break
            current_start = last_time + 1

            # Rate limit 방지
            await asyncio.sleep(0.1)

        if not all_data:
            return pd.DataFrame()

        # DataFrame 변환
        df = pd.DataFrame(all_data)
        df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["fundingRate"] = df["fundingRate"].astype(float)
        df["markPrice"] = df["markPrice"].astype(float)
        df = df.rename(columns={"fundingTime": "timestamp"})
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

        return df

    async def download_symbol(self, symbol: str, force_full: bool = False) -> pd.DataFrame:
        """단일 심볼 funding rate 다운로드 및 저장"""
        async with aiohttp.ClientSession() as session:
            # 기존 데이터 로드
            existing_df = self._get_existing_data(symbol)

            if existing_df is not None and not force_full:
                # 마지막 데이터 이후부터 다운로드
                last_time = existing_df["timestamp"].max()
                start_time = last_time.to_pydatetime()
                logger.info(f"Downloading {symbol} from {start_time}")
            else:
                # 30일 전부터 다운로드
                start_time = datetime.now(timezone.utc) - timedelta(days=30)
                logger.info(f"Downloading {symbol} full history (30 days)")
                existing_df = None

            # 새 데이터 다운로드
            new_df = await self._fetch_all_funding_rates(session, symbol, start_time)

            if new_df.empty:
                logger.warning(f"No new data for {symbol}")
                return existing_df if existing_df is not None else pd.DataFrame()

            # 기존 데이터와 병합
            if existing_df is not None:
                combined_df = pd.concat([existing_df, new_df])
                combined_df = combined_df.drop_duplicates(subset=["timestamp"])
                combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)
            else:
                combined_df = new_df.reset_index(drop=True)

            # 저장
            path = self._get_data_path(symbol)
            combined_df.to_parquet(path, index=False)
            logger.info(f"Saved {symbol}: {len(combined_df)} records ({len(new_df)} new)")

            return combined_df

    async def download_all(self, force_full: bool = False) -> dict[str, pd.DataFrame]:
        """모든 심볼 다운로드"""
        results = {}

        for symbol in self.config.symbols:
            try:
                df = await self.download_symbol(symbol, force_full)
                results[symbol] = df
                await asyncio.sleep(0.5)  # Rate limit 방지
            except Exception as e:
                logger.error(f"Failed to download {symbol}: {e}")

        return results

    def load_symbol(self, symbol: str) -> pd.DataFrame | None:
        """저장된 심볼 데이터 로드"""
        return self._get_existing_data(symbol)

    def load_all(self) -> dict[str, pd.DataFrame]:
        """모든 저장된 데이터 로드"""
        results = {}
        for symbol in self.config.symbols:
            df = self.load_symbol(symbol)
            if df is not None:
                results[symbol] = df
        return results


class FundingRateAnalyzer:
    """Funding Rate 분석기"""

    def __init__(self, data: dict[str, pd.DataFrame]):
        self.data = data

    def get_summary(self, symbol: str) -> dict:
        """심볼별 요약 통계"""
        if symbol not in self.data:
            return {}

        df = self.data[symbol]
        if df.empty:
            return {}

        rates = df["fundingRate"]

        # 기본 통계
        mean_rate = rates.mean()
        std_rate = rates.std()
        min_rate = rates.min()
        max_rate = rates.max()
        positive_ratio = (rates > 0).sum() / len(rates)

        # 연환산 수익률 (8시간마다 3회/일)
        annual_return = mean_rate * 3 * 365

        # 최근 7일/30일 평균
        recent_7d = df[df["timestamp"] >= df["timestamp"].max() - timedelta(days=7)]
        recent_30d = df[df["timestamp"] >= df["timestamp"].max() - timedelta(days=30)]

        mean_7d = recent_7d["fundingRate"].mean() if not recent_7d.empty else 0
        mean_30d = recent_30d["fundingRate"].mean() if not recent_30d.empty else 0

        return {
            "symbol": symbol,
            "records": len(df),
            "start_date": df["timestamp"].min(),
            "end_date": df["timestamp"].max(),
            "mean_rate": mean_rate,
            "std_rate": std_rate,
            "min_rate": min_rate,
            "max_rate": max_rate,
            "positive_ratio": positive_ratio,
            "annual_return_pct": annual_return * 100,
            "mean_7d": mean_7d,
            "mean_30d": mean_30d,
            "annual_7d_pct": mean_7d * 3 * 365 * 100,
            "annual_30d_pct": mean_30d * 3 * 365 * 100,
        }

    def get_all_summaries(self) -> pd.DataFrame:
        """모든 심볼 요약"""
        summaries = []
        for symbol in self.data:
            summary = self.get_summary(symbol)
            if summary:
                summaries.append(summary)

        if not summaries:
            return pd.DataFrame()

        df = pd.DataFrame(summaries)
        df = df.sort_values("annual_return_pct", ascending=False)
        return df

    def get_combined_history(self, symbols: list[str] | None = None) -> pd.DataFrame:
        """여러 심볼 funding rate 이력 결합"""
        if symbols is None:
            symbols = list(self.data.keys())

        dfs = []
        for symbol in symbols:
            if symbol in self.data:
                df = self.data[symbol].copy()
                df["symbol"] = symbol
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs).sort_values("timestamp")

    def find_best_symbols(self, top_n: int = 5, min_positive_ratio: float = 0.7) -> list[str]:
        """최적 심볼 선정 (높은 연환산 수익률 + 안정성)"""
        summaries = self.get_all_summaries()
        if summaries.empty:
            return []

        # 필터: 양수 비율 70% 이상
        filtered = summaries[summaries["positive_ratio"] >= min_positive_ratio]

        # 정렬: 연환산 수익률 기준
        filtered = filtered.sort_values("annual_return_pct", ascending=False)

        return filtered.head(top_n)["symbol"].tolist()


class FundingArbitrageBacktester:
    """Funding Rate 차익거래 백테스터"""

    def __init__(
        self,
        funding_data: pd.DataFrame,
        price_data: pd.DataFrame,
        config: FundingArbitrageConfig | None = None
    ):
        """
        Args:
            funding_data: Funding rate 데이터 (timestamp, fundingRate, symbol)
            price_data: 가격 데이터 (timestamp, close)
            config: 백테스트 설정
        """
        self.funding_data = funding_data
        self.price_data = price_data
        self.config = config or FundingArbitrageConfig()

    def run(self) -> dict:
        """백테스트 실행"""
        config = self.config

        # 초기 상태
        capital = config.initial_capital
        spot_capital = capital * config.spot_allocation
        futures_capital = capital * config.futures_allocation

        position_open = False
        spot_btc = 0.0
        futures_btc = 0.0
        entry_price = 0.0
        entry_time = None

        # 기록
        trades = []
        funding_received = []
        equity_curve = []

        # 가격 데이터를 시간별로 인덱싱
        price_df = self.price_data.set_index("timestamp").sort_index()
        funding_df = self.funding_data.set_index("timestamp").sort_index()

        # 시뮬레이션 시작
        for idx, row in funding_df.iterrows():
            timestamp = idx
            funding_rate = row["fundingRate"]

            # 현재 가격 찾기 (가장 가까운 시간)
            try:
                price_idx = price_df.index.get_indexer([timestamp], method="ffill")[0]
                if price_idx < 0:
                    continue
                current_price = price_df.iloc[price_idx]["close"]
            except Exception:
                continue

            # 포지션 진입 조건
            if not position_open and funding_rate >= config.min_funding_rate:
                # 현물 매수 + 선물 숏 진입
                spot_fee = spot_capital * config.spot_fee
                futures_fee = futures_capital * config.futures_fee

                spot_btc = (spot_capital - spot_fee) / current_price
                futures_btc = futures_capital / current_price  # 숏 포지션 크기

                entry_price = current_price
                entry_time = timestamp
                position_open = True

                trades.append({
                    "type": "entry",
                    "timestamp": timestamp,
                    "price": current_price,
                    "spot_btc": spot_btc,
                    "futures_btc": futures_btc,
                    "spot_fee": spot_fee,
                    "futures_fee": futures_fee
                })

            # 포지션 청산 조건
            elif position_open and funding_rate <= config.exit_funding_rate:
                # 현물 매도 + 선물 롱 청산
                spot_value = spot_btc * current_price
                spot_fee = spot_value * config.spot_fee
                futures_pnl = (entry_price - current_price) / entry_price * futures_capital
                futures_fee = futures_capital * config.futures_fee

                # 총 펀딩 수익 계산
                total_funding = sum(f["amount"] for f in funding_received if f["timestamp"] >= entry_time)

                # 자본 업데이트
                spot_capital = spot_value - spot_fee
                futures_capital = futures_capital + futures_pnl - futures_fee
                capital = spot_capital + futures_capital

                trades.append({
                    "type": "exit",
                    "timestamp": timestamp,
                    "price": current_price,
                    "spot_value": spot_value,
                    "futures_pnl": futures_pnl,
                    "total_funding": total_funding,
                    "capital": capital
                })

                position_open = False
                spot_btc = 0
                futures_btc = 0

            # 포지션 유지 중 펀딩 수취
            if position_open:
                # 숏 포지션이 펀딩비 수취 (양수 funding rate일 때)
                funding_amount = futures_btc * current_price * funding_rate
                futures_capital += funding_amount

                funding_received.append({
                    "timestamp": timestamp,
                    "rate": funding_rate,
                    "amount": funding_amount,
                    "btc_price": current_price
                })

            # Equity curve 기록
            if position_open:
                spot_value = spot_btc * current_price
                futures_pnl = (entry_price - current_price) / entry_price * (futures_capital - sum(f["amount"] for f in funding_received if f["timestamp"] >= entry_time))
                current_equity = spot_value + futures_capital
            else:
                current_equity = capital

            equity_curve.append({
                "timestamp": timestamp,
                "equity": current_equity,
                "position": position_open
            })

        # 최종 포지션 청산 (필요시)
        if position_open:
            final_price = price_df.iloc[-1]["close"]
            spot_value = spot_btc * final_price
            spot_fee = spot_value * config.spot_fee
            futures_pnl = (entry_price - final_price) / entry_price * futures_capital
            futures_fee = futures_capital * config.futures_fee

            total_funding = sum(f["amount"] for f in funding_received if f["timestamp"] >= entry_time)

            spot_capital = spot_value - spot_fee
            futures_capital = futures_capital + futures_pnl - futures_fee
            capital = spot_capital + futures_capital

            trades.append({
                "type": "final_exit",
                "timestamp": price_df.index[-1],
                "price": final_price,
                "spot_value": spot_value,
                "futures_pnl": futures_pnl,
                "total_funding": total_funding,
                "capital": capital
            })

        # 결과 계산
        total_funding_received = sum(f["amount"] for f in funding_received)
        total_return = (capital - config.initial_capital) / config.initial_capital * 100

        equity_df = pd.DataFrame(equity_curve)
        if not equity_df.empty:
            equity_df["peak"] = equity_df["equity"].cummax()
            equity_df["drawdown"] = (equity_df["equity"] - equity_df["peak"]) / equity_df["peak"] * 100
            max_drawdown = equity_df["drawdown"].min()
        else:
            max_drawdown = 0

        # 기간 계산
        if len(funding_df) > 1:
            days = (funding_df.index[-1] - funding_df.index[0]).days
        else:
            days = 1

        annual_return = total_return * 365 / max(days, 1)

        return {
            "initial_capital": config.initial_capital,
            "final_capital": capital,
            "total_return_pct": total_return,
            "annual_return_pct": annual_return,
            "total_funding_received": total_funding_received,
            "funding_count": len(funding_received),
            "avg_funding_per_event": total_funding_received / max(len(funding_received), 1),
            "total_trades": len(trades),
            "max_drawdown_pct": max_drawdown,
            "days": days,
            "trades": trades,
            "funding_history": funding_received,
            "equity_curve": equity_df.to_dict("records") if not equity_df.empty else []
        }


class FundingArbitrageSimulator:
    """Funding Rate 차익거래 시뮬레이터 (단순화된 버전)"""

    def __init__(self, funding_data: pd.DataFrame, config: FundingArbitrageConfig | None = None):
        """
        Args:
            funding_data: Funding rate 데이터 (timestamp, fundingRate)
            config: 설정
        """
        self.funding_data = funding_data.sort_values("timestamp")
        self.config = config or FundingArbitrageConfig()

    def run_simple(self) -> dict:
        """
        단순 시뮬레이션: 전 기간 포지션 유지 가정

        가정:
        - 현물 매수 + 선물 숏 동시 진입
        - 전 기간 포지션 유지
        - 가격 변동은 현물/선물이 상쇄됨
        - 수익 = 펀딩비 수취 - 수수료
        """
        config = self.config
        df = self.funding_data

        if df.empty:
            return {"error": "No data"}

        # 초기 자본
        capital = config.initial_capital
        position_size = capital  # 전체 자본을 포지션에 사용

        # 진입 수수료
        entry_fee = position_size * (config.spot_fee + config.futures_fee)
        capital -= entry_fee

        # 펀딩비 수취 시뮬레이션
        funding_history = []
        cumulative_funding = 0

        for _, row in df.iterrows():
            funding_rate = row["fundingRate"]
            # 숏 포지션이 받는 펀딩 (양수면 수취, 음수면 지불)
            funding_amount = position_size * funding_rate
            cumulative_funding += funding_amount

            funding_history.append({
                "timestamp": row["timestamp"],
                "rate": funding_rate,
                "amount": funding_amount,
                "cumulative": cumulative_funding
            })

        # 청산 수수료
        exit_fee = position_size * (config.spot_fee + config.futures_fee)

        # 최종 자본
        final_capital = capital + cumulative_funding - exit_fee

        # 통계 계산
        total_return = (final_capital - config.initial_capital) / config.initial_capital * 100

        days = (df["timestamp"].max() - df["timestamp"].min()).days
        if days == 0:
            days = 1
        annual_return = total_return * 365 / days

        # 양수/음수 펀딩 분리
        positive_funding = sum(f["amount"] for f in funding_history if f["amount"] > 0)
        negative_funding = sum(f["amount"] for f in funding_history if f["amount"] < 0)

        # 최대 낙폭 계산
        equity_curve = []
        running_equity = capital
        peak = running_equity
        max_drawdown = 0

        for f in funding_history:
            running_equity += f["amount"]
            peak = max(peak, running_equity)
            drawdown = (running_equity - peak) / peak * 100
            max_drawdown = min(max_drawdown, drawdown)
            equity_curve.append({
                "timestamp": f["timestamp"],
                "equity": running_equity,
                "drawdown": drawdown
            })

        return {
            "initial_capital": config.initial_capital,
            "final_capital": final_capital,
            "total_return_pct": total_return,
            "annual_return_pct": annual_return,
            "total_funding_received": cumulative_funding,
            "positive_funding": positive_funding,
            "negative_funding": negative_funding,
            "funding_events": len(funding_history),
            "avg_funding_rate": df["fundingRate"].mean(),
            "positive_rate_ratio": (df["fundingRate"] > 0).sum() / len(df),
            "entry_fee": entry_fee,
            "exit_fee": exit_fee,
            "total_fees": entry_fee + exit_fee,
            "net_profit": final_capital - config.initial_capital,
            "max_drawdown_pct": max_drawdown,
            "days": days,
            "sharpe_ratio": self._calculate_sharpe(funding_history),
            "funding_history": funding_history,
            "equity_curve": equity_curve
        }

    def _calculate_sharpe(self, funding_history: list[dict], risk_free_rate: float = 0.05) -> float:
        """샤프 비율 계산 (8시간 기준)"""
        if len(funding_history) < 2:
            return 0.0

        returns = [f["amount"] / self.config.initial_capital for f in funding_history]
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        # 연환산 (8시간마다 3회/일, 365일)
        periods_per_year = 3 * 365
        annual_return = mean_return * periods_per_year
        annual_std = std_return * np.sqrt(periods_per_year)

        sharpe = (annual_return - risk_free_rate) / annual_std
        return sharpe


class FundingRateMonitor:
    """실시간 Funding Rate 모니터링"""

    def __init__(self, config: FundingConfig | None = None):
        self.config = config or FundingConfig()

    async def get_current_rates(self) -> pd.DataFrame:
        """현재 funding rate 조회"""
        url = f"{self.config.base_url}/fapi/v1/premiumIndex"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"API error: {response.status}")

                data = await response.json()

        df = pd.DataFrame(data)
        df = df[["symbol", "lastFundingRate", "nextFundingTime", "markPrice"]]
        df["lastFundingRate"] = df["lastFundingRate"].astype(float)
        df["markPrice"] = df["markPrice"].astype(float)
        df["nextFundingTime"] = pd.to_datetime(df["nextFundingTime"], unit="ms", utc=True)
        df["annual_return_pct"] = df["lastFundingRate"] * 3 * 365 * 100

        # USDT 페어만 필터링
        df = df[df["symbol"].str.endswith("USDT")]
        df = df.sort_values("annual_return_pct", ascending=False)

        return df

    async def get_top_opportunities(self, top_n: int = 10, min_annual_return: float = 10.0) -> pd.DataFrame:
        """최고 수익률 기회 탐색"""
        df = await self.get_current_rates()

        # 필터: 양수 funding rate + 최소 연환산 수익률
        filtered = df[
            (df["lastFundingRate"] > 0) &
            (df["annual_return_pct"] >= min_annual_return)
        ]

        return filtered.head(top_n)


# CLI 함수들

def run_download(symbols: list[str] | None = None, force_full: bool = False):
    """Funding rate 데이터 다운로드"""
    config = FundingConfig()
    if symbols:
        config.symbols = symbols

    collector = FundingRateCollector(config)

    async def _download():
        return await collector.download_all(force_full)

    results = asyncio.run(_download())

    print("\n" + "=" * 60)
    print("FUNDING RATE DOWNLOAD COMPLETE")
    print("=" * 60 + "\n")

    for symbol, df in results.items():
        if not df.empty:
            print(f"{symbol}: {len(df)} records ({df['timestamp'].min().date()} to {df['timestamp'].max().date()})")
        else:
            print(f"{symbol}: No data")

    return results


def run_analyze(symbols: list[str] | None = None):
    """Funding rate 분석"""
    config = FundingConfig()
    if symbols:
        config.symbols = symbols

    collector = FundingRateCollector(config)
    data = collector.load_all()

    if not data:
        print("No data found. Run download first.")
        return

    analyzer = FundingRateAnalyzer(data)
    summaries = analyzer.get_all_summaries()

    print("\n" + "=" * 70)
    print("FUNDING RATE ANALYSIS")
    print("=" * 70 + "\n")

    for _, row in summaries.iterrows():
        symbol = row["symbol"]
        print(f"\n{symbol}:")
        print(f"  Records: {row['records']} | Period: {row['start_date'].date()} to {row['end_date'].date()}")
        print(f"  Mean Rate: {row['mean_rate']*100:.4f}% | Positive Ratio: {row['positive_ratio']*100:.1f}%")
        print(f"  Annual Return: {row['annual_return_pct']:.2f}%")
        print(f"  7-day Avg: {row['mean_7d']*100:.4f}% ({row['annual_7d_pct']:.2f}% annual)")
        print(f"  30-day Avg: {row['mean_30d']*100:.4f}% ({row['annual_30d_pct']:.2f}% annual)")

    # 최적 심볼 추천
    best = analyzer.find_best_symbols(top_n=3)
    print("\n" + "-" * 40)
    print(f"RECOMMENDED SYMBOLS: {', '.join(best)}")

    return summaries


def run_backtest(symbol: str = "BTCUSDT", initial_capital: float = 10000):
    """Funding rate 차익거래 백테스트"""
    config = FundingConfig()
    collector = FundingRateCollector(config)

    # Funding data 로드
    funding_df = collector.load_symbol(symbol)
    if funding_df is None or funding_df.empty:
        print(f"No funding data for {symbol}. Run download first.")
        return

    # 백테스트 설정
    arb_config = FundingArbitrageConfig(initial_capital=initial_capital)

    # 단순 시뮬레이션 실행
    simulator = FundingArbitrageSimulator(funding_df, arb_config)
    result = simulator.run_simple()

    print("\n" + "=" * 70)
    print(f"FUNDING ARBITRAGE BACKTEST: {symbol}")
    print("=" * 70 + "\n")

    print(f"Period: {result['days']} days")
    print(f"Initial Capital: ${result['initial_capital']:,.2f}")
    print(f"Final Capital: ${result['final_capital']:,.2f}")
    print(f"Net Profit: ${result['net_profit']:,.2f}")
    print(f"Total Return: {result['total_return_pct']:+.2f}%")
    print(f"Annual Return: {result['annual_return_pct']:+.2f}%")
    print()
    print(f"Funding Events: {result['funding_events']}")
    print(f"Avg Funding Rate: {result['avg_funding_rate']*100:.4f}%")
    print(f"Positive Rate Ratio: {result['positive_rate_ratio']*100:.1f}%")
    print()
    print(f"Total Funding Received: ${result['total_funding_received']:,.2f}")
    print(f"  - Positive: ${result['positive_funding']:,.2f}")
    print(f"  - Negative: ${result['negative_funding']:,.2f}")
    print(f"Total Fees: ${result['total_fees']:,.2f}")
    print()
    print(f"Max Drawdown: {result['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")

    return result


def run_monitor():
    """실시간 funding rate 모니터링"""
    monitor = FundingRateMonitor()

    async def _monitor():
        return await monitor.get_top_opportunities(top_n=15, min_annual_return=5.0)

    df = asyncio.run(_monitor())

    print("\n" + "=" * 70)
    print("TOP FUNDING RATE OPPORTUNITIES (Real-time)")
    print("=" * 70 + "\n")

    print(f"{'Symbol':<12} {'Funding Rate':>14} {'Annual Return':>14} {'Next Funding':>20}")
    print("-" * 60)

    for _, row in df.iterrows():
        print(f"{row['symbol']:<12} {row['lastFundingRate']*100:>13.4f}% {row['annual_return_pct']:>13.2f}% {row['nextFundingTime'].strftime('%Y-%m-%d %H:%M'):>20}")

    return df


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    if len(sys.argv) < 2:
        print("Usage: python funding_rate.py <command>")
        print("Commands: download, analyze, backtest, monitor")
        sys.exit(1)

    command = sys.argv[1]

    if command == "download":
        run_download()
    elif command == "analyze":
        run_analyze()
    elif command == "backtest":
        symbol = sys.argv[2] if len(sys.argv) > 2 else "BTCUSDT"
        run_backtest(symbol)
    elif command == "monitor":
        run_monitor()
    else:
        print(f"Unknown command: {command}")

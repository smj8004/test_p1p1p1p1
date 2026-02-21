"""
Quant Hybrid Strategy - 수십 년간의 퀀트 리서치 종합

핵심 원리 (학술 연구 기반):
1. Multi-Factor Scoring (Fama-French, AQR)
   - Momentum: Jegadeesh & Titman (1993) - 3-12개월 수익률 지속
   - Volatility: 낮은 변동성 → 낮은 리스크 프리미엄
   - Quality: 안정적인 추세 = 높은 품질

2. Regime Detection (Renaissance, Two Sigma)
   - Hidden Markov Model 기반 시장 상태 감지
   - Trending / Mean-Reverting / Volatile 구분

3. Volatility Targeting (Bridgewater)
   - 목표 변동성 유지로 일관된 리스크
   - 변동성 높을 때 포지션 축소

4. Kelly Criterion Position Sizing
   - 최적 베팅 크기 (Fractional Kelly 0.25-0.5x)
   - 승률과 손익비 기반 계산

5. Crash Risk Management (AQR)
   - 모멘텀 크래시 방지
   - 변동성 스케일링

6. Crypto Adaptations
   - 24/7 시장 특성 반영
   - Funding Rate 통합 (옵션)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal
from enum import Enum


class MarketRegime(Enum):
    """시장 레짐 (Hidden Markov Model 개념)"""
    TRENDING_UP = "trending_up"      # 상승 추세
    TRENDING_DOWN = "trending_down"  # 하락 추세
    MEAN_REVERTING = "mean_reverting"  # 횡보/평균회귀
    HIGH_VOLATILITY = "high_volatility"  # 고변동성 (위험)


@dataclass
class QuantHybridConfig:
    """전략 설정"""
    # Factor Weights (총합 1.0)
    momentum_weight: float = 0.35      # 모멘텀 팩터 가중치
    volatility_weight: float = 0.25    # 변동성 팩터 가중치
    trend_quality_weight: float = 0.25 # 추세 품질 가중치
    mean_reversion_weight: float = 0.15  # 평균회귀 가중치

    # Momentum Parameters (Jegadeesh-Titman)
    momentum_short: int = 5    # 단기 모멘텀 (bars)
    momentum_medium: int = 20  # 중기 모멘텀
    momentum_long: int = 60    # 장기 모멘텀

    # Volatility Parameters
    vol_short: int = 10        # 단기 변동성
    vol_long: int = 50         # 장기 변동성
    target_volatility: float = 0.15  # 목표 연간 변동성 (15%)

    # Regime Detection
    regime_lookback: int = 20  # 레짐 판단 기간
    trend_threshold: float = 0.6  # ADX 기반 추세 임계값
    vol_spike_threshold: float = 2.0  # 변동성 스파이크 배수

    # Position Sizing (Kelly)
    kelly_fraction: float = 0.25  # Fractional Kelly (보수적)
    max_position_pct: float = 0.5  # 최대 포지션 크기 (자본의 50%)
    min_position_pct: float = 0.05  # 최소 포지션 크기

    # Risk Management
    base_sl_atr: float = 2.0   # 기본 손절 ATR 배수
    base_tp_atr: float = 3.0   # 기본 익절 ATR 배수
    max_daily_trades: int = 3  # 일일 최대 거래

    # Signal Thresholds
    entry_score_threshold: float = 0.6   # 진입 신호 임계값
    exit_score_threshold: float = -0.3   # 청산 신호 임계값


class QuantHybridStrategy:
    """
    수십 년간의 퀀트 리서치를 종합한 하이브리드 전략

    주요 구성요소:
    1. 멀티팩터 스코어링 (모멘텀 + 변동성 + 품질 + 평균회귀)
    2. 레짐 감지 (추세/횡보/고변동성)
    3. 변동성 타겟팅 (포지션 사이즈 동적 조정)
    4. 켈리 기준 포지션 사이징
    """

    def __init__(self, config: QuantHybridConfig | None = None):
        self.config = config or QuantHybridConfig()
        self.current_regime = MarketRegime.MEAN_REVERTING
        self.historical_returns: list[float] = []
        self.win_count = 0
        self.loss_count = 0
        self.avg_win = 0.0
        self.avg_loss = 0.0

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """모든 지표 계산"""
        df = df.copy()
        cfg = self.config

        # === 기본 지표 ===
        # ATR (Average True Range)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()

        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # === 1. 모멘텀 팩터 (Jegadeesh-Titman) ===
        # 다중 시간대 모멘텀
        df['mom_short'] = df['close'].pct_change(cfg.momentum_short)
        df['mom_medium'] = df['close'].pct_change(cfg.momentum_medium)
        df['mom_long'] = df['close'].pct_change(cfg.momentum_long)

        # 모멘텀 스코어 (정규화)
        df['momentum_score'] = (
            0.5 * self._normalize(df['mom_short']) +
            0.3 * self._normalize(df['mom_medium']) +
            0.2 * self._normalize(df['mom_long'])
        )

        # === 2. 변동성 팩터 ===
        # 실현 변동성
        df['vol_short'] = df['returns'].rolling(cfg.vol_short).std() * np.sqrt(252 * 24)  # 연환산
        df['vol_long'] = df['returns'].rolling(cfg.vol_long).std() * np.sqrt(252 * 24)

        # 변동성 비율 (낮을수록 좋음 - Low Volatility Anomaly)
        df['vol_ratio'] = df['vol_short'] / df['vol_long'].replace(0, np.nan)

        # 변동성 스코어 (역방향 - 낮은 변동성이 높은 점수)
        df['volatility_score'] = -self._normalize(df['vol_ratio'])

        # === 3. 추세 품질 팩터 ===
        # EMA들
        df['ema_fast'] = df['close'].ewm(span=10).mean()
        df['ema_slow'] = df['close'].ewm(span=30).mean()
        df['ema_trend'] = df['close'].ewm(span=50).mean()

        # 추세 강도 (ADX 대용)
        df['trend_direction'] = (df['ema_fast'] - df['ema_slow']) / df['atr']
        df['trend_consistency'] = df['trend_direction'].rolling(cfg.regime_lookback).apply(
            lambda x: abs(x.mean()) / (x.std() + 1e-8)
        )

        # 추세 품질 스코어
        df['trend_quality_score'] = self._normalize(df['trend_consistency'])

        # === 4. 평균회귀 팩터 ===
        # 볼린저 밴드 위치
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_position'] = (df['close'] - df['bb_mid']) / (2 * df['bb_std'] + 1e-8)

        # RSI
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))

        # 평균회귀 스코어 (극단적일수록 반대 방향)
        df['mean_reversion_score'] = -self._normalize(df['bb_position'])

        # === 5. 레짐 감지 ===
        df['regime'] = df.apply(lambda row: self._detect_regime(row, df), axis=1)

        # === 6. 종합 팩터 스코어 ===
        df['factor_score'] = (
            cfg.momentum_weight * df['momentum_score'] +
            cfg.volatility_weight * df['volatility_score'] +
            cfg.trend_quality_weight * df['trend_quality_score'] +
            cfg.mean_reversion_weight * df['mean_reversion_score']
        )

        # === 7. 레짐별 스코어 조정 ===
        df['adjusted_score'] = df.apply(
            lambda row: self._adjust_score_by_regime(row), axis=1
        )

        # === 8. 변동성 타겟팅 스케일러 ===
        df['vol_scalar'] = cfg.target_volatility / (df['vol_short'] + 1e-8)
        df['vol_scalar'] = df['vol_scalar'].clip(0.2, 3.0)  # 범위 제한

        return df

    def _normalize(self, series: pd.Series, window: int = 100) -> pd.Series:
        """Rolling z-score 정규화"""
        mean = series.rolling(window, min_periods=20).mean()
        std = series.rolling(window, min_periods=20).std()
        return (series - mean) / (std + 1e-8)

    def _detect_regime(self, row: pd.Series, df: pd.DataFrame) -> str:
        """시장 레짐 감지 (HMM 단순화 버전)"""
        cfg = self.config

        # 변동성 스파이크 체크
        if pd.notna(row.get('vol_ratio')) and row['vol_ratio'] > cfg.vol_spike_threshold:
            return MarketRegime.HIGH_VOLATILITY.value

        # 추세 체크
        if pd.notna(row.get('trend_consistency')):
            if row['trend_consistency'] > cfg.trend_threshold:
                if row.get('trend_direction', 0) > 0:
                    return MarketRegime.TRENDING_UP.value
                else:
                    return MarketRegime.TRENDING_DOWN.value

        return MarketRegime.MEAN_REVERTING.value

    def _adjust_score_by_regime(self, row: pd.Series) -> float:
        """레짐에 따른 스코어 조정"""
        regime = row.get('regime', MarketRegime.MEAN_REVERTING.value)
        factor_score = row.get('factor_score', 0)

        if regime == MarketRegime.HIGH_VOLATILITY.value:
            # 고변동성: 진입 신호 억제 (리스크 감소)
            return factor_score * 0.3

        elif regime == MarketRegime.TRENDING_UP.value:
            # 상승 추세: 롱 신호 강화
            if factor_score > 0:
                return factor_score * 1.3
            else:
                return factor_score * 0.7

        elif regime == MarketRegime.TRENDING_DOWN.value:
            # 하락 추세: 숏 신호 강화
            if factor_score < 0:
                return factor_score * 1.3
            else:
                return factor_score * 0.7

        else:  # MEAN_REVERTING
            # 횡보: 평균회귀 가중치 증가
            return factor_score

    def calculate_kelly_fraction(self) -> float:
        """켈리 비율 계산 (동적)"""
        total_trades = self.win_count + self.loss_count

        if total_trades < 20:  # 충분한 데이터 없음
            return self.config.kelly_fraction

        win_rate = self.win_count / total_trades
        if self.avg_loss == 0:
            return self.config.kelly_fraction

        win_loss_ratio = abs(self.avg_win / self.avg_loss) if self.avg_loss != 0 else 1

        # Kelly: f* = (bp - q) / b
        # b = win/loss ratio, p = win rate, q = loss rate
        kelly = (win_loss_ratio * win_rate - (1 - win_rate)) / win_loss_ratio

        # Fractional Kelly (더 보수적)
        kelly = kelly * self.config.kelly_fraction / 0.25  # 기준 대비 조정

        return np.clip(kelly, self.config.min_position_pct, self.config.max_position_pct)

    def calculate_position_size(self, row: pd.Series, capital: float) -> float:
        """포지션 크기 계산 (변동성 타겟팅 + 켈리)"""
        vol_scalar = row.get('vol_scalar', 1.0)
        kelly = self.calculate_kelly_fraction()

        # 기본 포지션 크기 (켈리 기반)
        base_size = capital * kelly

        # 변동성 조정
        adjusted_size = base_size * vol_scalar

        # 범위 제한
        min_size = capital * self.config.min_position_pct
        max_size = capital * self.config.max_position_pct

        return np.clip(adjusted_size, min_size, max_size)

    def generate_signal(self, row: pd.Series) -> int:
        """
        신호 생성

        Returns:
            1: 롱
           -1: 숏
            0: 없음
        """
        cfg = self.config
        score = row.get('adjusted_score', 0)
        regime = row.get('regime', '')

        # 고변동성 레짐에서는 거래 제한
        if regime == MarketRegime.HIGH_VOLATILITY.value:
            return 0

        # 스코어 기반 신호
        if score > cfg.entry_score_threshold:
            return 1  # 롱
        elif score < -cfg.entry_score_threshold:
            return -1  # 숏

        return 0

    def calculate_sl_tp(self, row: pd.Series, direction: int) -> tuple[float, float]:
        """
        손절/익절 계산 (동적 ATR 기반)

        레짐에 따라 SL/TP 조정:
        - 추세장: TP 확대
        - 횡보장: TP 축소, SL 타이트
        """
        cfg = self.config
        atr = row.get('atr', 0)
        price = row['close']
        regime = row.get('regime', '')

        # 기본 ATR 배수
        sl_mult = cfg.base_sl_atr
        tp_mult = cfg.base_tp_atr

        # 레짐별 조정
        if regime == MarketRegime.TRENDING_UP.value and direction == 1:
            tp_mult *= 1.5  # 추세 방향 TP 확대
        elif regime == MarketRegime.TRENDING_DOWN.value and direction == -1:
            tp_mult *= 1.5
        elif regime == MarketRegime.MEAN_REVERTING.value:
            tp_mult *= 0.8  # 횡보장 TP 축소
            sl_mult *= 0.9

        # 변동성 조정 (높은 변동성 = 넓은 SL)
        vol_ratio = row.get('vol_ratio', 1)
        if vol_ratio > 1.5:
            sl_mult *= 1.2

        if direction == 1:  # Long
            sl = price - atr * sl_mult
            tp = price + atr * tp_mult
        else:  # Short
            sl = price + atr * sl_mult
            tp = price - atr * tp_mult

        return sl, tp

    def update_trade_stats(self, pnl_pct: float):
        """거래 통계 업데이트 (켈리 계산용)"""
        self.historical_returns.append(pnl_pct)

        if pnl_pct > 0:
            self.win_count += 1
            # 이동 평균 업데이트
            n = self.win_count
            self.avg_win = self.avg_win * (n - 1) / n + pnl_pct / n
        else:
            self.loss_count += 1
            n = self.loss_count
            self.avg_loss = self.avg_loss * (n - 1) / n + pnl_pct / n


class QuantHybridBacktester:
    """Quant Hybrid 전략 백테스터"""

    def __init__(self, config: QuantHybridConfig | None = None):
        self.config = config or QuantHybridConfig()
        self.strategy = QuantHybridStrategy(config)

    def run(
        self,
        df: pd.DataFrame,
        initial_capital: float = 10000,
        leverage: float = 1.0,
        fee_rate: float = 0.0004
    ) -> dict:
        """백테스트 실행"""
        # 지표 계산
        df = self.strategy.calculate_indicators(df)
        df = df.dropna().reset_index(drop=True)

        if len(df) < 100:
            return {"error": "Insufficient data"}

        # 초기화
        capital = initial_capital
        position = 0  # 0: 없음, 1: 롱, -1: 숏
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        position_size = 0

        trades: list[dict] = []
        equity_curve: list[float] = []
        daily_trades: dict[str, int] = {}

        cfg = self.config

        for i in range(100, len(df)):
            row = df.iloc[i]
            price = row['close']
            date_str = str(row.name)[:10] if hasattr(row.name, '__str__') else str(i)

            # 일일 거래 횟수 체크
            if date_str not in daily_trades:
                daily_trades[date_str] = 0

            # === 포지션 관리 ===
            if position != 0:
                # SL/TP 체크
                if position == 1:
                    if row['low'] <= stop_loss:
                        pnl = (stop_loss - entry_price) / entry_price * leverage
                        fee = fee_rate * 2
                        net_pnl = pnl - fee
                        capital = capital + position_size * net_pnl
                        self.strategy.update_trade_stats(net_pnl * 100)
                        trades.append({
                            'type': 'sl', 'pnl': net_pnl * 100,
                            'regime': row['regime']
                        })
                        position = 0
                    elif row['high'] >= take_profit:
                        pnl = (take_profit - entry_price) / entry_price * leverage
                        fee = fee_rate * 2
                        net_pnl = pnl - fee
                        capital = capital + position_size * net_pnl
                        self.strategy.update_trade_stats(net_pnl * 100)
                        trades.append({
                            'type': 'tp', 'pnl': net_pnl * 100,
                            'regime': row['regime']
                        })
                        position = 0

                elif position == -1:
                    if row['high'] >= stop_loss:
                        pnl = (entry_price - stop_loss) / entry_price * leverage
                        fee = fee_rate * 2
                        net_pnl = pnl - fee
                        capital = capital + position_size * net_pnl
                        self.strategy.update_trade_stats(net_pnl * 100)
                        trades.append({
                            'type': 'sl', 'pnl': net_pnl * 100,
                            'regime': row['regime']
                        })
                        position = 0
                    elif row['low'] <= take_profit:
                        pnl = (entry_price - take_profit) / entry_price * leverage
                        fee = fee_rate * 2
                        net_pnl = pnl - fee
                        capital = capital + position_size * net_pnl
                        self.strategy.update_trade_stats(net_pnl * 100)
                        trades.append({
                            'type': 'tp', 'pnl': net_pnl * 100,
                            'regime': row['regime']
                        })
                        position = 0

            # === 진입 ===
            if position == 0 and daily_trades[date_str] < cfg.max_daily_trades and capital > 0:
                signal = self.strategy.generate_signal(row)

                if signal != 0:
                    position = signal
                    entry_price = price
                    position_size = self.strategy.calculate_position_size(row, capital)
                    stop_loss, take_profit = self.strategy.calculate_sl_tp(row, signal)
                    daily_trades[date_str] += 1
                    trades.append({
                        'type': 'entry',
                        'direction': 'long' if signal == 1 else 'short',
                        'price': price,
                        'regime': row['regime'],
                        'score': row['adjusted_score'],
                        'kelly': self.strategy.calculate_kelly_fraction()
                    })

            equity_curve.append(capital)

            if capital <= 0:
                break

        # === 최종 정리 ===
        if position != 0 and capital > 0:
            final_price = df.iloc[-1]['close']
            pnl = (final_price - entry_price) / entry_price * leverage * position
            fee = fee_rate * 2
            net_pnl = pnl - fee
            capital = capital + position_size * net_pnl
            trades.append({'type': 'final', 'pnl': net_pnl * 100})

        # === 결과 계산 ===
        total_return = (capital - initial_capital) / initial_capital * 100
        days = len(df) / (24 * 60 / 5) if len(df) > 0 else 1  # 5분봉 기준
        annual_return = total_return * 365 / max(days, 1)

        # 승률
        completed_trades = [t for t in trades if 'pnl' in t]
        wins = len([t for t in completed_trades if t['pnl'] > 0])
        win_rate = wins / len(completed_trades) * 100 if completed_trades else 0

        # Max DD
        if equity_curve:
            eq_arr = np.array(equity_curve)
            peak = np.maximum.accumulate(eq_arr)
            dd = (eq_arr - peak) / peak * 100
            max_dd = dd.min()
        else:
            max_dd = 0

        # Sharpe (단순화)
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24 / 0.5)
        else:
            sharpe = 0

        # 레짐별 통계
        regime_stats = {}
        for t in completed_trades:
            regime = t.get('regime', 'unknown')
            if regime not in regime_stats:
                regime_stats[regime] = {'count': 0, 'wins': 0, 'total_pnl': 0}
            regime_stats[regime]['count'] += 1
            regime_stats[regime]['total_pnl'] += t['pnl']
            if t['pnl'] > 0:
                regime_stats[regime]['wins'] += 1

        return {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return_pct': total_return,
            'annual_return_pct': annual_return,
            'max_drawdown_pct': max_dd,
            'sharpe_ratio': sharpe,
            'total_trades': len(completed_trades),
            'win_rate_pct': win_rate,
            'avg_win': self.strategy.avg_win,
            'avg_loss': self.strategy.avg_loss,
            'final_kelly': self.strategy.calculate_kelly_fraction(),
            'regime_stats': regime_stats,
            'equity_curve': equity_curve,
            'trades': trades
        }


def run_quant_hybrid_backtest(
    symbol: str = "BTCUSDT",
    timeframe: str = "4h",
    leverage: float = 1.0
) -> dict:
    """Quant Hybrid 백테스트 실행"""
    from pathlib import Path
    import logging

    logger = logging.getLogger(__name__)

    # 데이터 로드
    data_path = Path("data/futures/clean") / symbol / "ohlcv_1m.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    df = pd.read_parquet(data_path)
    if 'open_time' in df.columns:
        df = df.rename(columns={'open_time': 'timestamp'})
    df = df.set_index('timestamp')

    # 리샘플링
    tf_map = {'5m': '5min', '15m': '15min', '30m': '30min', '1h': '1h', '4h': '4h'}
    resample_freq = tf_map.get(timeframe, '1h')

    df_resampled = df.resample(resample_freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    logger.info(f"Running Quant Hybrid backtest on {len(df_resampled)} bars ({timeframe})")

    # 백테스트
    backtester = QuantHybridBacktester()
    result = backtester.run(df_resampled, leverage=leverage)

    # 결과 출력
    print()
    print("=" * 70)
    print("QUANT HYBRID STRATEGY BACKTEST RESULTS")
    print("=" * 70)
    print()
    print(f"Symbol: {symbol} | Timeframe: {timeframe} | Leverage: {leverage}x")
    print()
    print(f"Initial Capital: ${result['initial_capital']:,.2f}")
    print(f"Final Capital: ${result['final_capital']:,.2f}")
    print()
    print(f"Total Return: {result['total_return_pct']:+.2f}%")
    print(f"Annual Return: {result['annual_return_pct']:+.2f}%")
    print(f"Max Drawdown: {result['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
    print()
    print(f"Total Trades: {result['total_trades']}")
    print(f"Win Rate: {result['win_rate_pct']:.1f}%")
    print(f"Avg Win: {result['avg_win']:+.2f}%")
    print(f"Avg Loss: {result['avg_loss']:.2f}%")
    print(f"Final Kelly Fraction: {result['final_kelly']:.2%}")
    print()
    print("Regime Performance:")
    for regime, stats in result['regime_stats'].items():
        wr = stats['wins'] / stats['count'] * 100 if stats['count'] > 0 else 0
        print(f"  {regime}: {stats['count']} trades, {wr:.1f}% WR, {stats['total_pnl']:+.2f}% total")
    print()

    return result


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # 테스트 실행
    for tf in ["4h", "1h"]:
        for lev in [1, 2]:
            print(f"\n{'='*70}")
            print(f"Testing {tf} with {lev}x leverage")
            print("=" * 70)
            try:
                run_quant_hybrid_backtest(timeframe=tf, leverage=lev)
            except Exception as e:
                print(f"Error: {e}")

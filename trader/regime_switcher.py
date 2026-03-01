"""
Regime-Based Strategy Switcher

시장 상황을 감지하고 최적의 전략을 자동으로 선택합니다.

Regime Detection:
1. UPTREND (상승장) → Breakout Strategy
2. DOWNTREND (하락장) → Cash or Short
3. SIDEWAYS (횡보장) → Mean Reversion
4. HIGH_VOLATILITY (고변동성) → Volatility Adaptive

Based on 54,884 backtest results analysis.
"""

import pandas as pd
import numpy as np
from typing import Literal, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class MarketRegime(Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"


@dataclass
class RegimeSignal:
    regime: MarketRegime
    confidence: float  # 0-1
    recommended_strategy: str
    recommended_params: Dict[str, Any]


class RegimeDetector:
    """
    시장 상황(Regime)을 감지하는 클래스

    사용하는 지표:
    - ADX: 추세 강도
    - ATR: 변동성
    - SMA 기울기: 추세 방향
    - Bollinger Band Width: 변동성 + 횡보 감지
    """

    def __init__(
        self,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        volatility_lookback: int = 20,
        volatility_threshold: float = 1.5,  # ATR이 평균의 1.5배 이상이면 고변동성
        trend_sma_period: int = 50,
    ):
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.volatility_lookback = volatility_lookback
        self.volatility_threshold = volatility_threshold
        self.trend_sma_period = trend_sma_period

    def calculate_adx(self, df: pd.DataFrame) -> pd.Series:
        """ADX (Average Directional Index) 계산"""
        high = df['high']
        low = df['low']
        close = df['close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smoothed averages
        atr = tr.rolling(self.adx_period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(self.adx_period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(self.adx_period).mean() / atr

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(self.adx_period).mean()

        return adx, plus_di, minus_di

    def calculate_volatility_ratio(self, df: pd.DataFrame) -> pd.Series:
        """현재 변동성 / 평균 변동성 비율"""
        high = df['high']
        low = df['low']
        close = df['close']

        # ATR 계산
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(14).mean()
        avg_atr = atr.rolling(self.volatility_lookback * 5).mean()  # 장기 평균

        return atr / (avg_atr + 1e-10)

    def calculate_trend_direction(self, df: pd.DataFrame) -> pd.Series:
        """추세 방향 (-1: 하락, 0: 횡보, 1: 상승)"""
        close = df['close']
        sma = close.rolling(self.trend_sma_period).mean()

        # SMA 기울기 (최근 10개 기간)
        sma_slope = (sma - sma.shift(10)) / sma.shift(10) * 100

        return sma_slope

    def detect_regime(self, df: pd.DataFrame) -> RegimeSignal:
        """
        현재 시장 상황 감지

        로직:
        1. 변동성이 높으면 → HIGH_VOLATILITY
        2. ADX > 25 이고 상승 추세 → UPTREND
        3. ADX > 25 이고 하락 추세 → DOWNTREND
        4. ADX < 25 → SIDEWAYS
        """
        if len(df) < 100:
            raise ValueError("Need at least 100 candles for regime detection")

        # 지표 계산
        adx, plus_di, minus_di = self.calculate_adx(df)
        volatility_ratio = self.calculate_volatility_ratio(df)
        trend_slope = self.calculate_trend_direction(df)

        # 최신 값
        current_adx = adx.iloc[-1]
        current_vol_ratio = volatility_ratio.iloc[-1]
        current_slope = trend_slope.iloc[-1]
        current_plus_di = plus_di.iloc[-1]
        current_minus_di = minus_di.iloc[-1]

        # Regime 판단
        confidence = 0.5

        # 1. 고변동성 체크 (최우선)
        if current_vol_ratio > self.volatility_threshold:
            regime = MarketRegime.HIGH_VOLATILITY
            confidence = min(0.9, 0.5 + (current_vol_ratio - self.volatility_threshold) * 0.2)
            strategy = "volregime_adaptive"
            params = STRATEGY_PARAMS["volregime_adaptive"]

        # 2. 강한 추세 체크
        elif current_adx > self.adx_threshold:
            if current_plus_di > current_minus_di and current_slope > 0:
                regime = MarketRegime.UPTREND
                confidence = min(0.9, 0.5 + (current_adx - self.adx_threshold) / 50)
                strategy = "breakout_atr_channel"
                params = STRATEGY_PARAMS["breakout_atr_channel"]
            else:
                regime = MarketRegime.DOWNTREND
                confidence = min(0.9, 0.5 + (current_adx - self.adx_threshold) / 50)
                strategy = "cash"  # 하락장에서는 현금 보유
                params = {}

        # 3. 횡보장
        else:
            regime = MarketRegime.SIDEWAYS
            confidence = min(0.9, 0.5 + (self.adx_threshold - current_adx) / 50)
            strategy = "meanrev_bollinger"
            params = STRATEGY_PARAMS["meanrev_bollinger"]

        return RegimeSignal(
            regime=regime,
            confidence=confidence,
            recommended_strategy=strategy,
            recommended_params=params
        )


# 백테스트 결과 기반 최적 파라미터
STRATEGY_PARAMS = {
    "breakout_atr_channel": {
        "family": "breakout",
        "type": "atr_channel",
        "timeframe": "1h",
        "params": {
            "sma_period": 20,
            "atr_period": 20,
            "atr_mult": 3.0
        },
        "risk": {
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "leverage": 1
        },
        "expected": {
            "sharpe": 0.95,
            "win_rate": 42.9,
            "annual_return": 89.3
        }
    },
    "meanrev_bollinger": {
        "family": "meanrev",
        "type": "bollinger",
        "timeframe": "4h",
        "params": {
            "bb_period": 30,
            "bb_std": 2.5,
            "rsi_period": 21,
            "rsi_oversold": 30,
            "rsi_overbought": 70
        },
        "risk": {
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.025,
            "leverage": 2
        },
        "expected": {
            "sharpe": 1.01,
            "win_rate": 61.1,
            "annual_return": 44.7
        }
    },
    "volregime_adaptive": {
        "family": "volregime",
        "type": "adaptive",
        "timeframe": "1d",
        "params": {
            "vol_short": 10,
            "vol_long": 50,
            "low_vol_mult": 0.8,
            "high_vol_mult": 1.5,
            "extreme_vol_mult": 2.5,
            "ema_fast": 12,
            "ema_slow": 26
        },
        "risk": {
            "stop_loss_pct": 0.025,
            "take_profit_pct": 0.04,
            "leverage": 2
        },
        "expected": {
            "sharpe": 1.06,
            "win_rate": 63.3,
            "annual_return": 60.9
        }
    }
}


class RegimeSwitcher:
    """
    시장 상황에 따라 전략을 자동으로 전환하는 시스템
    """

    def __init__(self, min_regime_duration: int = 3):
        """
        Args:
            min_regime_duration: 전략 전환 전 최소 유지 기간 (캔들 수)
        """
        self.detector = RegimeDetector()
        self.min_regime_duration = min_regime_duration
        self.current_regime: Optional[MarketRegime] = None
        self.regime_count = 0
        self.active_strategy: Optional[str] = None
        self.history = []

    def update(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        새로운 데이터로 regime 업데이트

        Returns:
            {
                "regime": MarketRegime,
                "strategy": str,
                "params": dict,
                "should_switch": bool,
                "confidence": float
            }
        """
        signal = self.detector.detect_regime(df)

        # Regime 변경 감지
        if signal.regime != self.current_regime:
            self.regime_count = 1
        else:
            self.regime_count += 1

        # 전략 전환 결정
        should_switch = False
        if self.regime_count >= self.min_regime_duration:
            if signal.recommended_strategy != self.active_strategy:
                should_switch = True
                self.active_strategy = signal.recommended_strategy

        self.current_regime = signal.regime

        # 기록
        result = {
            "regime": signal.regime,
            "strategy": signal.recommended_strategy,
            "params": signal.recommended_params,
            "should_switch": should_switch,
            "confidence": signal.confidence,
            "regime_duration": self.regime_count
        }
        self.history.append(result)

        return result

    def get_current_strategy(self) -> Optional[Dict[str, Any]]:
        """현재 활성화된 전략 정보 반환"""
        if self.active_strategy is None:
            return None
        return STRATEGY_PARAMS.get(self.active_strategy)

    def get_regime_stats(self) -> Dict[str, Any]:
        """Regime 통계"""
        if not self.history:
            return {}

        regimes = [h["regime"] for h in self.history]
        regime_counts = {}
        for r in MarketRegime:
            regime_counts[r.value] = regimes.count(r)

        return {
            "total_periods": len(self.history),
            "regime_distribution": regime_counts,
            "current_regime": self.current_regime.value if self.current_regime else None,
            "current_strategy": self.active_strategy,
            "switches": sum(1 for h in self.history if h["should_switch"])
        }


def demo():
    """데모: 실제 데이터로 regime switcher 테스트"""
    print("=" * 60)
    print("REGIME SWITCHER DEMO")
    print("=" * 60)

    # 샘플 데이터 생성 (실제로는 API에서 가져옴)
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=500, freq='1h')

    # 다양한 시장 상황 시뮬레이션
    price = 40000
    prices = [price]

    for i in range(499):
        if i < 100:  # 상승장
            change = np.random.normal(0.001, 0.01)
        elif i < 200:  # 횡보장
            change = np.random.normal(0, 0.005)
        elif i < 300:  # 하락장
            change = np.random.normal(-0.001, 0.01)
        elif i < 400:  # 고변동성
            change = np.random.normal(0, 0.025)
        else:  # 다시 상승
            change = np.random.normal(0.0015, 0.01)

        price = price * (1 + change)
        prices.append(price)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(100, 1000, 500)
    })

    # Regime Switcher 테스트
    switcher = RegimeSwitcher(min_regime_duration=5)

    print("\n시장 상황 변화 감지 중...\n")

    for i in range(100, len(df), 50):
        result = switcher.update(df.iloc[:i])

        if result["should_switch"]:
            print(f"[{df.iloc[i-1]['timestamp'].strftime('%Y-%m-%d %H:%M')}]")
            print(f"  Regime: {result['regime'].value.upper()}")
            print(f"  Strategy: {result['strategy']}")
            print(f"  Confidence: {result['confidence']:.1%}")
            print()

    print("=" * 60)
    print("REGIME STATISTICS")
    print("=" * 60)
    stats = switcher.get_regime_stats()
    print(f"Total periods analyzed: {stats['total_periods']}")
    print(f"Strategy switches: {stats['switches']}")
    print("\nRegime distribution:")
    for regime, count in stats['regime_distribution'].items():
        pct = count / stats['total_periods'] * 100 if stats['total_periods'] > 0 else 0
        print(f"  {regime}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    demo()

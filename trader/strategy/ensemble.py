"""
Ensemble Strategy - 다중 전략 앙상블

핵심 개념:
1. 다중 전략 결합 (Bridgewater 원칙: 10-15개 비상관 수익원)
2. Risk Parity 가중치 (동일 리스크 기여)
3. 신호 컨센서스 (다수결 투표)
4. 동적 전략 가중치 (최근 성과 기반)

연구 기반:
- AQR "Value and Momentum Everywhere" (다중 팩터 분산)
- Bridgewater All Weather (리스크 패리티)
- Two Sigma (앙상블 ML)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable
from enum import Enum


class SignalType(Enum):
    LONG = 1
    SHORT = -1
    NEUTRAL = 0


@dataclass
class EnsembleConfig:
    """앙상블 설정"""
    # 컨센서스 임계값
    consensus_threshold: float = 0.5  # 전략의 50% 이상 동의 시 진입

    # 가중치 조정
    use_performance_weights: bool = True  # 최근 성과 기반 가중치
    performance_lookback: int = 50  # 성과 평가 기간 (거래 수)

    # 리스크 관리
    max_position_pct: float = 0.3  # 최대 포지션 크기
    base_sl_atr: float = 2.0
    base_tp_atr: float = 4.0
    max_daily_trades: int = 3

    # 변동성 타겟팅
    target_volatility: float = 0.15
    vol_lookback: int = 20


@dataclass
class StrategySignal:
    """개별 전략 신호"""
    name: str
    signal: SignalType
    confidence: float  # 0-1 신뢰도
    weight: float = 1.0  # 전략 가중치


class MomentumModule:
    """모멘텀 모듈 (Jegadeesh-Titman)"""

    def __init__(self, periods: list[int] = None):
        self.periods = periods or [5, 10, 20, 50]

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for p in self.periods:
            df[f'mom_{p}'] = df['close'].pct_change(p)

        # 복합 모멘텀 스코어
        weights = [0.4, 0.3, 0.2, 0.1]
        df['momentum_composite'] = sum(
            w * df[f'mom_{p}'].fillna(0)
            for w, p in zip(weights, self.periods)
        )
        return df

    def generate_signal(self, row: pd.Series) -> StrategySignal:
        score = row.get('momentum_composite', 0)
        if pd.isna(score):
            return StrategySignal('Momentum', SignalType.NEUTRAL, 0)

        if score > 0.02:
            return StrategySignal('Momentum', SignalType.LONG, min(abs(score) * 10, 1))
        elif score < -0.02:
            return StrategySignal('Momentum', SignalType.SHORT, min(abs(score) * 10, 1))
        return StrategySignal('Momentum', SignalType.NEUTRAL, 0)


class MeanReversionModule:
    """평균회귀 모듈 (Poterba-Summers, DeBondt-Thaler)"""

    def __init__(self, bb_period: int = 20, rsi_period: int = 14):
        self.bb_period = bb_period
        self.rsi_period = rsi_period

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 볼린저 밴드
        df['bb_mid'] = df['close'].rolling(self.bb_period).mean()
        df['bb_std'] = df['close'].rolling(self.bb_period).std()
        df['bb_zscore'] = (df['close'] - df['bb_mid']) / (df['bb_std'] + 1e-8)

        # RSI
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(self.rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(self.rsi_period).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
        df['rsi_zscore'] = (df['rsi'] - 50) / 25  # 정규화

        # 복합 평균회귀 스코어
        df['mean_rev_composite'] = -0.6 * df['bb_zscore'] - 0.4 * df['rsi_zscore']

        return df

    def generate_signal(self, row: pd.Series) -> StrategySignal:
        score = row.get('mean_rev_composite', 0)
        bb_z = row.get('bb_zscore', 0)

        if pd.isna(score):
            return StrategySignal('MeanReversion', SignalType.NEUTRAL, 0)

        # 극단적 상황에서만 신호
        if bb_z < -2 and score > 0.5:
            return StrategySignal('MeanReversion', SignalType.LONG, min(abs(score), 1))
        elif bb_z > 2 and score < -0.5:
            return StrategySignal('MeanReversion', SignalType.SHORT, min(abs(score), 1))

        return StrategySignal('MeanReversion', SignalType.NEUTRAL, 0)


class TrendFollowModule:
    """추세추종 모듈 (Turtle Trading, Donchian)"""

    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        self.fast_period = fast_period
        self.slow_period = slow_period

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # EMA 크로스
        df['ema_fast'] = df['close'].ewm(span=self.fast_period).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow_period).mean()
        df['ema_diff'] = (df['ema_fast'] - df['ema_slow']) / df['ema_slow']

        # Donchian 채널
        df['donchian_high'] = df['high'].rolling(20).max()
        df['donchian_low'] = df['low'].rolling(20).min()
        df['donchian_mid'] = (df['donchian_high'] + df['donchian_low']) / 2
        df['donchian_pos'] = (df['close'] - df['donchian_mid']) / (
            (df['donchian_high'] - df['donchian_low']) / 2 + 1e-8
        )

        # 추세 강도
        df['trend_strength'] = df['ema_diff'].rolling(10).apply(
            lambda x: abs(x.mean()) / (x.std() + 1e-8)
        )

        return df

    def generate_signal(self, row: pd.Series) -> StrategySignal:
        ema_diff = row.get('ema_diff', 0)
        trend_str = row.get('trend_strength', 0)
        donchian_pos = row.get('donchian_pos', 0)

        if pd.isna(ema_diff) or pd.isna(trend_str):
            return StrategySignal('TrendFollow', SignalType.NEUTRAL, 0)

        # 강한 추세 + 방향 일치
        if trend_str > 1.5 and ema_diff > 0.01 and donchian_pos > 0.5:
            return StrategySignal('TrendFollow', SignalType.LONG, min(trend_str / 3, 1))
        elif trend_str > 1.5 and ema_diff < -0.01 and donchian_pos < -0.5:
            return StrategySignal('TrendFollow', SignalType.SHORT, min(trend_str / 3, 1))

        return StrategySignal('TrendFollow', SignalType.NEUTRAL, 0)


class VolatilityModule:
    """변동성 모듈 (AQR, Bridgewater)"""

    def __init__(self, short_period: int = 10, long_period: int = 50):
        self.short_period = short_period
        self.long_period = long_period

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()

        # 변동성
        df['returns'] = df['close'].pct_change()
        df['vol_short'] = df['returns'].rolling(self.short_period).std() * np.sqrt(252 * 24)
        df['vol_long'] = df['returns'].rolling(self.long_period).std() * np.sqrt(252 * 24)
        df['vol_ratio'] = df['vol_short'] / (df['vol_long'] + 1e-8)

        # 변동성 레짐
        df['vol_expanding'] = df['vol_ratio'] > 1.5
        df['vol_contracting'] = df['vol_ratio'] < 0.7

        return df

    def get_position_scalar(self, row: pd.Series, target_vol: float = 0.15) -> float:
        """변동성 타겟팅 스케일러"""
        vol = row.get('vol_short', target_vol)
        if pd.isna(vol) or vol == 0:
            return 1.0
        return np.clip(target_vol / vol, 0.2, 2.0)


class VolumeModule:
    """거래량 모듈"""

    def __init__(self, period: int = 20):
        self.period = period

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['vol_ma'] = df['volume'].rolling(self.period).mean()
        df['vol_ratio'] = df['volume'] / (df['vol_ma'] + 1e-8)

        # OBV (On-Balance Volume)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        df['obv_ma'] = df['obv'].rolling(self.period).mean()
        df['obv_signal'] = df['obv'] - df['obv_ma']

        return df

    def confirm_signal(self, row: pd.Series, direction: SignalType) -> float:
        """거래량으로 신호 확인 (0-1 승수)"""
        vol_ratio = row.get('vol_ratio', 1)
        obv_signal = row.get('obv_signal', 0)

        if pd.isna(vol_ratio):
            return 1.0

        # 높은 거래량 + OBV 방향 일치 시 신뢰도 증가
        vol_boost = 1.0
        if vol_ratio > 1.5:
            vol_boost = 1.2
        elif vol_ratio < 0.5:
            vol_boost = 0.8

        obv_boost = 1.0
        if direction == SignalType.LONG and obv_signal > 0:
            obv_boost = 1.1
        elif direction == SignalType.SHORT and obv_signal < 0:
            obv_boost = 1.1
        elif (direction == SignalType.LONG and obv_signal < 0) or \
             (direction == SignalType.SHORT and obv_signal > 0):
            obv_boost = 0.9

        return vol_boost * obv_boost


class EnsembleStrategy:
    """앙상블 전략: 다중 모듈 결합"""

    def __init__(self, config: EnsembleConfig | None = None):
        self.config = config or EnsembleConfig()

        # 모듈 초기화
        self.momentum = MomentumModule()
        self.mean_reversion = MeanReversionModule()
        self.trend_follow = TrendFollowModule()
        self.volatility = VolatilityModule()
        self.volume = VolumeModule()

        # 성과 추적
        self.strategy_performance: dict[str, list[float]] = {
            'Momentum': [],
            'MeanReversion': [],
            'TrendFollow': []
        }

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """모든 모듈 지표 계산"""
        df = self.momentum.calculate(df)
        df = self.mean_reversion.calculate(df)
        df = self.trend_follow.calculate(df)
        df = self.volatility.calculate(df)
        df = self.volume.calculate(df)
        return df

    def get_strategy_weights(self) -> dict[str, float]:
        """최근 성과 기반 전략 가중치 계산"""
        cfg = self.config
        weights = {}

        for name, perf in self.strategy_performance.items():
            if not cfg.use_performance_weights or len(perf) < 10:
                weights[name] = 1.0
            else:
                # 최근 성과로 가중치 조정
                recent = perf[-cfg.performance_lookback:]
                win_rate = sum(1 for p in recent if p > 0) / len(recent)
                avg_return = np.mean(recent)

                # 가중치: 승률 * (1 + 평균수익률)
                weights[name] = max(0.3, win_rate * (1 + avg_return / 10))

        # 정규화
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def generate_ensemble_signal(self, row: pd.Series) -> tuple[SignalType, float, dict]:
        """앙상블 신호 생성"""
        cfg = self.config

        # 각 전략 신호 수집
        signals = [
            self.momentum.generate_signal(row),
            self.mean_reversion.generate_signal(row),
            self.trend_follow.generate_signal(row)
        ]

        # 가중치 적용
        weights = self.get_strategy_weights()
        for sig in signals:
            sig.weight = weights.get(sig.name, 1.0)

        # 컨센서스 계산
        long_score = sum(
            s.confidence * s.weight
            for s in signals if s.signal == SignalType.LONG
        )
        short_score = sum(
            s.confidence * s.weight
            for s in signals if s.signal == SignalType.SHORT
        )
        total_weight = sum(s.weight for s in signals)

        # 거래량 확인
        if long_score > short_score:
            vol_mult = self.volume.confirm_signal(row, SignalType.LONG)
            long_score *= vol_mult
        elif short_score > long_score:
            vol_mult = self.volume.confirm_signal(row, SignalType.SHORT)
            short_score *= vol_mult

        # 최종 신호 결정
        signal_info = {
            'long_score': long_score,
            'short_score': short_score,
            'signals': [(s.name, s.signal.name, s.confidence, s.weight) for s in signals],
            'weights': weights
        }

        threshold = cfg.consensus_threshold * total_weight

        if long_score > threshold and long_score > short_score * 1.5:
            return SignalType.LONG, long_score / total_weight, signal_info
        elif short_score > threshold and short_score > long_score * 1.5:
            return SignalType.SHORT, short_score / total_weight, signal_info

        return SignalType.NEUTRAL, 0, signal_info

    def update_performance(self, strategy_name: str, pnl: float):
        """전략 성과 업데이트"""
        if strategy_name in self.strategy_performance:
            self.strategy_performance[strategy_name].append(pnl)


class EnsembleBacktester:
    """앙상블 전략 백테스터"""

    def __init__(self, config: EnsembleConfig | None = None):
        self.config = config or EnsembleConfig()
        self.strategy = EnsembleStrategy(config)

    def run(
        self,
        df: pd.DataFrame,
        initial_capital: float = 10000,
        leverage: float = 1.0,
        fee_rate: float = 0.0004
    ) -> dict:
        """백테스트 실행"""
        cfg = self.config

        # 지표 계산
        df = self.strategy.calculate_all_indicators(df)
        df = df.dropna().reset_index(drop=True)

        if len(df) < 100:
            return {"error": "Insufficient data"}

        # 상태 초기화
        capital = initial_capital
        position = 0
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        position_size = 0
        entry_signals = {}

        trades = []
        equity_curve = []
        daily_trades = {}

        for i in range(50, len(df)):
            row = df.iloc[i]
            price = row['close']
            atr = row.get('atr', price * 0.02)
            date_str = str(i // (24 * 12))  # 가상 날짜

            if date_str not in daily_trades:
                daily_trades[date_str] = 0

            # === 포지션 관리 ===
            if position != 0:
                if position == 1:
                    if row['low'] <= stop_loss:
                        pnl = (stop_loss - entry_price) / entry_price * leverage
                        net_pnl = pnl - fee_rate * 2
                        capital = capital + position_size * net_pnl

                        # 성과 업데이트
                        for name in entry_signals.get('contributing', []):
                            self.strategy.update_performance(name, net_pnl * 100)

                        trades.append({'type': 'sl', 'pnl': net_pnl * 100})
                        position = 0

                    elif row['high'] >= take_profit:
                        pnl = (take_profit - entry_price) / entry_price * leverage
                        net_pnl = pnl - fee_rate * 2
                        capital = capital + position_size * net_pnl

                        for name in entry_signals.get('contributing', []):
                            self.strategy.update_performance(name, net_pnl * 100)

                        trades.append({'type': 'tp', 'pnl': net_pnl * 100})
                        position = 0

                elif position == -1:
                    if row['high'] >= stop_loss:
                        pnl = (entry_price - stop_loss) / entry_price * leverage
                        net_pnl = pnl - fee_rate * 2
                        capital = capital + position_size * net_pnl

                        for name in entry_signals.get('contributing', []):
                            self.strategy.update_performance(name, net_pnl * 100)

                        trades.append({'type': 'sl', 'pnl': net_pnl * 100})
                        position = 0

                    elif row['low'] <= take_profit:
                        pnl = (entry_price - take_profit) / entry_price * leverage
                        net_pnl = pnl - fee_rate * 2
                        capital = capital + position_size * net_pnl

                        for name in entry_signals.get('contributing', []):
                            self.strategy.update_performance(name, net_pnl * 100)

                        trades.append({'type': 'tp', 'pnl': net_pnl * 100})
                        position = 0

            # === 진입 ===
            if position == 0 and daily_trades[date_str] < cfg.max_daily_trades and capital > 0:
                signal, confidence, info = self.strategy.generate_ensemble_signal(row)

                if signal != SignalType.NEUTRAL and confidence > 0.3:
                    position = signal.value
                    entry_price = price

                    # 변동성 타겟팅으로 포지션 크기 결정
                    vol_scalar = self.strategy.volatility.get_position_scalar(
                        row, cfg.target_volatility
                    )
                    base_size = capital * cfg.max_position_pct * confidence
                    position_size = base_size * vol_scalar

                    # SL/TP 설정
                    sl_mult = cfg.base_sl_atr * (1 / (confidence + 0.5))
                    tp_mult = cfg.base_tp_atr * confidence

                    if position == 1:
                        stop_loss = price - atr * sl_mult
                        take_profit = price + atr * tp_mult
                    else:
                        stop_loss = price + atr * sl_mult
                        take_profit = price - atr * tp_mult

                    # 기여 전략 기록
                    contributing = [
                        sig[0] for sig in info['signals']
                        if sig[1] == signal.name and sig[2] > 0
                    ]
                    entry_signals = {'contributing': contributing}
                    daily_trades[date_str] += 1

                    trades.append({
                        'type': 'entry',
                        'direction': 'long' if position == 1 else 'short',
                        'confidence': confidence,
                        'contributing': contributing,
                        'weights': info['weights']
                    })

            equity_curve.append(capital)
            if capital <= 0:
                break

        # 최종 정리
        if position != 0 and capital > 0:
            final_price = df.iloc[-1]['close']
            pnl = (final_price - entry_price) / entry_price * leverage * position
            capital = capital + position_size * (pnl - fee_rate * 2)

        # 결과 계산
        total_return = (capital - initial_capital) / initial_capital * 100
        days = len(df) / (24 * 12)  # 5분봉 기준
        annual_return = total_return * 365 / max(days, 1)

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

        # Sharpe
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24 / 0.5)
        else:
            sharpe = 0

        # 전략별 성과
        strategy_stats = {}
        for name, perfs in self.strategy.strategy_performance.items():
            if perfs:
                strategy_stats[name] = {
                    'trades': len(perfs),
                    'avg_pnl': np.mean(perfs),
                    'win_rate': sum(1 for p in perfs if p > 0) / len(perfs) * 100
                }

        return {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return_pct': total_return,
            'annual_return_pct': annual_return,
            'max_drawdown_pct': max_dd,
            'sharpe_ratio': sharpe,
            'total_trades': len(completed_trades),
            'win_rate_pct': win_rate,
            'strategy_stats': strategy_stats,
            'final_weights': self.strategy.get_strategy_weights(),
            'equity_curve': equity_curve
        }


def run_ensemble_backtest(
    symbol: str = "BTCUSDT",
    timeframe: str = "4h",
    leverage: float = 1.0
) -> dict:
    """앙상블 백테스트 실행"""
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

    logger.info(f"Running Ensemble backtest on {len(df_resampled)} bars ({timeframe})")

    # 백테스트
    backtester = EnsembleBacktester()
    result = backtester.run(df_resampled, leverage=leverage)

    # 결과 출력
    print()
    print("=" * 70)
    print("ENSEMBLE STRATEGY BACKTEST RESULTS")
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
    print()
    print("Strategy Performance:")
    for name, stats in result.get('strategy_stats', {}).items():
        print(f"  {name}: {stats['trades']} trades, {stats['win_rate']:.1f}% WR, {stats['avg_pnl']:+.2f}% avg")
    print()
    print("Final Strategy Weights:")
    for name, weight in result.get('final_weights', {}).items():
        print(f"  {name}: {weight:.2%}")
    print()

    return result


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    for tf in ["4h", "1h"]:
        for lev in [1, 2]:
            print(f"\n{'='*70}")
            print(f"Testing {tf} with {lev}x leverage")
            print("=" * 70)
            try:
                run_ensemble_backtest(timeframe=tf, leverage=lev)
            except Exception as e:
                print(f"Error: {e}")

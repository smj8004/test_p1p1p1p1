"""
7일 장기 분석 시스템 (확장판)

실행: python long_analysis.py
중단: Ctrl+C (중단해도 결과 저장됨)

2월 21일 18:00까지 최대한 많은 분석 실행
"""

import os
import sys
import json
import signal
import logging
import traceback
import atexit
from pathlib import Path
from datetime import datetime, timedelta
from typing import Callable
import itertools
import random

import numpy as np
import pandas as pd

# ============================================================================
# 설정
# ============================================================================

DEADLINE = datetime(2026, 2, 21, 18, 0, 0)

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("data/analysis_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"analysis_{datetime.now():%Y%m%d_%H%M%S}.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# 상태 관리
# ============================================================================

class AnalysisState:
    def __init__(self):
        self.running = True
        self.current_analysis = None
        self.completed_analyses = []
        self.all_results = {}
        self.start_time = datetime.now()

    def should_stop(self) -> bool:
        if not self.running:
            return True
        if datetime.now() >= DEADLINE:
            logger.info("Deadline reached")
            return True
        return False

    def save_all(self):
        if self.all_results:
            save_final_results(self.all_results, self.completed_analyses)

state = AnalysisState()


def signal_handler(signum, frame):
    logger.info("\nInterrupt received. Saving results...")
    state.running = False
    state.save_all()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(lambda: state.save_all() if state.all_results else None)


# ============================================================================
# 유틸리티
# ============================================================================

def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(v) for v in obj]
    return obj


def save_results(name: str, data: dict) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{name}_{timestamp}.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(convert_to_serializable(data), f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {path}")
    return path


def save_final_results(all_results: dict, completed: list):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {
        'analysis_period': {
            'start': str(state.start_time),
            'end': str(datetime.now()),
            'duration_hours': (datetime.now() - state.start_time).total_seconds() / 3600
        },
        'completed_analyses': len(completed),
        'analysis_names': completed,
        'results': all_results
    }

    path = RESULTS_DIR / f"FINAL_SUMMARY_{timestamp}.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(convert_to_serializable(summary), f, indent=2, ensure_ascii=False)
    logger.info(f"Final summary: {path}")

    # 텍스트 요약
    txt_path = RESULTS_DIR / f"FINAL_SUMMARY_{timestamp}.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("ANALYSIS FINAL SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Start: {state.start_time}\n")
        f.write(f"End: {datetime.now()}\n")
        f.write(f"Duration: {(datetime.now() - state.start_time).total_seconds() / 3600:.1f} hours\n")
        f.write(f"Completed: {len(completed)} analyses\n\n")

        # 주요 결과 추출
        f.write("=" * 70 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("=" * 70 + "\n\n")

        for name, result in all_results.items():
            if 'best' in str(result).lower() or 'top' in str(result).lower():
                f.write(f"\n{name}:\n")
                f.write("-" * 40 + "\n")
                if isinstance(result, dict):
                    for k, v in list(result.items())[:5]:
                        f.write(f"  {k}: {str(v)[:100]}\n")

    logger.info(f"Text summary: {txt_path}")


def load_data(symbol: str = "BTCUSDT") -> pd.DataFrame:
    path = Path(f"data/futures/clean/{symbol}/ohlcv_1m.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}")
    df = pd.read_parquet(path)
    if 'open_time' in df.columns:
        df = df.rename(columns={'open_time': 'timestamp'})
    df = df.set_index('timestamp')
    return df


def resample_data(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    tf_map = {'5m': '5min', '15m': '15min', '30m': '30min', '1h': '1h', '4h': '4h', '1d': '1D'}
    freq = tf_map.get(timeframe, timeframe)
    return df.resample(freq).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()


# ============================================================================
# 지표 계산
# ============================================================================

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ATR
    df['tr'] = np.maximum(df['high'] - df['low'],
        np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()

    # EMAs
    for p in [5, 10, 20, 50, 100, 200]:
        df[f'ema_{p}'] = df['close'].ewm(span=p).mean()
        df[f'sma_{p}'] = df['close'].rolling(p).mean()

    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))

    # MACD
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Bollinger
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)

    # Momentum
    for p in [5, 10, 20, 50]:
        df[f'mom_{p}'] = df['close'].pct_change(p)

    # Volatility
    df['returns'] = df['close'].pct_change()
    df['vol_10'] = df['returns'].rolling(10).std() * np.sqrt(252 * 24)
    df['vol_50'] = df['returns'].rolling(50).std() * np.sqrt(252 * 24)
    df['vol_ratio'] = df['vol_10'] / (df['vol_50'] + 1e-8)

    # Volume
    df['vol_ma'] = df['volume'].rolling(20).mean()
    df['vol_ratio_v'] = df['volume'] / (df['vol_ma'] + 1e-8)

    return df


# ============================================================================
# 백테스트 엔진
# ============================================================================

def backtest(df: pd.DataFrame, signal_func: Callable, sl_atr=2.0, tp_atr=3.0,
             leverage=1.0, fee=0.0004, max_daily=3) -> dict:
    df = calculate_indicators(df).dropna().reset_index(drop=True)
    if len(df) < 100:
        return {'annual_return': 0, 'trades': 0, 'error': 'insufficient_data'}

    capital = 10000
    initial = capital
    position = 0
    entry_price = 0
    sl = tp = 0
    trades = []
    equity = []
    daily_trades = {}

    for i in range(50, len(df)):
        row = df.iloc[i]
        price = row['close']
        atr = row['atr'] if not pd.isna(row['atr']) else price * 0.02
        day = i // 288

        if day not in daily_trades:
            daily_trades[day] = 0

        # Exit check
        if position == 1:
            if row['low'] <= sl:
                pnl = (sl - entry_price) / entry_price * leverage - fee * 2
                capital *= (1 + pnl)
                trades.append(pnl * 100)
                position = 0
            elif row['high'] >= tp:
                pnl = (tp - entry_price) / entry_price * leverage - fee * 2
                capital *= (1 + pnl)
                trades.append(pnl * 100)
                position = 0
        elif position == -1:
            if row['high'] >= sl:
                pnl = (entry_price - sl) / entry_price * leverage - fee * 2
                capital *= (1 + pnl)
                trades.append(pnl * 100)
                position = 0
            elif row['low'] <= tp:
                pnl = (entry_price - tp) / entry_price * leverage - fee * 2
                capital *= (1 + pnl)
                trades.append(pnl * 100)
                position = 0

        # Entry
        if position == 0 and daily_trades[day] < max_daily and capital > 0:
            try:
                sig = signal_func(df, i)
            except:
                sig = 0

            if sig == 1:
                position = 1
                entry_price = price
                sl = price - atr * sl_atr
                tp = price + atr * tp_atr
                daily_trades[day] += 1
            elif sig == -1:
                position = -1
                entry_price = price
                sl = price + atr * sl_atr
                tp = price - atr * tp_atr
                daily_trades[day] += 1

        equity.append(capital)
        if capital <= 0:
            break

    if not trades:
        return {'annual_return': 0, 'trades': 0}

    total_ret = (capital - initial) / initial * 100
    days = len(df) / 288
    annual_ret = total_ret * 365 / max(days, 1)
    wins = sum(1 for t in trades if t > 0)

    eq = np.array(equity)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / (peak + 1e-8) * 100
    max_dd = dd.min()

    if len(equity) > 1:
        rets = np.diff(equity) / (np.array(equity[:-1]) + 1e-8)
        sharpe = np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(252 * 288)
    else:
        sharpe = 0

    return {
        'annual_return': annual_ret,
        'total_return': total_ret,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'trades': len(trades),
        'win_rate': wins / len(trades) * 100,
        'avg_trade': np.mean(trades)
    }


# ============================================================================
# 신호 함수 팩토리
# ============================================================================

def make_trend(fast=10, slow=30):
    def sig(df, i):
        row, prev = df.iloc[i], df.iloc[i-1]
        ef, es = row.get(f'ema_{fast}', row['ema_10']), row.get(f'ema_{slow}', row['ema_50'])
        pf, ps = prev.get(f'ema_{fast}', prev['ema_10']), prev.get(f'ema_{slow}', prev['ema_50'])
        if pd.isna(ef) or pd.isna(es): return 0
        if pf <= ps and ef > es: return 1
        if pf >= ps and ef < es: return -1
        return 0
    return sig

def make_momentum(period=20, thresh=0.03):
    def sig(df, i):
        row = df.iloc[i]
        mom = row.get(f'mom_{period}', row.get('mom_20', 0))
        vol = row.get('vol_ratio_v', 1)
        if pd.isna(mom): return 0
        if mom > thresh and vol > 1.2: return 1
        if mom < -thresh and vol > 1.2: return -1
        return 0
    return sig

def make_meanrev(rsi_lo=30, rsi_hi=70):
    def sig(df, i):
        row = df.iloc[i]
        rsi, bb = row['rsi'], row['bb_pct']
        if pd.isna(rsi) or pd.isna(bb): return 0
        if rsi < rsi_lo and bb < 0.1: return 1
        if rsi > rsi_hi and bb > 0.9: return -1
        return 0
    return sig

def make_breakout(squeeze=0.03, spike=1.5):
    def sig(df, i):
        row = df.iloc[i]
        bw, vr = row['bb_width'], row['vol_ratio_v']
        if pd.isna(bw): return 0
        if bw < squeeze and vr > spike:
            if row['close'] > row['bb_upper']: return 1
            if row['close'] < row['bb_lower']: return -1
        return 0
    return sig

def make_macd():
    def sig(df, i):
        row, prev = df.iloc[i], df.iloc[i-1]
        if pd.isna(row['macd']): return 0
        if prev['macd'] <= prev['macd_signal'] and row['macd'] > row['macd_signal'] and row['macd_hist'] > 0:
            return 1
        if prev['macd'] >= prev['macd_signal'] and row['macd'] < row['macd_signal'] and row['macd_hist'] < 0:
            return -1
        return 0
    return sig

def make_combo(weights):
    funcs = {'trend': make_trend(), 'mom': make_momentum(), 'mr': make_meanrev(),
             'vol': make_breakout(), 'macd': make_macd()}
    def sig(df, i):
        score = sum(funcs[k](df, i) * weights.get(k, 0) for k in funcs)
        if score > 0.3: return 1
        if score < -0.3: return -1
        return 0
    return sig


# ============================================================================
# 분석 모듈들 (확장)
# ============================================================================

def analysis_exhaustive_grid(symbol: str, tf: str) -> dict:
    """초대형 파라미터 그리드"""
    logger.info(f"Exhaustive Grid: {symbol} {tf}")
    df = load_data(symbol)
    df = resample_data(df, tf)

    results = []
    count = 0

    # EMA 조합 (확장)
    for fast in range(5, 51, 5):
        for slow in range(20, 201, 10):
            if fast >= slow or state.should_stop():
                continue
            r = backtest(df, make_trend(fast, slow))
            results.append({'type': 'ema', 'fast': fast, 'slow': slow, **r})
            count += 1
            if count % 100 == 0:
                logger.info(f"  Grid progress: {count}")

    # SL/TP 조합 (확장)
    for sl in np.arange(0.5, 4.1, 0.5):
        for tp in np.arange(1.0, 8.1, 0.5):
            if state.should_stop():
                break
            r = backtest(df, make_trend(), sl_atr=sl, tp_atr=tp)
            results.append({'type': 'sl_tp', 'sl': sl, 'tp': tp, **r})

    # RSI 레벨 (확장)
    for lo in range(15, 41, 5):
        for hi in range(60, 86, 5):
            if state.should_stop():
                break
            r = backtest(df, make_meanrev(lo, hi))
            results.append({'type': 'rsi', 'low': lo, 'high': hi, **r})

    # 모멘텀 (확장)
    for period in [5, 10, 15, 20, 30, 50]:
        for thresh in np.arange(0.01, 0.11, 0.01):
            if state.should_stop():
                break
            r = backtest(df, make_momentum(period, thresh))
            results.append({'type': 'mom', 'period': period, 'thresh': thresh, **r})

    # 레버리지 조합
    for lev in [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]:
        if state.should_stop():
            break
        r = backtest(df, make_trend(), leverage=lev)
        results.append({'type': 'leverage', 'leverage': lev, **r})

    # 최적값 찾기
    best = {}
    for r in results:
        t = r.get('type', 'unknown')
        if t not in best or r.get('annual_return', 0) > best[t].get('annual_return', 0):
            best[t] = r

    return {'symbol': symbol, 'tf': tf, 'total_tests': len(results),
            'all_results': results, 'best': best}


def analysis_walk_forward_extended(symbol: str, tf: str) -> dict:
    """확장 Walk-Forward"""
    logger.info(f"Walk-Forward Extended: {symbol} {tf}")
    df = load_data(symbol)
    df = resample_data(df, tf)

    strategies = [
        ('trend_10_30', make_trend(10, 30)),
        ('trend_20_50', make_trend(20, 50)),
        ('trend_5_20', make_trend(5, 20)),
        ('mom_20_3', make_momentum(20, 0.03)),
        ('mom_10_2', make_momentum(10, 0.02)),
        ('mom_30_5', make_momentum(30, 0.05)),
        ('mr_30_70', make_meanrev(30, 70)),
        ('mr_25_75', make_meanrev(25, 75)),
        ('mr_20_80', make_meanrev(20, 80)),
        ('breakout', make_breakout()),
        ('macd', make_macd()),
        ('combo_equal', make_combo({'trend': 0.25, 'mom': 0.25, 'mr': 0.25, 'vol': 0.25})),
        ('combo_trend', make_combo({'trend': 0.5, 'mom': 0.3, 'mr': 0.1, 'vol': 0.1})),
        ('combo_mom', make_combo({'trend': 0.2, 'mom': 0.5, 'mr': 0.15, 'vol': 0.15})),
    ]

    # 다양한 train/test 기간
    configs = [
        (3, 1),  # 3개월 훈련, 1개월 테스트
        (6, 1),  # 6개월 훈련, 1개월 테스트
        (6, 2),  # 6개월 훈련, 2개월 테스트
        (12, 1), # 12개월 훈련, 1개월 테스트
        (12, 3), # 12개월 훈련, 3개월 테스트
    ]

    if 'h' in tf:
        bars_month = 30 * 24 // int(tf.replace('h', ''))
    else:
        bars_month = 30 * 24 * 60 // int(tf.replace('m', ''))

    all_results = {}

    for train_m, test_m in configs:
        if state.should_stop():
            break

        config_name = f"train{train_m}_test{test_m}"
        all_results[config_name] = {}

        train_bars = train_m * bars_month
        test_bars = test_m * bars_month

        for name, func in strategies:
            if state.should_stop():
                break

            results = []
            start = 0

            while start + train_bars + test_bars <= len(df):
                test_df = df.iloc[start + train_bars:start + train_bars + test_bars]
                r = backtest(test_df, func)
                results.append(r.get('annual_return', 0))
                start += test_bars

            if results:
                all_results[config_name][name] = {
                    'folds': len(results),
                    'avg': np.mean(results),
                    'std': np.std(results),
                    'min': min(results),
                    'max': max(results),
                    'positive_ratio': sum(1 for r in results if r > 0) / len(results)
                }

    return {'symbol': symbol, 'tf': tf, 'results': all_results}


def analysis_monte_carlo_extended(symbol: str, tf: str, n_sims: int = 5000) -> dict:
    """확장 Monte Carlo"""
    logger.info(f"Monte Carlo Extended: {symbol} {tf} ({n_sims} sims)")
    df = load_data(symbol)
    df = resample_data(df, tf)
    df = calculate_indicators(df).dropna().reset_index(drop=True)

    strategies = [
        ('trend', make_trend()),
        ('momentum', make_momentum()),
        ('meanrev', make_meanrev()),
        ('combo', make_combo({'trend': 0.3, 'mom': 0.3, 'mr': 0.2, 'vol': 0.2})),
    ]

    results = {}

    for name, func in strategies:
        if state.should_stop():
            break

        # 거래 수집
        trades = []
        pos = 0
        entry = 0

        for i in range(50, len(df)):
            row = df.iloc[i]
            atr = row['atr']
            price = row['close']

            if pos == 1:
                sl, tp = entry - atr * 2, entry + atr * 3
                if row['low'] <= sl:
                    trades.append((sl - entry) / entry * 100)
                    pos = 0
                elif row['high'] >= tp:
                    trades.append((tp - entry) / entry * 100)
                    pos = 0
            elif pos == -1:
                sl, tp = entry + atr * 2, entry - atr * 3
                if row['high'] >= sl:
                    trades.append((entry - sl) / entry * 100)
                    pos = 0
                elif row['low'] <= tp:
                    trades.append((entry - tp) / entry * 100)
                    pos = 0

            if pos == 0:
                sig = func(df, i)
                if sig != 0:
                    pos = sig
                    entry = price

        if not trades:
            continue

        # 실제 수익
        actual = 10000
        for t in trades:
            actual *= (1 + t / 100 - 0.0008)
        actual_ret = (actual - 10000) / 10000 * 100

        # 시뮬레이션
        sim_rets = []
        for _ in range(n_sims):
            if state.should_stop():
                break
            shuffled = trades.copy()
            random.shuffle(shuffled)
            cap = 10000
            for t in shuffled:
                cap *= (1 + t / 100 - 0.0008)
                if cap <= 0:
                    break
            sim_rets.append((cap - 10000) / 10000 * 100)

        if sim_rets:
            pct = sum(1 for s in sim_rets if s < actual_ret) / len(sim_rets) * 100
            results[name] = {
                'actual': actual_ret,
                'sim_mean': np.mean(sim_rets),
                'sim_std': np.std(sim_rets),
                'pct_5': np.percentile(sim_rets, 5),
                'pct_25': np.percentile(sim_rets, 25),
                'pct_50': np.percentile(sim_rets, 50),
                'pct_75': np.percentile(sim_rets, 75),
                'pct_95': np.percentile(sim_rets, 95),
                'actual_percentile': pct,
                'significant': pct > 90 or pct < 10,
                'n_trades': len(trades)
            }

    return {'symbol': symbol, 'tf': tf, 'n_simulations': n_sims, 'results': results}


def analysis_strategy_combinations_exhaustive(symbol: str, tf: str) -> dict:
    """전략 조합 완전 탐색"""
    logger.info(f"Strategy Combinations Exhaustive: {symbol} {tf}")
    df = load_data(symbol)
    df = resample_data(df, tf)

    weight_options = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    results = []
    count = 0

    for w1 in weight_options:
        for w2 in weight_options:
            for w3 in weight_options:
                for w4 in weight_options:
                    for w5 in weight_options:
                        if state.should_stop():
                            break

                        total = w1 + w2 + w3 + w4 + w5
                        if total == 0:
                            continue

                        weights = {
                            'trend': w1 / total,
                            'mom': w2 / total,
                            'mr': w3 / total,
                            'vol': w4 / total,
                            'macd': w5 / total
                        }

                        r = backtest(df, make_combo(weights))
                        results.append({'weights': weights, **r})
                        count += 1

                        if count % 1000 == 0:
                            logger.info(f"  Combinations: {count}")

    results.sort(key=lambda x: x.get('annual_return', 0), reverse=True)

    return {
        'symbol': symbol, 'tf': tf,
        'total_combinations': len(results),
        'top_50': results[:50],
        'bottom_10': results[-10:] if len(results) > 10 else []
    }


def analysis_regime_detailed(symbol: str, tf: str) -> dict:
    """상세 레짐 분석"""
    logger.info(f"Regime Detailed: {symbol} {tf}")
    df = load_data(symbol)
    df = resample_data(df, tf)
    df = calculate_indicators(df).dropna()

    # 레짐 분류 (더 세분화)
    df['regime'] = 'neutral'
    df.loc[(df['mom_50'] > 0.15) & (df['ema_20'] > df['ema_50']), 'regime'] = 'strong_bull'
    df.loc[(df['mom_50'] > 0.05) & (df['mom_50'] <= 0.15) & (df['ema_20'] > df['ema_50']), 'regime'] = 'bull'
    df.loc[(df['mom_50'] < -0.15) & (df['ema_20'] < df['ema_50']), 'regime'] = 'strong_bear'
    df.loc[(df['mom_50'] < -0.05) & (df['mom_50'] >= -0.15) & (df['ema_20'] < df['ema_50']), 'regime'] = 'bear'
    df.loc[df['vol_ratio'] > 2.5, 'regime'] = 'high_vol'
    df.loc[df['vol_ratio'] < 0.5, 'regime'] = 'low_vol'

    strategies = [
        ('trend', make_trend()),
        ('trend_fast', make_trend(5, 20)),
        ('momentum', make_momentum()),
        ('meanrev', make_meanrev()),
        ('breakout', make_breakout()),
        ('combo', make_combo({'trend': 0.3, 'mom': 0.3, 'mr': 0.2, 'vol': 0.2})),
    ]

    results = {}
    regime_stats = df['regime'].value_counts().to_dict()

    for regime in df['regime'].unique():
        if state.should_stop():
            break

        rdf = df[df['regime'] == regime]
        if len(rdf) < 200:
            continue

        results[regime] = {'bars': len(rdf)}
        for name, func in strategies:
            r = backtest(rdf.reset_index(), func)
            results[regime][name] = r

    # 최적 매핑
    optimal = {}
    for reg, strats in results.items():
        if reg == 'bars':
            continue
        best_name, best_ret = None, -999
        for name, r in strats.items():
            if isinstance(r, dict) and r.get('annual_return', -999) > best_ret:
                best_ret = r['annual_return']
                best_name = name
        if best_name:
            optimal[reg] = {'strategy': best_name, 'return': best_ret}

    return {'symbol': symbol, 'tf': tf, 'regime_stats': regime_stats,
            'results': results, 'optimal': optimal}


def analysis_multi_symbol_correlation(symbols: list, tf: str) -> dict:
    """심볼 간 상관관계 및 성과 비교"""
    logger.info(f"Multi-Symbol Correlation: {symbols} {tf}")

    data = {}
    for s in symbols:
        try:
            df = load_data(s)
            df = resample_data(df, tf)
            data[s] = df
        except:
            pass

    if len(data) < 2:
        return {'error': 'Need at least 2 symbols'}

    # 수익률 상관관계
    returns = {}
    for s, df in data.items():
        returns[s] = df['close'].pct_change().dropna()

    ret_df = pd.DataFrame(returns)
    correlation = ret_df.corr().to_dict()

    # 각 심볼별 전략 성과
    strategies = [
        ('trend', make_trend()),
        ('momentum', make_momentum()),
        ('combo', make_combo({'trend': 0.3, 'mom': 0.3, 'mr': 0.2, 'vol': 0.2})),
    ]

    performance = {}
    for s, df in data.items():
        if state.should_stop():
            break
        performance[s] = {}
        for name, func in strategies:
            r = backtest(df, func)
            performance[s][name] = r

    return {'timeframe': tf, 'symbols': list(data.keys()),
            'correlation': correlation, 'performance': performance}


def analysis_time_of_day(symbol: str) -> dict:
    """시간대별 성과 분석"""
    logger.info(f"Time of Day Analysis: {symbol}")
    df = load_data(symbol)

    # 시간대 추출
    df['hour'] = df.index.hour

    results = {}

    for hour in range(24):
        if state.should_stop():
            break

        hour_df = df[df['hour'] == hour]
        if len(hour_df) < 1000:
            continue

        hour_df = resample_data(hour_df, '1h')
        if len(hour_df) < 100:
            continue

        r = backtest(hour_df, make_trend())
        results[f'hour_{hour:02d}'] = r

    # 세션별 (아시아, 유럽, 미국)
    sessions = {
        'asia': list(range(0, 8)),
        'europe': list(range(8, 16)),
        'us': list(range(16, 24))
    }

    for sess, hours in sessions.items():
        if state.should_stop():
            break
        sess_df = df[df['hour'].isin(hours)]
        sess_df = resample_data(sess_df, '1h')
        if len(sess_df) >= 100:
            r = backtest(sess_df, make_trend())
            results[f'session_{sess}'] = r

    return {'symbol': symbol, 'results': results}


def analysis_drawdown_recovery(symbol: str, tf: str) -> dict:
    """드로우다운 복구 분석"""
    logger.info(f"Drawdown Recovery: {symbol} {tf}")
    df = load_data(symbol)
    df = resample_data(df, tf)

    strategies = [
        ('trend', make_trend()),
        ('momentum', make_momentum()),
        ('combo', make_combo({'trend': 0.3, 'mom': 0.3, 'mr': 0.2, 'vol': 0.2})),
    ]

    results = {}

    for name, func in strategies:
        if state.should_stop():
            break

        # 상세 백테스트로 에쿼티 추출
        df_calc = calculate_indicators(df).dropna().reset_index(drop=True)
        capital = 10000
        equity = []
        pos = 0
        entry = 0

        for i in range(50, len(df_calc)):
            row = df_calc.iloc[i]
            atr = row['atr']
            price = row['close']

            if pos == 1:
                if row['low'] <= entry - atr * 2:
                    capital *= (1 + (entry - atr * 2 - entry) / entry - 0.0008)
                    pos = 0
                elif row['high'] >= entry + atr * 3:
                    capital *= (1 + (entry + atr * 3 - entry) / entry - 0.0008)
                    pos = 0
            elif pos == -1:
                if row['high'] >= entry + atr * 2:
                    capital *= (1 + (entry - (entry + atr * 2)) / entry - 0.0008)
                    pos = 0
                elif row['low'] <= entry - atr * 3:
                    capital *= (1 + (entry - (entry - atr * 3)) / entry - 0.0008)
                    pos = 0

            if pos == 0:
                sig = func(df_calc, i)
                if sig != 0:
                    pos = sig
                    entry = price

            equity.append(capital)

        if not equity:
            continue

        eq = np.array(equity)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak * 100

        # 드로우다운 분석
        dd_periods = []
        in_dd = False
        dd_start = 0

        for i, d in enumerate(dd):
            if d < -5 and not in_dd:
                in_dd = True
                dd_start = i
            elif d >= 0 and in_dd:
                dd_periods.append({
                    'start': dd_start,
                    'end': i,
                    'duration': i - dd_start,
                    'max_dd': dd[dd_start:i].min()
                })
                in_dd = False

        results[name] = {
            'final_capital': capital,
            'max_dd': dd.min(),
            'avg_dd': dd.mean(),
            'dd_periods': len(dd_periods),
            'avg_dd_duration': np.mean([p['duration'] for p in dd_periods]) if dd_periods else 0,
            'max_dd_duration': max([p['duration'] for p in dd_periods]) if dd_periods else 0
        }

    return {'symbol': symbol, 'tf': tf, 'results': results}


# ============================================================================
# 메인 실행
# ============================================================================

def run_all():
    logger.info("=" * 70)
    logger.info("EXTENDED LONG-TERM ANALYSIS")
    logger.info(f"Start: {state.start_time}")
    logger.info(f"Deadline: {DEADLINE}")
    logger.info(f"Remaining: {DEADLINE - datetime.now()}")
    logger.info("Ctrl+C to stop and save")
    logger.info("=" * 70)

    symbols = ["BTCUSDT", "ETHUSDT"]
    timeframes = ["4h", "1h", "15m", "30m"]

    available = [s for s in symbols if Path(f"data/futures/clean/{s}/ohlcv_1m.parquet").exists()]
    logger.info(f"Available: {available}")

    # 분석 큐 생성 (시간 오래 걸리는 것 포함)
    queue = []

    for s in available:
        for tf in timeframes:
            queue.append(('exhaustive_grid', s, tf))
            queue.append(('walk_forward_ext', s, tf))
            queue.append(('monte_carlo_ext', s, tf))
            queue.append(('combinations_exhaust', s, tf))
            queue.append(('regime_detailed', s, tf))
            queue.append(('drawdown_recovery', s, tf))
        queue.append(('time_of_day', s, None))

    # 멀티심볼 분석
    if len(available) >= 2:
        for tf in timeframes:
            queue.append(('multi_symbol', available, tf))

    # 추가 반복 (더 많은 Monte Carlo 등)
    for s in available:
        for tf in ['4h', '1h']:
            queue.append(('monte_carlo_10k', s, tf))

    logger.info(f"Total queued: {len(queue)}")

    for idx, item in enumerate(queue):
        if state.should_stop():
            break

        analysis_type = item[0]

        if analysis_type == 'multi_symbol':
            name = f"multi_symbol_{item[2]}"
            args = (item[1], item[2])
        elif item[2] is None:
            name = f"{analysis_type}_{item[1]}"
            args = (item[1],)
        else:
            name = f"{analysis_type}_{item[1]}_{item[2]}"
            args = (item[1], item[2])

        logger.info(f"\n[{idx+1}/{len(queue)}] {name}")
        state.current_analysis = name

        try:
            if analysis_type == 'exhaustive_grid':
                result = analysis_exhaustive_grid(*args)
            elif analysis_type == 'walk_forward_ext':
                result = analysis_walk_forward_extended(*args)
            elif analysis_type == 'monte_carlo_ext':
                result = analysis_monte_carlo_extended(*args)
            elif analysis_type == 'monte_carlo_10k':
                result = analysis_monte_carlo_extended(args[0], args[1], n_sims=10000)
            elif analysis_type == 'combinations_exhaust':
                result = analysis_strategy_combinations_exhaustive(*args)
            elif analysis_type == 'regime_detailed':
                result = analysis_regime_detailed(*args)
            elif analysis_type == 'drawdown_recovery':
                result = analysis_drawdown_recovery(*args)
            elif analysis_type == 'time_of_day':
                result = analysis_time_of_day(*args)
            elif analysis_type == 'multi_symbol':
                result = analysis_multi_symbol_correlation(*args)
            else:
                continue

            state.all_results[name] = result
            state.completed_analyses.append(name)
            save_results(name, result)

        except Exception as e:
            logger.error(f"Error: {e}")
            state.all_results[name] = {'error': str(e)}

    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS COMPLETE")
    logger.info(f"Completed: {len(state.completed_analyses)}/{len(queue)}")
    logger.info(f"Duration: {datetime.now() - state.start_time}")
    logger.info("=" * 70)

    state.save_all()


if __name__ == "__main__":
    run_all()

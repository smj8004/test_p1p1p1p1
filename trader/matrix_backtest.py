"""
Matrix Backtest - 레버리지 x 전략 x 파라미터 조합 테스트

목표: 연 30-50% 수익률 달성 가능한 설정 탐색

테스트 항목:
- 레버리지: 1x, 2x, 3x, 5x, 7x, 10x
- 전략: 추세 추종, 볼륨 브레이크아웃, 변동성 필터, 평균회귀
- 타임프레임: 15m, 1h, 4h
- SL/TP: 다양한 조합
- 거래 빈도: 보수적 ~ 공격적
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable
import json

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """백테스트 설정"""
    initial_capital: float = 10000.0
    leverage: int = 1
    fee_rate: float = 0.0004  # 0.04% taker fee
    slippage_pct: float = 0.01  # 0.01% slippage

    # Risk management
    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 3.0
    max_daily_trades: int = 3

    # Position sizing
    position_pct: float = 0.95  # Use 95% of capital


@dataclass
class BacktestResult:
    """백테스트 결과"""
    strategy: str
    timeframe: str
    leverage: int
    params: dict

    initial_capital: float
    final_capital: float
    total_return_pct: float
    annual_return_pct: float

    total_trades: int
    win_rate: float
    profit_factor: float

    max_drawdown_pct: float
    sharpe_ratio: float

    avg_trade_pct: float
    avg_win_pct: float
    avg_loss_pct: float

    days: int
    trades_per_day: float


class StrategyBase:
    """전략 기본 클래스"""

    name: str = "Base"

    def __init__(self, params: dict = None):
        self.params = params or {}

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """지표 계산"""
        raise NotImplementedError

    def generate_signal(self, df: pd.DataFrame, i: int) -> int:
        """신호 생성: 1=Long, -1=Short, 0=None"""
        raise NotImplementedError


class TrendFollowStrategy(StrategyBase):
    """추세 추종 전략"""

    name = "TrendFollow"

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        fast = self.params.get('ema_fast', 20)
        slow = self.params.get('ema_slow', 50)

        df['ema_fast'] = df['close'].ewm(span=fast).mean()
        df['ema_slow'] = df['close'].ewm(span=slow).mean()

        # ADX
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()

        # +DM, -DM
        df['up_move'] = df['high'] - df['high'].shift(1)
        df['down_move'] = df['low'].shift(1) - df['low']
        df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
        df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)

        df['plus_di'] = 100 * (df['plus_dm'].rolling(14).mean() / df['atr'])
        df['minus_di'] = 100 * (df['minus_dm'].rolling(14).mean() / df['atr'])
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx'] = df['dx'].rolling(14).mean()

        # Volume
        df['vol_ma'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma']

        return df

    def generate_signal(self, df: pd.DataFrame, i: int) -> int:
        if i < 60:
            return 0

        row = df.iloc[i]
        prev = df.iloc[i-1]

        adx_thresh = self.params.get('adx_threshold', 25)
        vol_thresh = self.params.get('vol_threshold', 1.0)

        if pd.isna(row['adx']) or pd.isna(row['vol_ratio']):
            return 0

        # Strong trend + volume
        if row['adx'] > adx_thresh and row['vol_ratio'] > vol_thresh:
            # EMA crossover
            if prev['ema_fast'] <= prev['ema_slow'] and row['ema_fast'] > row['ema_slow']:
                return 1
            elif prev['ema_fast'] >= prev['ema_slow'] and row['ema_fast'] < row['ema_slow']:
                return -1

        return 0


class MomentumStrategy(StrategyBase):
    """모멘텀 전략"""

    name = "Momentum"

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # RSI
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))

        # ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()

        # EMA
        df['ema20'] = df['close'].ewm(span=20).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()

        # Volume
        df['vol_ma'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma']

        # Momentum (ROC)
        period = self.params.get('momentum_period', 10)
        df['momentum'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100

        return df

    def generate_signal(self, df: pd.DataFrame, i: int) -> int:
        if i < 60:
            return 0

        row = df.iloc[i]

        rsi_low = self.params.get('rsi_low', 40)
        rsi_high = self.params.get('rsi_high', 60)
        mom_thresh = self.params.get('momentum_threshold', 1.0)
        vol_thresh = self.params.get('vol_threshold', 1.2)

        if pd.isna(row['rsi']) or pd.isna(row['momentum']):
            return 0

        # Trend filter
        uptrend = row['ema20'] > row['ema50']
        downtrend = row['ema20'] < row['ema50']

        # Volume filter
        vol_ok = row['vol_ratio'] > vol_thresh

        # Momentum signals
        if uptrend and vol_ok and row['momentum'] > mom_thresh and row['rsi'] > rsi_low and row['rsi'] < 70:
            return 1
        elif downtrend and vol_ok and row['momentum'] < -mom_thresh and row['rsi'] < rsi_high and row['rsi'] > 30:
            return -1

        return 0


class VolatilityBreakoutStrategy(StrategyBase):
    """변동성 돌파 전략"""

    name = "VolBreakout"

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
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
        df['atr_pct'] = df['atr'] / df['close'] * 100
        df['atr_ma'] = df['atr_pct'].rolling(20).mean()

        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

        # High volatility indicator
        df['high_vol'] = df['atr_pct'] > df['atr_ma'] * 1.2

        # Previous day high/low (approximate using rolling)
        lookback = self.params.get('breakout_period', 24)  # 24 bars
        df['range_high'] = df['high'].rolling(lookback).max().shift(1)
        df['range_low'] = df['low'].rolling(lookback).min().shift(1)

        # Volume
        df['vol_ma'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma']

        return df

    def generate_signal(self, df: pd.DataFrame, i: int) -> int:
        if i < 60:
            return 0

        row = df.iloc[i]
        prev = df.iloc[i-1]

        vol_mult = self.params.get('vol_multiplier', 1.5)
        require_high_vol = self.params.get('require_high_vol', True)

        if pd.isna(row['range_high']) or pd.isna(row['vol_ratio']):
            return 0

        # Volume spike
        vol_spike = row['vol_ratio'] > vol_mult

        # Volatility condition
        vol_ok = not require_high_vol or row['high_vol']

        # Breakout signals
        if vol_spike and vol_ok:
            if row['close'] > row['range_high'] and prev['close'] <= prev['range_high']:
                return 1
            elif row['close'] < row['range_low'] and prev['close'] >= prev['range_low']:
                return -1

        return 0


class MeanReversionStrategy(StrategyBase):
    """평균 회귀 전략"""

    name = "MeanReversion"

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # RSI
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))

        # ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()

        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']

        # Distance from mean
        df['dist_from_mean'] = (df['close'] - df['bb_mid']) / df['bb_std']

        # Volume
        df['vol_ma'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma']

        # EMA for trend filter
        df['ema50'] = df['close'].ewm(span=50).mean()
        df['ema200'] = df['close'].ewm(span=200).mean()

        return df

    def generate_signal(self, df: pd.DataFrame, i: int) -> int:
        if i < 60:
            return 0

        row = df.iloc[i]
        prev = df.iloc[i-1]

        rsi_oversold = self.params.get('rsi_oversold', 30)
        rsi_overbought = self.params.get('rsi_overbought', 70)
        bb_threshold = self.params.get('bb_threshold', 2.0)
        require_trend = self.params.get('require_trend', True)

        if pd.isna(row['rsi']) or pd.isna(row['dist_from_mean']):
            return 0

        # Trend filter (optional - trade with trend only)
        if require_trend:
            uptrend = row['ema50'] > row['ema200']
            downtrend = row['ema50'] < row['ema200']
        else:
            uptrend = downtrend = True

        # Oversold bounce (long)
        if uptrend and row['rsi'] < rsi_oversold and row['dist_from_mean'] < -bb_threshold:
            if prev['rsi'] < row['rsi']:  # RSI turning up
                return 1

        # Overbought reversal (short)
        if downtrend and row['rsi'] > rsi_overbought and row['dist_from_mean'] > bb_threshold:
            if prev['rsi'] > row['rsi']:  # RSI turning down
                return -1

        return 0


class MatrixBacktester:
    """매트릭스 백테스터"""

    def __init__(self, data_dir: Path = Path("data/futures")):
        self.data_dir = data_dir
        self.results: list[BacktestResult] = []

    def load_data(self, symbol: str = "BTCUSDT") -> pd.DataFrame:
        """1분봉 데이터 로드"""
        path = self.data_dir / "clean" / symbol / "ohlcv_1m.parquet"
        df = pd.read_parquet(path)
        if 'open_time' in df.columns:
            df = df.rename(columns={'open_time': 'timestamp'})
        return df

    def resample_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """데이터 리샘플링"""
        if timeframe == "1m":
            return df

        rule_map = {
            "5m": "5min",
            "15m": "15min",
            "1h": "1h",
            "4h": "4h",
            "1d": "1D"
        }

        rule = rule_map.get(timeframe, timeframe)

        resampled = df.resample(rule, on='timestamp').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        return resampled

    def run_single_backtest(
        self,
        df: pd.DataFrame,
        strategy: StrategyBase,
        config: BacktestConfig,
        timeframe: str
    ) -> BacktestResult:
        """단일 백테스트 실행"""

        # Calculate indicators
        df = strategy.calculate_indicators(df)

        # Initialize
        capital = config.initial_capital
        position = 0  # 1=long, -1=short, 0=flat
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0

        trades = []
        equity_curve = []
        daily_trades = {}

        # Run simulation
        for i in range(60, len(df)):
            row = df.iloc[i]
            price = row['close']
            atr = row['atr'] if 'atr' in row and not pd.isna(row['atr']) else price * 0.01

            date_key = str(row.name.date()) if hasattr(row.name, 'date') else str(i // 100)
            if date_key not in daily_trades:
                daily_trades[date_key] = 0

            # Check SL/TP
            if position == 1:
                if row['low'] <= stop_loss:
                    pnl = (stop_loss - entry_price) / entry_price * config.leverage
                    fee = config.fee_rate * 2 + config.slippage_pct / 100 * 2
                    pnl -= fee
                    capital *= (1 + pnl)
                    trades.append({'pnl': pnl * 100, 'type': 'long_sl'})
                    position = 0
                elif row['high'] >= take_profit:
                    pnl = (take_profit - entry_price) / entry_price * config.leverage
                    fee = config.fee_rate * 2 + config.slippage_pct / 100 * 2
                    pnl -= fee
                    capital *= (1 + pnl)
                    trades.append({'pnl': pnl * 100, 'type': 'long_tp'})
                    position = 0

            elif position == -1:
                if row['high'] >= stop_loss:
                    pnl = (entry_price - stop_loss) / entry_price * config.leverage
                    fee = config.fee_rate * 2 + config.slippage_pct / 100 * 2
                    pnl -= fee
                    capital *= (1 + pnl)
                    trades.append({'pnl': pnl * 100, 'type': 'short_sl'})
                    position = 0
                elif row['low'] <= take_profit:
                    pnl = (entry_price - take_profit) / entry_price * config.leverage
                    fee = config.fee_rate * 2 + config.slippage_pct / 100 * 2
                    pnl -= fee
                    capital *= (1 + pnl)
                    trades.append({'pnl': pnl * 100, 'type': 'short_tp'})
                    position = 0

            # Entry signals
            if position == 0 and daily_trades[date_key] < config.max_daily_trades and capital > 0:
                signal = strategy.generate_signal(df, i)

                if signal == 1:
                    position = 1
                    entry_price = price
                    stop_loss = entry_price - atr * config.sl_atr_mult
                    take_profit = entry_price + atr * config.tp_atr_mult
                    daily_trades[date_key] += 1

                elif signal == -1:
                    position = -1
                    entry_price = price
                    stop_loss = entry_price + atr * config.sl_atr_mult
                    take_profit = entry_price - atr * config.tp_atr_mult
                    daily_trades[date_key] += 1

            equity_curve.append(capital)

            if capital <= 0:
                break

        # Close final position
        if position != 0 and capital > 0:
            final_price = df.iloc[-1]['close']
            if position == 1:
                pnl = (final_price - entry_price) / entry_price * config.leverage
            else:
                pnl = (entry_price - final_price) / entry_price * config.leverage
            pnl -= config.fee_rate * 2
            capital *= (1 + pnl)
            trades.append({'pnl': pnl * 100, 'type': 'final'})

        # Calculate metrics
        days = (df.index[-1] - df.index[0]).days if hasattr(df.index[-1], 'days') else len(df) // 24
        if days == 0:
            days = 1

        total_return = (capital - config.initial_capital) / config.initial_capital * 100
        annual_return = total_return * 365 / days

        wins = [t['pnl'] for t in trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in trades if t['pnl'] <= 0]

        win_rate = len(wins) / len(trades) * 100 if trades else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        avg_trade = np.mean([t['pnl'] for t in trades]) if trades else 0

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Max drawdown
        if equity_curve:
            eq_arr = np.array(equity_curve)
            peak = np.maximum.accumulate(eq_arr)
            dd = (eq_arr - peak) / peak * 100
            max_dd = dd.min()
        else:
            max_dd = 0

        # Sharpe ratio (simplified)
        if trades:
            returns = [t['pnl'] / 100 for t in trades]
            sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
        else:
            sharpe = 0

        return BacktestResult(
            strategy=strategy.name,
            timeframe=timeframe,
            leverage=config.leverage,
            params=strategy.params,
            initial_capital=config.initial_capital,
            final_capital=capital,
            total_return_pct=total_return,
            annual_return_pct=annual_return,
            total_trades=len(trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            avg_trade_pct=avg_trade,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            days=days,
            trades_per_day=len(trades) / days if days > 0 else 0
        )

    def run_matrix_test(
        self,
        symbol: str = "BTCUSDT",
        output_dir: Path = Path("data/matrix_results")
    ) -> pd.DataFrame:
        """전체 매트릭스 테스트 실행"""

        output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        logger.info(f"Loading {symbol} data...")
        df_1m = self.load_data(symbol)
        logger.info(f"Loaded {len(df_1m)} 1m bars")

        # Define test matrix
        leverages = [1, 2, 3, 5, 7, 10]
        timeframes = ["15m", "1h", "4h"]

        # SL/TP combinations
        sl_tp_combos = [
            (1.0, 2.0),  # 1:2
            (1.0, 3.0),  # 1:3
            (1.5, 3.0),  # 1:2
            (1.5, 4.5),  # 1:3
            (2.0, 4.0),  # 1:2
            (2.0, 6.0),  # 1:3
        ]

        # Strategies with parameter variations
        strategies_configs = [
            # TrendFollow variations
            ("TrendFollow", TrendFollowStrategy, {"ema_fast": 10, "ema_slow": 30, "adx_threshold": 20, "vol_threshold": 1.0}),
            ("TrendFollow", TrendFollowStrategy, {"ema_fast": 20, "ema_slow": 50, "adx_threshold": 25, "vol_threshold": 1.0}),
            ("TrendFollow", TrendFollowStrategy, {"ema_fast": 20, "ema_slow": 50, "adx_threshold": 20, "vol_threshold": 1.2}),

            # Momentum variations
            ("Momentum", MomentumStrategy, {"rsi_low": 40, "rsi_high": 60, "momentum_threshold": 1.0, "vol_threshold": 1.2}),
            ("Momentum", MomentumStrategy, {"rsi_low": 35, "rsi_high": 65, "momentum_threshold": 1.5, "vol_threshold": 1.5}),
            ("Momentum", MomentumStrategy, {"rsi_low": 45, "rsi_high": 55, "momentum_threshold": 0.5, "vol_threshold": 1.0}),

            # VolBreakout variations
            ("VolBreakout", VolatilityBreakoutStrategy, {"breakout_period": 24, "vol_multiplier": 1.5, "require_high_vol": True}),
            ("VolBreakout", VolatilityBreakoutStrategy, {"breakout_period": 48, "vol_multiplier": 2.0, "require_high_vol": True}),
            ("VolBreakout", VolatilityBreakoutStrategy, {"breakout_period": 24, "vol_multiplier": 1.5, "require_high_vol": False}),

            # MeanReversion variations
            ("MeanReversion", MeanReversionStrategy, {"rsi_oversold": 30, "rsi_overbought": 70, "bb_threshold": 2.0, "require_trend": True}),
            ("MeanReversion", MeanReversionStrategy, {"rsi_oversold": 25, "rsi_overbought": 75, "bb_threshold": 2.5, "require_trend": True}),
            ("MeanReversion", MeanReversionStrategy, {"rsi_oversold": 30, "rsi_overbought": 70, "bb_threshold": 2.0, "require_trend": False}),
        ]

        # Max daily trades options
        max_trades_options = [2, 3, 5]

        # Calculate total combinations
        total_tests = len(leverages) * len(timeframes) * len(sl_tp_combos) * len(strategies_configs) * len(max_trades_options)
        logger.info(f"Running {total_tests} backtest combinations...")

        results = []
        test_num = 0

        for tf in timeframes:
            logger.info(f"\nResampling to {tf}...")
            df_tf = self.resample_data(df_1m, tf)
            logger.info(f"  {len(df_tf)} bars")

            for strat_name, strat_class, params in strategies_configs:
                strategy = strat_class(params)

                for lev in leverages:
                    for sl_mult, tp_mult in sl_tp_combos:
                        for max_trades in max_trades_options:
                            test_num += 1

                            config = BacktestConfig(
                                leverage=lev,
                                sl_atr_mult=sl_mult,
                                tp_atr_mult=tp_mult,
                                max_daily_trades=max_trades
                            )

                            try:
                                result = self.run_single_backtest(df_tf, strategy, config, tf)
                                results.append(result)

                                if test_num % 100 == 0:
                                    logger.info(f"  Progress: {test_num}/{total_tests} ({test_num/total_tests*100:.1f}%)")

                            except Exception as e:
                                logger.error(f"Error in test {test_num}: {e}")

        # Convert to DataFrame
        results_df = pd.DataFrame([
            {
                "strategy": r.strategy,
                "timeframe": r.timeframe,
                "leverage": r.leverage,
                "sl_mult": r.params.get('sl_atr_mult', config.sl_atr_mult),
                "tp_mult": r.params.get('tp_atr_mult', config.tp_atr_mult),
                "params": json.dumps(r.params),
                "initial_capital": r.initial_capital,
                "final_capital": r.final_capital,
                "total_return_pct": r.total_return_pct,
                "annual_return_pct": r.annual_return_pct,
                "total_trades": r.total_trades,
                "win_rate": r.win_rate,
                "profit_factor": r.profit_factor,
                "max_drawdown_pct": r.max_drawdown_pct,
                "sharpe_ratio": r.sharpe_ratio,
                "avg_trade_pct": r.avg_trade_pct,
                "avg_win_pct": r.avg_win_pct,
                "avg_loss_pct": r.avg_loss_pct,
                "days": r.days,
                "trades_per_day": r.trades_per_day
            }
            for r in results
        ])

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df.to_csv(output_dir / f"matrix_results_{timestamp}.csv", index=False)
        results_df.to_parquet(output_dir / f"matrix_results_{timestamp}.parquet", index=False)

        logger.info(f"\nResults saved to {output_dir}")

        return results_df

    def analyze_results(self, results_df: pd.DataFrame) -> None:
        """결과 분석 및 출력"""

        print("\n" + "=" * 80)
        print("MATRIX BACKTEST RESULTS ANALYSIS")
        print("=" * 80)

        # Filter profitable strategies
        profitable = results_df[results_df['annual_return_pct'] > 0]
        target_30 = results_df[(results_df['annual_return_pct'] >= 30) & (results_df['annual_return_pct'] <= 100)]
        target_50 = results_df[(results_df['annual_return_pct'] >= 50) & (results_df['annual_return_pct'] <= 150)]

        print(f"\nTotal tests: {len(results_df)}")
        print(f"Profitable: {len(profitable)} ({len(profitable)/len(results_df)*100:.1f}%)")
        print(f"30-100% annual: {len(target_30)} ({len(target_30)/len(results_df)*100:.1f}%)")
        print(f"50-150% annual: {len(target_50)} ({len(target_50)/len(results_df)*100:.1f}%)")

        # Best by annual return
        print("\n" + "-" * 80)
        print("TOP 20 BY ANNUAL RETURN (with reasonable drawdown < 50%)")
        print("-" * 80)

        reasonable = results_df[results_df['max_drawdown_pct'] > -50]
        top_20 = reasonable.nlargest(20, 'annual_return_pct')

        print(f"{'Strategy':<15} {'TF':<5} {'Lev':>4} {'Annual%':>10} {'MaxDD%':>8} {'WR%':>6} {'Trades':>7} {'Sharpe':>7}")
        print("-" * 80)
        for _, row in top_20.iterrows():
            print(f"{row['strategy']:<15} {row['timeframe']:<5} {row['leverage']:>4}x "
                  f"{row['annual_return_pct']:>+9.1f}% {row['max_drawdown_pct']:>7.1f}% "
                  f"{row['win_rate']:>5.1f}% {row['total_trades']:>7} {row['sharpe_ratio']:>7.2f}")

        # Best by Sharpe ratio
        print("\n" + "-" * 80)
        print("TOP 20 BY SHARPE RATIO")
        print("-" * 80)

        top_sharpe = reasonable.nlargest(20, 'sharpe_ratio')

        print(f"{'Strategy':<15} {'TF':<5} {'Lev':>4} {'Annual%':>10} {'MaxDD%':>8} {'WR%':>6} {'Sharpe':>7}")
        print("-" * 80)
        for _, row in top_sharpe.iterrows():
            print(f"{row['strategy']:<15} {row['timeframe']:<5} {row['leverage']:>4}x "
                  f"{row['annual_return_pct']:>+9.1f}% {row['max_drawdown_pct']:>7.1f}% "
                  f"{row['win_rate']:>5.1f}% {row['sharpe_ratio']:>7.2f}")

        # Leverage analysis
        print("\n" + "-" * 80)
        print("LEVERAGE ANALYSIS (Average metrics)")
        print("-" * 80)

        lev_analysis = results_df.groupby('leverage').agg({
            'annual_return_pct': 'mean',
            'max_drawdown_pct': 'mean',
            'win_rate': 'mean',
            'sharpe_ratio': 'mean',
            'total_trades': 'mean'
        }).round(2)

        print(f"{'Leverage':>8} {'Avg Annual%':>12} {'Avg MaxDD%':>12} {'Avg WR%':>10} {'Avg Sharpe':>12}")
        print("-" * 60)
        for lev, row in lev_analysis.iterrows():
            print(f"{lev:>7}x {row['annual_return_pct']:>+11.2f}% {row['max_drawdown_pct']:>11.2f}% "
                  f"{row['win_rate']:>9.2f}% {row['sharpe_ratio']:>12.2f}")

        # Strategy analysis
        print("\n" + "-" * 80)
        print("STRATEGY ANALYSIS (Average metrics)")
        print("-" * 80)

        strat_analysis = results_df.groupby('strategy').agg({
            'annual_return_pct': 'mean',
            'max_drawdown_pct': 'mean',
            'win_rate': 'mean',
            'sharpe_ratio': 'mean'
        }).round(2)

        print(f"{'Strategy':<15} {'Avg Annual%':>12} {'Avg MaxDD%':>12} {'Avg WR%':>10} {'Avg Sharpe':>12}")
        print("-" * 65)
        for strat, row in strat_analysis.iterrows():
            print(f"{strat:<15} {row['annual_return_pct']:>+11.2f}% {row['max_drawdown_pct']:>11.2f}% "
                  f"{row['win_rate']:>9.2f}% {row['sharpe_ratio']:>12.2f}")

        # Timeframe analysis
        print("\n" + "-" * 80)
        print("TIMEFRAME ANALYSIS (Average metrics)")
        print("-" * 80)

        tf_analysis = results_df.groupby('timeframe').agg({
            'annual_return_pct': 'mean',
            'max_drawdown_pct': 'mean',
            'win_rate': 'mean',
            'sharpe_ratio': 'mean',
            'trades_per_day': 'mean'
        }).round(2)

        print(f"{'Timeframe':<10} {'Avg Annual%':>12} {'Avg MaxDD%':>12} {'Avg WR%':>10} {'Trades/Day':>12}")
        print("-" * 60)
        for tf, row in tf_analysis.iterrows():
            print(f"{tf:<10} {row['annual_return_pct']:>+11.2f}% {row['max_drawdown_pct']:>11.2f}% "
                  f"{row['win_rate']:>9.2f}% {row['trades_per_day']:>11.2f}")

        # Target 30-50% strategies
        print("\n" + "=" * 80)
        print("STRATEGIES ACHIEVING 30-50% ANNUAL RETURN")
        print("=" * 80)

        target_range = results_df[
            (results_df['annual_return_pct'] >= 30) &
            (results_df['annual_return_pct'] <= 50) &
            (results_df['max_drawdown_pct'] > -30)  # Reasonable DD
        ].sort_values('sharpe_ratio', ascending=False)

        if len(target_range) > 0:
            print(f"\nFound {len(target_range)} configurations in target range with DD < 30%:")
            print()
            for _, row in target_range.head(15).iterrows():
                print(f"  {row['strategy']:<15} {row['timeframe']:<5} {row['leverage']}x | "
                      f"Annual: {row['annual_return_pct']:+.1f}% | DD: {row['max_drawdown_pct']:.1f}% | "
                      f"WR: {row['win_rate']:.1f}% | Sharpe: {row['sharpe_ratio']:.2f}")
        else:
            print("\nNo configurations found in 30-50% range with DD < 30%")
            print("Best available options:")
            best_alt = results_df[results_df['max_drawdown_pct'] > -40].nlargest(10, 'sharpe_ratio')
            for _, row in best_alt.iterrows():
                print(f"  {row['strategy']:<15} {row['timeframe']:<5} {row['leverage']}x | "
                      f"Annual: {row['annual_return_pct']:+.1f}% | DD: {row['max_drawdown_pct']:.1f}%")


def run_matrix_backtest(
    symbol: str = "BTCUSDT",
    output_dir: str = "data/matrix_results"
):
    """CLI 실행 함수"""
    backtester = MatrixBacktester()
    results = backtester.run_matrix_test(symbol, Path(output_dir))
    backtester.analyze_results(results)
    return results


if __name__ == "__main__":
    import sys

    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/matrix_results"

    run_matrix_backtest(symbol, output_dir)

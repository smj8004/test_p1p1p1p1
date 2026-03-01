"""
Regime-Based Strategy Backtest

시장 상황에 따라 전략을 전환하는 시스템의 백테스트
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

# 프로젝트 루트 추가
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from trader.regime_switcher import RegimeSwitcher, MarketRegime, STRATEGY_PARAMS


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    side: str  # 'long' or 'short'
    size: float
    pnl: float
    pnl_pct: float
    strategy: str
    regime: str


class RegimeBacktester:
    """Regime-based 전략 전환 시스템 백테스트"""

    def __init__(
        self,
        initial_capital: float = 10000,
        risk_per_trade: float = 0.02,  # 거래당 자본의 2% 리스크
        commission: float = 0.001,  # 0.1% 수수료
    ):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.commission = commission

    def load_data(self, symbol: str = "BTCUSDT") -> Dict[str, pd.DataFrame]:
        """캐시된 데이터 로드"""
        data_dir = Path(__file__).parent.parent / "data" / "futures" / "clean" / symbol

        # 모든 timeframe 데이터 로드
        dfs = {}
        tf_map = {"1h": "ohlcv_1h.parquet", "4h": "ohlcv_4h.parquet", "1d": None}

        for tf, filename in tf_map.items():
            if filename:
                path = data_dir / filename
            else:
                # 1d는 1h에서 resample
                path = data_dir / "ohlcv_1h.parquet"

            if path.exists():
                df = pd.read_parquet(path)

                # timestamp 컬럼 정리
                if "open_time" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["open_time"], utc=True)
                elif "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

                # 1d인 경우 resample
                if tf == "1d":
                    df = df.set_index("timestamp")
                    df = df.resample("1D").agg({
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum"
                    }).dropna().reset_index()

                df = df.sort_values('timestamp').reset_index(drop=True)
                dfs[tf] = df
                print(f"Loaded {tf}: {len(df)} candles ({df['timestamp'].min().date()} to {df['timestamp'].max().date()})")

        return dfs

    def run_backtest(
        self,
        dfs: Dict[str, pd.DataFrame],
        regime_timeframe: str = "1d"
    ) -> Dict[str, Any]:
        """
        백테스트 실행

        Args:
            dfs: timeframe별 OHLCV 데이터
            regime_timeframe: regime 감지에 사용할 timeframe
        """
        if regime_timeframe not in dfs:
            raise ValueError(f"No data for timeframe {regime_timeframe}")

        regime_df = dfs[regime_timeframe]
        switcher = RegimeSwitcher(min_regime_duration=3)

        # 결과 저장
        trades: List[Trade] = []
        equity_curve = [self.initial_capital]
        capital = self.initial_capital

        # 현재 포지션
        position = None  # {'entry_price', 'side', 'size', 'stop_loss', 'take_profit', 'strategy', 'regime'}

        print("\n" + "=" * 60)
        print("REGIME-BASED BACKTEST RUNNING...")
        print("=" * 60 + "\n")

        # 최소 100개 캔들 필요
        start_idx = 100

        for i in range(start_idx, len(regime_df)):
            current_bar = regime_df.iloc[i]
            current_time = current_bar['timestamp']
            current_price = current_bar['close']

            # Regime 업데이트
            regime_result = switcher.update(regime_df.iloc[:i+1])
            current_regime = regime_result['regime']
            recommended_strategy = regime_result['strategy']

            # 포지션 체크 (Stop Loss / Take Profit)
            if position is not None:
                exit_price = None
                exit_reason = None

                # Stop Loss 체크
                if position['side'] == 'long':
                    if current_bar['low'] <= position['stop_loss']:
                        exit_price = position['stop_loss']
                        exit_reason = 'stop_loss'
                    elif current_bar['high'] >= position['take_profit']:
                        exit_price = position['take_profit']
                        exit_reason = 'take_profit'
                else:  # short
                    if current_bar['high'] >= position['stop_loss']:
                        exit_price = position['stop_loss']
                        exit_reason = 'stop_loss'
                    elif current_bar['low'] <= position['take_profit']:
                        exit_price = position['take_profit']
                        exit_reason = 'take_profit'

                # 전략 전환 시 청산
                if regime_result['should_switch'] and position is not None:
                    exit_price = current_price
                    exit_reason = 'regime_switch'

                # 청산 실행
                if exit_price is not None:
                    if position['side'] == 'long':
                        pnl = (exit_price - position['entry_price']) * position['size']
                        pnl_pct = (exit_price / position['entry_price'] - 1)
                    else:
                        pnl = (position['entry_price'] - exit_price) * position['size']
                        pnl_pct = (position['entry_price'] / exit_price - 1)

                    # 수수료
                    pnl -= exit_price * position['size'] * self.commission

                    capital += pnl

                    trades.append(Trade(
                        entry_time=position['entry_time'],
                        exit_time=current_time,
                        entry_price=position['entry_price'],
                        exit_price=exit_price,
                        side=position['side'],
                        size=position['size'],
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        strategy=position['strategy'],
                        regime=position['regime']
                    ))

                    position = None

            # 새로운 진입 신호 체크
            if position is None and recommended_strategy != 'cash':
                strategy_params = STRATEGY_PARAMS.get(recommended_strategy)
                if strategy_params:
                    risk_params = strategy_params['risk']
                    stop_loss_pct = risk_params['stop_loss_pct']
                    take_profit_pct = risk_params['take_profit_pct']
                    leverage = risk_params['leverage']

                    # 간단한 진입 조건 (실제로는 각 전략의 로직 사용)
                    should_enter = self._check_entry_signal(
                        regime_df.iloc[:i+1],
                        current_regime,
                        recommended_strategy
                    )

                    if should_enter:
                        # Position sizing
                        risk_amount = capital * self.risk_per_trade
                        position_size = (risk_amount / stop_loss_pct) * leverage
                        position_size = min(position_size, capital * 0.95)  # 최대 95% 사용

                        # 수수료
                        capital -= current_price * (position_size / current_price) * self.commission

                        position = {
                            'entry_time': current_time,
                            'entry_price': current_price,
                            'side': 'long',  # 일단 롱만
                            'size': position_size / current_price,
                            'stop_loss': current_price * (1 - stop_loss_pct),
                            'take_profit': current_price * (1 + take_profit_pct),
                            'strategy': recommended_strategy,
                            'regime': current_regime.value
                        }

            equity_curve.append(capital)

        # 마지막 포지션 청산
        if position is not None:
            current_price = regime_df.iloc[-1]['close']
            if position['side'] == 'long':
                pnl = (current_price - position['entry_price']) * position['size']
                pnl_pct = (current_price / position['entry_price'] - 1)
            else:
                pnl = (position['entry_price'] - current_price) * position['size']
                pnl_pct = (position['entry_price'] / current_price - 1)

            capital += pnl
            trades.append(Trade(
                entry_time=position['entry_time'],
                exit_time=regime_df.iloc[-1]['timestamp'],
                entry_price=position['entry_price'],
                exit_price=current_price,
                side=position['side'],
                size=position['size'],
                pnl=pnl,
                pnl_pct=pnl_pct,
                strategy=position['strategy'],
                regime=position['regime']
            ))

        # 결과 계산
        results = self._calculate_results(trades, equity_curve, regime_df)
        results['regime_stats'] = switcher.get_regime_stats()

        return results

    def _check_entry_signal(
        self,
        df: pd.DataFrame,
        regime: MarketRegime,
        strategy: str
    ) -> bool:
        """
        간단한 진입 신호 체크

        실제로는 각 전략의 상세 로직을 구현해야 함
        여기서는 regime에 맞는 기본 조건만 확인
        """
        if len(df) < 50:
            return False

        close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]

        if regime == MarketRegime.UPTREND:
            # 상승 돌파: 이전 고가 돌파
            high_20 = df['high'].iloc[-21:-1].max()
            return close > high_20

        elif regime == MarketRegime.SIDEWAYS:
            # 평균 회귀: 볼린저 하단 터치
            sma = df['close'].rolling(20).mean().iloc[-1]
            std = df['close'].rolling(20).std().iloc[-1]
            lower_band = sma - 2 * std
            return close < lower_band and prev_close >= lower_band

        elif regime == MarketRegime.HIGH_VOLATILITY:
            # 변동성: EMA 크로스
            ema_fast = df['close'].ewm(span=12).mean().iloc[-1]
            ema_slow = df['close'].ewm(span=26).mean().iloc[-1]
            ema_fast_prev = df['close'].ewm(span=12).mean().iloc[-2]
            ema_slow_prev = df['close'].ewm(span=26).mean().iloc[-2]
            return ema_fast > ema_slow and ema_fast_prev <= ema_slow_prev

        return False

    def _calculate_results(
        self,
        trades: List[Trade],
        equity_curve: List[float],
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """결과 통계 계산"""
        if not trades:
            return {
                "total_trades": 0,
                "error": "No trades executed"
            }

        # 기본 통계
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]

        win_rate = len(winning_trades) / total_trades * 100

        total_pnl = sum(t.pnl for t in trades)
        avg_pnl = total_pnl / total_trades

        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0

        profit_factor = abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades and sum(t.pnl for t in losing_trades) != 0 else float('inf')

        # 수익률
        final_equity = equity_curve[-1]
        total_return = (final_equity / self.initial_capital - 1) * 100

        # 기간 계산
        days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days
        annual_return = total_return * (365 / days) if days > 0 else 0

        # 최대 낙폭
        peak = equity_curve[0]
        max_drawdown = 0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_drawdown:
                max_drawdown = dd

        # Sharpe Ratio (간단 계산)
        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(365) if returns.std() > 0 else 0

        # 전략별 성과
        strategy_stats = {}
        for strategy in set(t.strategy for t in trades):
            strat_trades = [t for t in trades if t.strategy == strategy]
            strat_wins = [t for t in strat_trades if t.pnl > 0]
            strategy_stats[strategy] = {
                "trades": len(strat_trades),
                "win_rate": len(strat_wins) / len(strat_trades) * 100 if strat_trades else 0,
                "total_pnl": sum(t.pnl for t in strat_trades),
                "avg_pnl": np.mean([t.pnl for t in strat_trades])
            }

        # Regime별 성과
        regime_stats = {}
        for regime in set(t.regime for t in trades):
            reg_trades = [t for t in trades if t.regime == regime]
            reg_wins = [t for t in reg_trades if t.pnl > 0]
            regime_stats[regime] = {
                "trades": len(reg_trades),
                "win_rate": len(reg_wins) / len(reg_trades) * 100 if reg_trades else 0,
                "total_pnl": sum(t.pnl for t in reg_trades)
            }

        return {
            "initial_capital": self.initial_capital,
            "final_equity": final_equity,
            "total_return_pct": total_return,
            "annual_return_pct": annual_return,
            "max_drawdown_pct": max_drawdown * 100,
            "sharpe_ratio": sharpe,
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_trade_pnl": avg_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "days": days,
            "trades_per_day": total_trades / days if days > 0 else 0,
            "strategy_performance": strategy_stats,
            "regime_performance": regime_stats
        }


def main():
    """메인 실행"""
    print("=" * 70)
    print(" REGIME-BASED STRATEGY BACKTEST")
    print("=" * 70)

    backtester = RegimeBacktester(
        initial_capital=10000,
        risk_per_trade=0.02,
        commission=0.001
    )

    # 데이터 로드
    print("\n[1] Loading data...")
    dfs = backtester.load_data("BTCUSDT")

    if not dfs:
        print("ERROR: No data found. Run data fetcher first.")
        return

    # 백테스트 실행
    print("\n[2] Running backtest...")
    results = backtester.run_backtest(dfs, regime_timeframe="1d")

    # 결과 출력
    print("\n" + "=" * 70)
    print(" BACKTEST RESULTS")
    print("=" * 70)

    print(f"""
    Initial Capital:     ${results['initial_capital']:,.2f}
    Final Equity:        ${results['final_equity']:,.2f}

    Total Return:        {results['total_return_pct']:+.2f}%
    Annual Return:       {results['annual_return_pct']:+.2f}%
    Max Drawdown:        {results['max_drawdown_pct']:.2f}%
    Sharpe Ratio:        {results['sharpe_ratio']:.2f}

    Total Trades:        {results['total_trades']}
    Win Rate:            {results['win_rate']:.1f}%
    Profit Factor:       {results['profit_factor']:.2f}
    Avg Trade PnL:       ${results['avg_trade_pnl']:,.2f}

    Test Period:         {results['days']} days
    Trades/Day:          {results['trades_per_day']:.2f}
    """)

    print("\n" + "-" * 40)
    print(" Performance by Strategy")
    print("-" * 40)
    for strategy, stats in results['strategy_performance'].items():
        print(f"  {strategy}:")
        print(f"    Trades: {stats['trades']} | Win Rate: {stats['win_rate']:.1f}% | PnL: ${stats['total_pnl']:,.2f}")

    print("\n" + "-" * 40)
    print(" Performance by Regime")
    print("-" * 40)
    for regime, stats in results['regime_performance'].items():
        print(f"  {regime}:")
        print(f"    Trades: {stats['trades']} | Win Rate: {stats['win_rate']:.1f}% | PnL: ${stats['total_pnl']:,.2f}")

    # 결과 저장
    output_dir = Path(__file__).parent.parent / "out"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "regime_backtest_results.json"
    with open(output_file, 'w') as f:
        # numpy 타입 변환
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json.dump({k: convert(v) if not isinstance(v, dict) else {kk: convert(vv) for kk, vv in v.items()} for k, v in results.items()}, f, indent=2, default=str)

    print(f"\n Results saved to: {output_file}")


if __name__ == "__main__":
    main()

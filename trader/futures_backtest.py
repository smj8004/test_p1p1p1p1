"""
USDT-M Futures Comprehensive Backtester

Tests all strategy combinations with futures-specific features:
- Funding rate costs (8h intervals)
- Leverage and margin
- Liquidation simulation
- Mark price for PnL
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from trader.logger_utils import get_logger, setup_logging
from trader.strategy.base import Bar, Strategy, StrategyPosition

logger = get_logger(__name__)


@dataclass
class FuturesBacktestConfig:
    """Configuration for futures backtesting."""
    symbol: str = "BTCUSDT"
    data_dir: Path = field(default_factory=lambda: Path("data/futures"))
    output_dir: Path = field(default_factory=lambda: Path("data/futures_backtest"))

    initial_equity: float = 10_000.0

    # Timeframes to test
    timeframes: list[str] = field(default_factory=lambda: ["5m", "15m", "1h", "4h"])

    # Leverage options
    leverages: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])

    # Direction options
    directions: list[str] = field(default_factory=lambda: ["long_only", "long_short"])

    # Stop loss / Take profit options (as percentage)
    stop_losses: list[float] = field(default_factory=lambda: [0.01, 0.02, 0.03, 0.05])
    take_profits: list[float] = field(default_factory=lambda: [0.02, 0.04, 0.06, 0.10])

    # Trading costs
    taker_fee_pct: float = 0.0004  # 0.04%
    maker_fee_pct: float = 0.0002  # 0.02%
    slippage_pct: float = 0.0005   # 0.05%

    # Liquidation
    maintenance_margin_pct: float = 0.005  # 0.5%


@dataclass
class StrategyConfig:
    """Single strategy configuration."""
    name: str
    strategy_type: str
    params: dict[str, Any]
    timeframe: str
    leverage: int
    direction: str
    stop_loss_pct: float
    take_profit_pct: float


@dataclass
class BacktestResult:
    """Result of a single backtest."""
    config: StrategyConfig

    # Performance
    final_equity: float = 0.0
    return_pct: float = 0.0
    total_pnl: float = 0.0
    funding_paid: float = 0.0
    fees_paid: float = 0.0

    # Risk metrics
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0

    # Trade stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0

    # Liquidations
    liquidations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.config.name,
            "strategy_type": self.config.strategy_type,
            "timeframe": self.config.timeframe,
            "leverage": self.config.leverage,
            "direction": self.config.direction,
            "stop_loss_pct": self.config.stop_loss_pct,
            "take_profit_pct": self.config.take_profit_pct,
            "params": json.dumps(self.config.params),
            "final_equity": self.final_equity,
            "return_pct": self.return_pct,
            "total_pnl": self.total_pnl,
            "funding_paid": self.funding_paid,
            "fees_paid": self.fees_paid,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_trade_pnl": self.avg_trade_pnl,
            "liquidations": self.liquidations,
        }


def generate_all_strategy_configs(config: FuturesBacktestConfig) -> list[StrategyConfig]:
    """Generate all strategy combinations to test."""
    strategies = []

    # EMA Cross parameters
    ema_params = []
    for fast in [5, 8, 12, 20]:
        for slow in [20, 26, 50, 100, 200]:
            if fast < slow:
                ema_params.append({"fast_len": fast, "slow_len": slow})

    # RSI parameters
    rsi_params = []
    for period in [7, 14, 21]:
        for oversold in [20, 25, 30]:
            overbought = 100 - oversold
            rsi_params.append({"period": period, "oversold": oversold, "overbought": overbought})

    # MACD parameters
    macd_params = []
    for fast in [8, 12]:
        for slow in [21, 26]:
            for signal in [7, 9]:
                if fast < slow:
                    macd_params.append({"fast_period": fast, "slow_period": slow, "signal_period": signal})

    # Bollinger parameters
    bb_params = []
    for period in [15, 20, 25]:
        for std in [1.5, 2.0, 2.5]:
            for mode in ["mean_reversion", "breakout"]:
                bb_params.append({"period": period, "std_dev": std, "mode": mode})

    all_strategy_params = [
        ("ema_cross", ema_params),
        ("rsi", rsi_params),
        ("macd", macd_params),
        ("bollinger", bb_params),
    ]

    # Generate all combinations
    for strategy_type, param_list in all_strategy_params:
        for params in param_list:
            for timeframe in config.timeframes:
                for leverage in config.leverages:
                    for direction in config.directions:
                        for sl in config.stop_losses:
                            for tp in config.take_profits:
                                # Generate unique name
                                param_str = "_".join(f"{v}" for v in params.values())
                                name = f"{strategy_type}_{param_str}_{timeframe}_L{leverage}_{direction[:4]}_sl{int(sl*100)}_tp{int(tp*100)}"

                                strategies.append(StrategyConfig(
                                    name=name,
                                    strategy_type=strategy_type,
                                    params=params,
                                    timeframe=timeframe,
                                    leverage=leverage,
                                    direction=direction,
                                    stop_loss_pct=sl,
                                    take_profit_pct=tp,
                                ))

    return strategies


def build_strategy(strategy_type: str, params: dict[str, Any], allow_short: bool, sl: float, tp: float) -> Strategy:
    """Build a strategy instance."""
    from trader.strategy.bollinger import BollingerBandStrategy
    from trader.strategy.ema_cross import EMACrossStrategy
    from trader.strategy.macd import MACDStrategy
    from trader.strategy.rsi import RSIStrategy

    if strategy_type == "ema_cross":
        return EMACrossStrategy(
            short_window=params["fast_len"],
            long_window=params["slow_len"],
            allow_short=allow_short,
            stop_loss_pct=sl,
            take_profit_pct=tp,
        )
    elif strategy_type == "rsi":
        return RSIStrategy(
            period=params["period"],
            overbought=params["overbought"],
            oversold=params["oversold"],
            allow_short=allow_short,
            stop_loss_pct=sl,
            take_profit_pct=tp,
        )
    elif strategy_type == "macd":
        return MACDStrategy(
            fast_period=params["fast_period"],
            slow_period=params["slow_period"],
            signal_period=params["signal_period"],
            allow_short=allow_short,
            stop_loss_pct=sl,
            take_profit_pct=tp,
        )
    elif strategy_type == "bollinger":
        return BollingerBandStrategy(
            period=params["period"],
            std_dev=params["std_dev"],
            mode=params["mode"],
            allow_short=allow_short,
            stop_loss_pct=sl,
            take_profit_pct=tp,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_type}")


class FuturesBacktester:
    """
    Futures backtester with realistic simulation.

    Features:
    - Funding rate costs
    - Leverage and margin
    - Liquidation simulation
    - Mark price PnL
    """

    def __init__(self, config: FuturesBacktestConfig) -> None:
        self.config = config
        self.ohlcv_cache: dict[str, pd.DataFrame] = {}
        self.funding_df: pd.DataFrame | None = None

    def load_data(self) -> None:
        """Load all required data."""
        logger.info("Loading data...")

        # Load OHLCV for each timeframe
        for tf in self.config.timeframes:
            path = self.config.data_dir / "clean" / self.config.symbol / f"ohlcv_{tf}.parquet"
            if path.exists():
                self.ohlcv_cache[tf] = pd.read_parquet(path)
                logger.info(f"  Loaded {tf}: {len(self.ohlcv_cache[tf]):,} bars")
            else:
                # Try CSV
                csv_path = path.with_suffix(".csv")
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
                    self.ohlcv_cache[tf] = df
                    logger.info(f"  Loaded {tf}: {len(self.ohlcv_cache[tf]):,} bars")
                else:
                    logger.warning(f"  {tf} data not found!")

        # Load funding rate
        funding_path = self.config.data_dir / "clean" / self.config.symbol / "funding_rate.parquet"
        if funding_path.exists():
            self.funding_df = pd.read_parquet(funding_path)
            logger.info(f"  Loaded funding rate: {len(self.funding_df):,} records")
        else:
            csv_path = funding_path.with_suffix(".csv")
            if csv_path.exists():
                self.funding_df = pd.read_csv(csv_path)
                self.funding_df["fundingTime"] = pd.to_datetime(self.funding_df["fundingTime"], utc=True)
                logger.info(f"  Loaded funding rate: {len(self.funding_df):,} records")

    def get_funding_rate(self, timestamp: pd.Timestamp) -> float:
        """Get funding rate for a given timestamp."""
        if self.funding_df is None or self.funding_df.empty:
            return 0.0

        # Find the most recent funding rate before this timestamp
        mask = self.funding_df["fundingTime"] <= timestamp
        if not mask.any():
            return 0.0

        return float(self.funding_df.loc[mask, "fundingRate"].iloc[-1])

    def is_funding_time(self, timestamp: pd.Timestamp) -> bool:
        """Check if this is a funding time (00:00, 08:00, 16:00 UTC)."""
        return timestamp.hour in [0, 8, 16] and timestamp.minute == 0

    def run_single_backtest(self, strategy_config: StrategyConfig) -> BacktestResult:
        """
        Run a single backtest.

        IMPORTANT: Signals are generated at candle CLOSE, but executed at NEXT candle OPEN.
        This prevents lookahead bias and simulates realistic trading.

        Flow:
        1. Bar[i] closes -> Strategy generates signal
        2. Bar[i+1] opens -> Execute signal at open price + slippage
        """
        df = self.ohlcv_cache.get(strategy_config.timeframe)
        if df is None or df.empty:
            return BacktestResult(config=strategy_config)

        # Build strategy
        allow_short = strategy_config.direction == "long_short"
        strategy = build_strategy(
            strategy_config.strategy_type,
            strategy_config.params,
            allow_short,
            strategy_config.stop_loss_pct,
            strategy_config.take_profit_pct,
        )

        # Initialize state
        equity = self.config.initial_equity
        peak_equity = equity
        max_drawdown = 0.0

        position_side: Literal["flat", "long", "short"] = "flat"
        position_qty = 0.0
        position_entry_price = 0.0

        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        total_profit = 0.0
        total_loss = 0.0
        funding_paid = 0.0
        fees_paid = 0.0
        liquidations = 0

        returns_list = []
        last_equity = equity

        leverage = strategy_config.leverage

        # Pending signal from previous bar (to execute at next bar's open)
        pending_signal: str | None = None

        for _, row in df.iterrows():
            bar = Bar(
                timestamp=row["open_time"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
            )

            # ============================================================
            # STEP 1: Execute pending signal at THIS bar's OPEN price
            # ============================================================
            if pending_signal is not None:
                exec_price = bar.open  # Execute at open of new bar

                # Calculate position size based on leverage
                trade_value = equity * leverage * 0.95
                trade_qty = trade_value / exec_price

                if pending_signal == "long" and position_side != "long":
                    # Close short if exists
                    if position_side == "short":
                        close_price = exec_price * (1 + self.config.slippage_pct)
                        pnl = position_qty * (position_entry_price - close_price)
                        fee = position_qty * close_price * self.config.taker_fee_pct
                        equity += pnl - fee
                        fees_paid += fee

                        total_trades += 1
                        if pnl > 0:
                            winning_trades += 1
                            total_profit += pnl
                        else:
                            losing_trades += 1
                            total_loss += abs(pnl)

                    # Open long
                    entry_price = exec_price * (1 + self.config.slippage_pct)
                    fee = trade_qty * entry_price * self.config.taker_fee_pct
                    equity -= fee
                    fees_paid += fee

                    position_qty = trade_qty
                    position_side = "long"
                    position_entry_price = entry_price

                elif pending_signal == "short" and allow_short and position_side != "short":
                    # Close long if exists
                    if position_side == "long":
                        close_price = exec_price * (1 - self.config.slippage_pct)
                        pnl = position_qty * (close_price - position_entry_price)
                        fee = position_qty * close_price * self.config.taker_fee_pct
                        equity += pnl - fee
                        fees_paid += fee

                        total_trades += 1
                        if pnl > 0:
                            winning_trades += 1
                            total_profit += pnl
                        else:
                            losing_trades += 1
                            total_loss += abs(pnl)

                    # Open short
                    entry_price = exec_price * (1 - self.config.slippage_pct)
                    fee = trade_qty * entry_price * self.config.taker_fee_pct
                    equity -= fee
                    fees_paid += fee

                    position_qty = trade_qty
                    position_side = "short"
                    position_entry_price = entry_price

                elif pending_signal == "exit" and position_qty != 0:
                    # Close position
                    if position_side == "long":
                        close_price = exec_price * (1 - self.config.slippage_pct)
                        pnl = position_qty * (close_price - position_entry_price)
                    else:
                        close_price = exec_price * (1 + self.config.slippage_pct)
                        pnl = position_qty * (position_entry_price - close_price)

                    fee = position_qty * close_price * self.config.taker_fee_pct
                    equity += pnl - fee
                    fees_paid += fee

                    total_trades += 1
                    if pnl > 0:
                        winning_trades += 1
                        total_profit += pnl
                    else:
                        losing_trades += 1
                        total_loss += abs(pnl)

                    position_qty = 0.0
                    position_side = "flat"
                    position_entry_price = 0.0

                pending_signal = None  # Clear after execution

            # ============================================================
            # STEP 2: Check liquidation using bar's LOW/HIGH
            # ============================================================
            if position_qty != 0:
                # Check worst price during this bar
                check_price = bar.low if position_side == "long" else bar.high

                if position_side == "long":
                    unrealized_pnl = position_qty * (check_price - position_entry_price)
                else:
                    unrealized_pnl = position_qty * (position_entry_price - check_price)

                position_value = position_qty * position_entry_price
                maintenance_margin = position_value * self.config.maintenance_margin_pct

                if equity + unrealized_pnl < maintenance_margin:
                    # Liquidated!
                    liquidations += 1
                    liq_price = check_price
                    fee = position_qty * liq_price * self.config.taker_fee_pct
                    equity = max(0, equity + unrealized_pnl - fee)
                    fees_paid += fee

                    total_trades += 1
                    losing_trades += 1
                    total_loss += abs(unrealized_pnl)

                    position_qty = 0.0
                    position_side = "flat"
                    position_entry_price = 0.0

            # ============================================================
            # STEP 3: Apply funding rate (at funding times)
            # ============================================================
            if position_qty != 0 and self.is_funding_time(bar.timestamp):
                funding_rate = self.get_funding_rate(bar.timestamp)
                position_value = position_qty * bar.close

                if position_side == "long":
                    funding_cost = position_value * funding_rate
                else:
                    funding_cost = -position_value * funding_rate

                equity -= funding_cost
                funding_paid += funding_cost

            # ============================================================
            # STEP 4: Generate signal at bar CLOSE (for next bar execution)
            # ============================================================
            position = StrategyPosition(
                side=position_side,
                qty=abs(position_qty),
                entry_price=position_entry_price,
            )
            signal = strategy.on_bar(bar, position=position)

            # Store as pending signal (will execute at next bar's open)
            if signal in ["long", "short", "exit"]:
                pending_signal = signal

            # ============================================================
            # STEP 5: Track equity and drawdown
            # ============================================================
            current_price = bar.close
            if position_qty != 0:
                if position_side == "long":
                    unrealized = position_qty * (current_price - position_entry_price)
                else:
                    unrealized = position_qty * (position_entry_price - current_price)
                total_equity = equity + unrealized
            else:
                total_equity = equity

            peak_equity = max(peak_equity, total_equity)
            if peak_equity > 0:
                dd = (peak_equity - total_equity) / peak_equity
                max_drawdown = max(max_drawdown, dd)

            if last_equity > 0:
                ret = (total_equity - last_equity) / last_equity
                returns_list.append(ret)
            last_equity = total_equity

        # ============================================================
        # Close any remaining position at last bar's close
        # ============================================================
        if position_qty != 0:
            current_price = df.iloc[-1]["close"]
            if position_side == "long":
                close_price = current_price * (1 - self.config.slippage_pct)
                pnl = position_qty * (close_price - position_entry_price)
            else:
                close_price = current_price * (1 + self.config.slippage_pct)
                pnl = position_qty * (position_entry_price - close_price)

            fee = position_qty * close_price * self.config.taker_fee_pct
            equity += pnl - fee
            fees_paid += fee

            total_trades += 1
            if pnl > 0:
                winning_trades += 1
                total_profit += pnl
            else:
                losing_trades += 1
                total_loss += abs(pnl)

        # Calculate metrics
        return_pct = (equity - self.config.initial_equity) / self.config.initial_equity * 100
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        profit_factor = total_profit / abs(total_loss) if total_loss != 0 else float('inf')
        avg_trade_pnl = (equity - self.config.initial_equity) / total_trades if total_trades > 0 else 0

        # Sharpe ratio (annualized, assuming daily returns)
        if returns_list:
            import numpy as np
            returns_arr = np.array(returns_list)
            if returns_arr.std() > 0:
                sharpe = (returns_arr.mean() / returns_arr.std()) * np.sqrt(252)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        return BacktestResult(
            config=strategy_config,
            final_equity=equity,
            return_pct=return_pct,
            total_pnl=equity - self.config.initial_equity,
            funding_paid=funding_paid,
            fees_paid=fees_paid,
            max_drawdown_pct=max_drawdown * 100,
            sharpe_ratio=sharpe,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor if profit_factor != float('inf') else 999.99,
            avg_trade_pnl=avg_trade_pnl,
            liquidations=liquidations,
        )

    def run_all(self) -> pd.DataFrame:
        """Run all backtests."""
        setup_logging(level="INFO")

        self.load_data()

        # Generate all strategy configs
        strategy_configs = generate_all_strategy_configs(self.config)
        total = len(strategy_configs)

        logger.info("")
        logger.info("=" * 70)
        logger.info("FUTURES COMPREHENSIVE BACKTESTER")
        logger.info("=" * 70)
        logger.info(f"  Symbol:           {self.config.symbol}")
        logger.info(f"  Timeframes:       {', '.join(self.config.timeframes)}")
        logger.info(f"  Leverages:        {self.config.leverages}")
        logger.info(f"  Total strategies: {total:,}")
        logger.info(f"  Initial equity:   ${self.config.initial_equity:,.2f}")
        logger.info("=" * 70)
        logger.info("")

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        start_time = time.monotonic()

        for i, sc in enumerate(strategy_configs, 1):
            result = self.run_single_backtest(sc)
            results.append(result.to_dict())

            # Progress every 1%
            if i % max(1, total // 100) == 0 or i == total:
                elapsed = time.monotonic() - start_time
                pct = i / total * 100
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total - i) / rate if rate > 0 else 0

                logger.info(
                    f"Progress: {pct:5.1f}% | "
                    f"{i:,}/{total:,} | "
                    f"{rate:.1f}/sec | "
                    f"ETA: {eta/60:.1f}min"
                )

        # Create DataFrame and save
        df = pd.DataFrame(results)
        df = df.sort_values("return_pct", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)

        # Save results
        output_file = self.config.output_dir / "results.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved: {output_file}")

        # Save top 100 separately
        top_file = self.config.output_dir / "top_100.csv"
        df.head(100).to_csv(top_file, index=False)
        logger.info(f"Top 100 saved: {top_file}")

        # Print summary
        elapsed = time.monotonic() - start_time
        logger.info("")
        logger.info("=" * 70)
        logger.info("BACKTEST COMPLETE")
        logger.info("=" * 70)
        logger.info(f"  Total time:       {elapsed/60:.1f} minutes")
        logger.info(f"  Strategies:       {total:,}")
        logger.info(f"  Speed:            {total/elapsed:.1f} strategies/sec")
        logger.info("")

        # Top 10
        logger.info("TOP 10 STRATEGIES:")
        logger.info("-" * 70)
        for _, row in df.head(10).iterrows():
            logger.info(
                f"  {row['rank']:2}. {row['name'][:40]:<40} "
                f"Return: {row['return_pct']:+8.2f}% | "
                f"DD: {row['max_drawdown_pct']:5.1f}% | "
                f"Sharpe: {row['sharpe_ratio']:5.2f}"
            )
        logger.info("=" * 70)

        # Stats
        profitable = (df["return_pct"] > 0).sum()
        logger.info(f"  Profitable:       {profitable:,}/{total:,} ({profitable/total*100:.1f}%)")
        logger.info(f"  Best return:      {df['return_pct'].max():+.2f}%")
        logger.info(f"  Worst return:     {df['return_pct'].min():+.2f}%")
        logger.info(f"  Avg return:       {df['return_pct'].mean():+.2f}%")
        logger.info(f"  Total liquidations: {df['liquidations'].sum():,}")
        logger.info("=" * 70)

        return df


def run_futures_backtest(
    symbol: str = "BTCUSDT",
    data_dir: str = "data/futures",
    output_dir: str = "data/futures_backtest",
    initial_equity: float = 10_000.0,
    timeframes: str = "5m,15m,1h,4h",
    leverages: str = "1,2,3,5,10",
) -> None:
    """Run comprehensive futures backtest."""
    config = FuturesBacktestConfig(
        symbol=symbol,
        data_dir=Path(data_dir),
        output_dir=Path(output_dir),
        initial_equity=initial_equity,
        timeframes=[t.strip() for t in timeframes.split(",")],
        leverages=[int(l.strip()) for l in leverages.split(",")],
    )

    backtester = FuturesBacktester(config)
    backtester.run_all()

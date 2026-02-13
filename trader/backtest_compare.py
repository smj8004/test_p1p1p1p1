"""
Multi-Strategy Historical Backtester

Runs multiple trading strategies on historical data to find the best performer.
Much faster than real-time testing - can test years of data in minutes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from trader.data.historical import HistoricalDataDownloader
from trader.logging import get_logger, setup_logging
from trader.multi_strategy_daemon import (
    MultiStrategyConfig,
    StrategyConfig,
    StrategyPerformance,
    generate_strategy_matrix,
)
from trader.strategy.base import Bar, Strategy, StrategyPosition

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for historical backtesting."""
    symbol: str = "BTC/USDT"
    timeframe: str = "1m"
    days: int = 365
    initial_equity: float = 10_000.0
    trade_size_usdt: float = 1_000.0
    data_dir: Path = field(default_factory=lambda: Path("data/backtest"))
    cache_dir: Path = field(default_factory=lambda: Path("data/historical"))
    slippage_bps: float = 5.0
    fee_bps: float = 4.0


class MultiStrategyBacktester:
    """
    Backtests multiple strategies on historical data.

    Features:
    - Fast parallel-like execution on historical data
    - Same metrics as live trading daemon
    - Detailed performance reports
    - CSV export for analysis
    """

    def __init__(
        self,
        config: BacktestConfig,
        strategy_configs: list[StrategyConfig] | None = None,
    ) -> None:
        self.config = config
        self.strategy_configs = strategy_configs or generate_strategy_matrix()

        self._strategies: dict[str, Strategy] = {}
        self._performances: dict[str, StrategyPerformance] = {}
        self._bars_processed = 0
        self._started_at: datetime | None = None

        # Initialize strategies and performance trackers
        for sc in self.strategy_configs:
            strategy = self._build_strategy(sc.strategy_type, sc.params)
            self._strategies[sc.name] = strategy
            self._performances[sc.name] = StrategyPerformance(
                name=sc.name,
                strategy_type=sc.strategy_type,
                params=sc.params,
                initial_equity=config.initial_equity,
                current_equity=config.initial_equity,
                peak_equity=config.initial_equity,
            )

        logger.info(f"Initialized {len(self._strategies)} strategies for backtesting")

    def _build_strategy(self, strategy_type: str, params: dict[str, Any]) -> Strategy:
        """Build a strategy instance."""
        from trader.strategy.bollinger import BollingerBandStrategy
        from trader.strategy.ema_cross import EMACrossStrategy
        from trader.strategy.macd import MACDStrategy
        from trader.strategy.rsi import RSIStrategy

        stop_loss = float(params.get("stop_loss_pct", 0.02))
        take_profit = float(params.get("take_profit_pct", 0.04))
        allow_short = bool(params.get("allow_short", True))

        if strategy_type == "ema_cross":
            return EMACrossStrategy(
                short_window=int(params.get("fast_len", 12)),
                long_window=int(params.get("slow_len", 26)),
                allow_short=allow_short,
                stop_loss_pct=stop_loss,
                take_profit_pct=take_profit,
            )
        if strategy_type == "rsi":
            return RSIStrategy(
                period=int(params.get("period", 14)),
                overbought=float(params.get("overbought", 70)),
                oversold=float(params.get("oversold", 30)),
                allow_short=allow_short,
                stop_loss_pct=stop_loss,
                take_profit_pct=take_profit,
            )
        if strategy_type == "macd":
            return MACDStrategy(
                fast_period=int(params.get("fast_period", 12)),
                slow_period=int(params.get("slow_period", 26)),
                signal_period=int(params.get("signal_period", 9)),
                allow_short=allow_short,
                stop_loss_pct=stop_loss,
                take_profit_pct=take_profit,
            )
        if strategy_type == "bollinger":
            return BollingerBandStrategy(
                period=int(params.get("period", 20)),
                std_dev=float(params.get("std_dev", 2.0)),
                mode=params.get("mode", "mean_reversion"),
                allow_short=allow_short,
                stop_loss_pct=stop_loss,
                take_profit_pct=take_profit,
            )
        raise ValueError(f"Unknown strategy type: {strategy_type}")

    def _simulate_fill(self, price: float, qty: float, side: str) -> tuple[float, float, float]:
        """Simulate order fill with slippage and fees."""
        slippage_mult = 1 + (self.config.slippage_bps / 10000) if side == "buy" else 1 - (self.config.slippage_bps / 10000)
        fill_price = price * slippage_mult
        fee = qty * fill_price * (self.config.fee_bps / 10000)
        return fill_price, qty, fee

    def _process_strategy(self, name: str, bar: Bar) -> None:
        """Process a bar for a single strategy."""
        strategy = self._strategies[name]
        perf = self._performances[name]

        # Get current position info
        position = StrategyPosition(
            side=perf.position_side,
            qty=abs(perf.position_qty),
            entry_price=perf.position_entry_price,
        )

        # Get signal
        signal = strategy.on_bar(bar, position=position)
        perf.last_signal = signal

        trade_qty = self.config.trade_size_usdt / bar.close

        # Execute trades based on signal
        if signal == "long" and perf.position_qty <= 0:
            # Close short if exists
            if perf.position_qty < 0:
                fill_price, filled_qty, fee = self._simulate_fill(bar.close, abs(perf.position_qty), "buy")
                pnl = abs(perf.position_qty) * (perf.position_entry_price - fill_price) - fee
                self._record_trade(perf, "short", perf.position_entry_price, fill_price, abs(perf.position_qty), pnl, str(bar.timestamp))

            # Open long
            fill_price, filled_qty, fee = self._simulate_fill(bar.close, trade_qty, "buy")
            perf.position_qty = filled_qty
            perf.position_side = "long"
            perf.position_entry_price = fill_price
            perf.current_equity -= fee

        elif signal == "short" and perf.position_qty >= 0:
            # Close long if exists
            if perf.position_qty > 0:
                fill_price, filled_qty, fee = self._simulate_fill(bar.close, perf.position_qty, "sell")
                pnl = perf.position_qty * (fill_price - perf.position_entry_price) - fee
                self._record_trade(perf, "long", perf.position_entry_price, fill_price, perf.position_qty, pnl, str(bar.timestamp))

            # Open short
            fill_price, filled_qty, fee = self._simulate_fill(bar.close, trade_qty, "sell")
            perf.position_qty = -filled_qty
            perf.position_side = "short"
            perf.position_entry_price = fill_price
            perf.current_equity -= fee

        elif signal == "exit" and perf.position_qty != 0:
            side = "sell" if perf.position_qty > 0 else "buy"
            fill_price, filled_qty, fee = self._simulate_fill(bar.close, abs(perf.position_qty), side)

            if perf.position_qty > 0:
                pnl = perf.position_qty * (fill_price - perf.position_entry_price) - fee
                self._record_trade(perf, "long", perf.position_entry_price, fill_price, perf.position_qty, pnl, str(bar.timestamp))
            else:
                pnl = abs(perf.position_qty) * (perf.position_entry_price - fill_price) - fee
                self._record_trade(perf, "short", perf.position_entry_price, fill_price, abs(perf.position_qty), pnl, str(bar.timestamp))

            perf.position_qty = 0.0
            perf.position_side = "flat"
            perf.position_entry_price = 0.0

        # Update unrealized PnL
        if perf.position_qty > 0:
            perf.unrealized_pnl = perf.position_qty * (bar.close - perf.position_entry_price)
        elif perf.position_qty < 0:
            perf.unrealized_pnl = abs(perf.position_qty) * (perf.position_entry_price - bar.close)
        else:
            perf.unrealized_pnl = 0.0

        # Update equity and drawdown
        total_equity = perf.current_equity + perf.unrealized_pnl
        perf.peak_equity = max(perf.peak_equity, total_equity)
        if perf.peak_equity > 0:
            drawdown = (perf.peak_equity - total_equity) / perf.peak_equity
            perf.max_drawdown = max(perf.max_drawdown, drawdown)

        # Record equity curve (sample every 100 bars to save memory)
        if self._bars_processed % 100 == 0:
            perf.equity_curve.append({
                "timestamp": str(bar.timestamp),
                "equity": total_equity,
                "price": bar.close,
            })

    def _record_trade(
        self,
        perf: StrategyPerformance,
        side: str,
        entry_price: float,
        exit_price: float,
        qty: float,
        pnl: float,
        timestamp: str,
    ) -> None:
        """Record a completed trade."""
        perf.total_trades += 1
        perf.realized_pnl += pnl
        perf.current_equity += pnl
        perf.last_trade_time = timestamp

        if pnl > 0:
            perf.winning_trades += 1
            perf.total_profit += pnl
        else:
            perf.losing_trades += 1
            perf.total_loss += pnl

        perf.trades.append({
            "timestamp": timestamp,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "qty": qty,
            "pnl": pnl,
            "equity_after": perf.current_equity,
        })

    def _get_leaderboard(self) -> list[StrategyPerformance]:
        """Get strategies sorted by return."""
        return sorted(
            self._performances.values(),
            key=lambda p: p.return_pct,
            reverse=True
        )

    def _print_leaderboard(self, top_n: int = 10) -> None:
        """Print strategy leaderboard."""
        leaderboard = self._get_leaderboard()

        logger.info("")
        logger.info("=" * 100)
        logger.info(f"{'STRATEGY LEADERBOARD':^100}")
        logger.info("=" * 100)
        logger.info(f"{'Rank':<5} {'Strategy':<25} {'Return':>10} {'Equity':>12} {'Trades':>8} {'Win%':>8} {'PF':>8} {'MaxDD':>8}")
        logger.info("-" * 100)

        for i, perf in enumerate(leaderboard[:top_n], 1):
            pf_str = f"{perf.profit_factor:.2f}" if perf.profit_factor != float('inf') else "∞"
            logger.info(
                f"{i:<5} {perf.name:<25} {perf.return_pct:>+9.2f}% "
                f"${perf.current_equity:>10,.2f} {perf.total_trades:>8} "
                f"{perf.win_rate:>7.1f}% {pf_str:>8} {perf.max_drawdown*100:>7.2f}%"
            )

        if len(leaderboard) > top_n:
            logger.info(f"... and {len(leaderboard) - top_n} more strategies")

        # Show best and worst
        if leaderboard:
            best = leaderboard[0]
            worst = leaderboard[-1]
            logger.info("-" * 100)
            logger.info(f"BEST:  {best.name} with {best.return_pct:+.2f}% return")
            logger.info(f"WORST: {worst.name} with {worst.return_pct:+.2f}% return")

        logger.info("=" * 100)
        logger.info("")

    def _save_results(self) -> None:
        """Save all results to disk."""
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

        # Save leaderboard
        leaderboard = self._get_leaderboard()
        leaderboard_data = [p.to_dict() for p in leaderboard]

        leaderboard_file = self.config.data_dir / "leaderboard.json"
        leaderboard_file.write_text(json.dumps(leaderboard_data, indent=2, default=str))

        # Save as CSV for easy analysis
        df = pd.DataFrame(leaderboard_data)
        df.to_csv(self.config.data_dir / "leaderboard.csv", index=False)

        # Save detailed results for each strategy
        for name, perf in self._performances.items():
            strategy_dir = self.config.data_dir / "strategies" / name
            strategy_dir.mkdir(parents=True, exist_ok=True)

            # Save trades
            if perf.trades:
                trades_df = pd.DataFrame(perf.trades)
                trades_df.to_csv(strategy_dir / "trades.csv", index=False)

            # Save equity curve
            if perf.equity_curve:
                equity_df = pd.DataFrame(perf.equity_curve)
                equity_df.to_csv(strategy_dir / "equity_curve.csv", index=False)

            # Save summary
            summary = perf.to_dict()
            (strategy_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

        # Save backtest state
        state = {
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "bars_processed": self._bars_processed,
            "symbol": self.config.symbol,
            "timeframe": self.config.timeframe,
            "days": self.config.days,
            "num_strategies": len(self._strategies),
            "best_strategy": leaderboard[0].name if leaderboard else None,
            "best_return": leaderboard[0].return_pct if leaderboard else 0.0,
        }
        (self.config.data_dir / "backtest_state.json").write_text(json.dumps(state, indent=2))

        logger.info(f"Results saved to {self.config.data_dir}")
        logger.info(f"  - leaderboard.csv")
        logger.info(f"  - strategies/*/trades.csv")
        logger.info(f"  - strategies/*/equity_curve.csv")

    def run(self) -> None:
        """Run the backtest."""
        setup_logging(level="INFO", file_level="DEBUG")

        self._started_at = datetime.now(timezone.utc)

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"{'MULTI-STRATEGY HISTORICAL BACKTEST':^80}")
        logger.info("=" * 80)
        logger.info(f"  Symbol:           {self.config.symbol}")
        logger.info(f"  Timeframe:        {self.config.timeframe}")
        logger.info(f"  Period:           {self.config.days} days")
        logger.info(f"  Initial Equity:   ${self.config.initial_equity:,.2f} per strategy")
        logger.info(f"  Trade Size:       ${self.config.trade_size_usdt:,.2f}")
        logger.info(f"  Strategies:       {len(self._strategies)}")
        logger.info(f"  Data Dir:         {self.config.data_dir}")
        logger.info("=" * 80)
        logger.info("")

        # Download/load historical data
        downloader = HistoricalDataDownloader(
            symbol=self.config.symbol,
            timeframe=self.config.timeframe,
            cache_dir=self.config.cache_dir,
        )

        df = downloader.download_and_cache(days=self.config.days)

        if df.empty:
            logger.error("No data available for backtesting!")
            return

        total_bars = len(df)
        logger.info(f"Starting backtest with {total_bars:,} bars...")
        logger.info("")

        start_time = time.monotonic()
        last_progress = 0

        for idx, row in df.iterrows():
            bar = Bar(
                timestamp=row["timestamp"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
            )

            # Process all strategies
            for name in self._strategies:
                self._process_strategy(name, bar)

            self._bars_processed += 1

            # Progress update every 5%
            progress = int(self._bars_processed / total_bars * 100)
            if progress >= last_progress + 5:
                elapsed = time.monotonic() - start_time
                bars_per_sec = self._bars_processed / elapsed if elapsed > 0 else 0
                eta_sec = (total_bars - self._bars_processed) / bars_per_sec if bars_per_sec > 0 else 0

                leaderboard = self._get_leaderboard()
                best = leaderboard[0] if leaderboard else None

                logger.info(
                    f"Progress: {progress:3d}% | "
                    f"{self._bars_processed:,}/{total_bars:,} bars | "
                    f"{bars_per_sec:,.0f} bars/sec | "
                    f"ETA: {eta_sec:.0f}s | "
                    f"Best: {best.name if best else 'N/A'} ({best.return_pct:+.2f}%)" if best else ""
                )
                last_progress = progress

        elapsed = time.monotonic() - start_time

        logger.info("")
        logger.info(f"Backtest completed in {elapsed:.1f} seconds")
        logger.info(f"  Processed {self._bars_processed:,} bars")
        logger.info(f"  Speed: {self._bars_processed / elapsed:,.0f} bars/second")

        # Save results
        self._save_results()

        # Print final results
        self._print_final_report()

    def _print_final_report(self) -> None:
        """Print comprehensive final report."""
        leaderboard = self._get_leaderboard()

        logger.info("")
        logger.info("=" * 100)
        logger.info(f"{'BACKTEST RESULTS - MULTI-STRATEGY COMPARISON':^100}")
        logger.info("=" * 100)
        logger.info(f"  Symbol:           {self.config.symbol}")
        logger.info(f"  Period:           {self.config.days} days")
        logger.info(f"  Bars Processed:   {self._bars_processed:,}")
        logger.info(f"  Strategies:       {len(self._strategies)}")
        logger.info(f"  Data Saved To:    {self.config.data_dir}")
        logger.info("")

        # Full leaderboard
        self._print_leaderboard(top_n=len(leaderboard))

        # Recommendation
        if leaderboard:
            best = leaderboard[0]
            logger.info("")
            logger.info("=" * 100)
            logger.info(f"{'RECOMMENDATION FOR LIVE TRADING':^100}")
            logger.info("=" * 100)
            logger.info(f"  Best Strategy:    {best.name}")
            logger.info(f"  Strategy Type:    {best.strategy_type}")
            logger.info(f"  Parameters:       {json.dumps(best.params)}")
            logger.info(f"  Return:           {best.return_pct:+.2f}%")
            logger.info(f"  Win Rate:         {best.win_rate:.1f}%")
            pf_str = f"{best.profit_factor:.2f}" if best.profit_factor != float('inf') else "∞"
            logger.info(f"  Profit Factor:    {pf_str}")
            logger.info(f"  Max Drawdown:     {best.max_drawdown*100:.2f}%")
            logger.info(f"  Total Trades:     {best.total_trades}")
            logger.info("=" * 100)
            logger.info("")

            # Command to use this strategy
            logger.info("To run this strategy live:")
            params_str = " ".join([f"--{k}={v}" for k, v in best.params.items() if k not in ["stop_loss_pct", "take_profit_pct"]])
            logger.info(f"  python main.py daemon --strategy {best.strategy_type} {params_str}")
            logger.info("")

        # Statistics summary
        profitable = sum(1 for p in leaderboard if p.return_pct > 0)
        losing = sum(1 for p in leaderboard if p.return_pct < 0)
        avg_return = sum(p.return_pct for p in leaderboard) / len(leaderboard) if leaderboard else 0

        logger.info("=" * 100)
        logger.info(f"{'SUMMARY STATISTICS':^100}")
        logger.info("=" * 100)
        logger.info(f"  Profitable Strategies: {profitable}/{len(leaderboard)} ({profitable/len(leaderboard)*100:.1f}%)")
        logger.info(f"  Losing Strategies:     {losing}/{len(leaderboard)} ({losing/len(leaderboard)*100:.1f}%)")
        logger.info(f"  Average Return:        {avg_return:+.2f}%")
        logger.info(f"  Best Return:           {leaderboard[0].return_pct:+.2f}%" if leaderboard else "  Best Return:           N/A")
        logger.info(f"  Worst Return:          {leaderboard[-1].return_pct:+.2f}%" if leaderboard else "  Worst Return:          N/A")
        logger.info("=" * 100)
        logger.info("")


def run_backtest_compare(
    symbol: str = "BTC/USDT",
    timeframe: str = "1m",
    days: int = 365,
    initial_equity: float = 10_000.0,
    data_dir: str = "data/backtest",
) -> None:
    """Run multi-strategy backtest comparison."""
    config = BacktestConfig(
        symbol=symbol,
        timeframe=timeframe,
        days=days,
        initial_equity=initial_equity,
        data_dir=Path(data_dir),
    )

    backtester = MultiStrategyBacktester(config)
    backtester.run()

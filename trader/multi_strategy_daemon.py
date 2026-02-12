"""
Multi-Strategy 24/7 Paper Trading Daemon

Runs multiple trading strategies simultaneously on the same market data,
comparing their performance to identify the best strategy for live trading.
"""

from __future__ import annotations

import ctypes
import json
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import pandas as pd

from trader.logging import get_logger, setup_logging
from trader.strategy.base import Bar, Strategy, StrategyPosition

logger = get_logger(__name__)

# Windows API constants
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def prevent_sleep() -> bool:
    """Prevent Windows from entering sleep mode."""
    if sys.platform != "win32":
        return False
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        logger.info("System sleep prevention enabled")
        return True
    except Exception as exc:
        logger.warning(f"Failed to prevent sleep: {exc}")
        return False


def allow_sleep() -> None:
    """Allow Windows to enter sleep mode again."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    except Exception:
        pass


@dataclass
class StrategyConfig:
    """Configuration for a single strategy."""
    name: str
    strategy_type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyPerformance:
    """Performance tracking for a single strategy."""
    name: str
    strategy_type: str
    params: dict[str, Any]

    # Trading state
    position_qty: float = 0.0
    position_side: Literal["flat", "long", "short"] = "flat"
    position_entry_price: float = 0.0

    # Performance metrics
    initial_equity: float = 10_000.0
    current_equity: float = 10_000.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    peak_equity: float = 10_000.0
    max_drawdown: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0

    # Timestamps
    last_trade_time: str = ""
    last_signal: str = "hold"

    # Trade history
    trades: list[dict[str, Any]] = field(default_factory=list)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)

    @property
    def return_pct(self) -> float:
        """Calculate return percentage."""
        if self.initial_equity <= 0:
            return 0.0
        return ((self.current_equity - self.initial_equity) / self.initial_equity) * 100

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100

    @property
    def profit_factor(self) -> float:
        """Calculate profit factor."""
        if self.total_loss == 0:
            return float('inf') if self.total_profit > 0 else 0.0
        return abs(self.total_profit / self.total_loss)

    @property
    def avg_trade_pnl(self) -> float:
        """Calculate average trade PnL."""
        if self.total_trades == 0:
            return 0.0
        return self.realized_pnl / self.total_trades

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "strategy_type": self.strategy_type,
            "params": self.params,
            "return_pct": self.return_pct,
            "current_equity": self.current_equity,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "max_drawdown": self.max_drawdown,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_trade_pnl": self.avg_trade_pnl,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "position_side": self.position_side,
            "position_qty": self.position_qty,
            "last_signal": self.last_signal,
            "last_trade_time": self.last_trade_time,
        }


@dataclass
class MultiStrategyConfig:
    """Configuration for multi-strategy daemon."""
    symbol: str = "BTC/USDT"
    timeframe: str = "1m"
    initial_equity: float = 10_000.0
    trade_size_usdt: float = 1_000.0
    data_dir: Path = field(default_factory=lambda: Path("data/multi_strategy"))
    testnet: bool = True
    prevent_sleep: bool = True
    save_interval_minutes: int = 5
    leaderboard_interval_minutes: int = 10
    slippage_bps: float = 5.0
    fee_bps: float = 4.0


def generate_strategy_matrix() -> list[StrategyConfig]:
    """Generate a matrix of strategy configurations to test."""
    strategies = []

    # EMA Cross variations
    for fast in [5, 8, 12]:
        for slow in [20, 26, 50]:
            if fast >= slow:
                continue
            strategies.append(StrategyConfig(
                name=f"ema_{fast}_{slow}",
                strategy_type="ema_cross",
                params={"fast_len": fast, "slow_len": slow, "stop_loss_pct": 0.02, "take_profit_pct": 0.04}
            ))

    # RSI variations
    for period in [7, 14, 21]:
        for oversold in [25, 30, 35]:
            overbought = 100 - oversold
            strategies.append(StrategyConfig(
                name=f"rsi_{period}_{oversold}_{overbought}",
                strategy_type="rsi",
                params={"period": period, "oversold": oversold, "overbought": overbought, "stop_loss_pct": 0.02, "take_profit_pct": 0.04}
            ))

    # MACD variations
    for fast in [8, 12]:
        for slow in [21, 26]:
            for sig in [7, 9]:
                if fast >= slow:
                    continue
                strategies.append(StrategyConfig(
                    name=f"macd_{fast}_{slow}_{sig}",
                    strategy_type="macd",
                    params={"fast_period": fast, "slow_period": slow, "signal_period": sig, "stop_loss_pct": 0.02, "take_profit_pct": 0.04}
                ))

    # Bollinger Band variations
    for period in [15, 20, 25]:
        for std in [1.5, 2.0, 2.5]:
            for mode in ["mean_reversion", "breakout"]:
                strategies.append(StrategyConfig(
                    name=f"bb_{period}_{std}_{mode[:4]}",
                    strategy_type="bollinger",
                    params={"period": period, "std_dev": std, "mode": mode, "stop_loss_pct": 0.02, "take_profit_pct": 0.04}
                ))

    return strategies


class MultiStrategyDaemon:
    """
    Runs multiple strategies simultaneously on the same market data.

    Features:
    - Parallel strategy execution on shared price feed
    - Independent performance tracking per strategy
    - Real-time leaderboard ranking
    - Detailed metrics for strategy comparison
    """

    def __init__(
        self,
        config: MultiStrategyConfig,
        strategy_configs: list[StrategyConfig] | None = None,
    ) -> None:
        self.config = config
        self.strategy_configs = strategy_configs or generate_strategy_matrix()

        self._strategies: dict[str, Strategy] = {}
        self._performances: dict[str, StrategyPerformance] = {}
        self._stop_event = threading.Event()
        self._bars_processed = 0
        self._started_at: datetime | None = None
        self._market_data: list[dict[str, Any]] = []

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

        logger.info(f"Initialized {len(self._strategies)} strategies for comparison")

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

    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown."""
        def handler(signum: int, frame: Any) -> None:
            logger.info("\nShutdown signal received...")
            self._stop_event.set()
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

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

        # Record equity curve point
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

            # Save equity curve (always save CSV, optionally parquet)
            if perf.equity_curve:
                equity_df = pd.DataFrame(perf.equity_curve)
                equity_df.to_csv(strategy_dir / "equity_curve.csv", index=False)
                try:
                    equity_df.to_parquet(strategy_dir / "equity_curve.parquet", index=False)
                except (ImportError, Exception):
                    pass  # Parquet is optional

            # Save summary
            summary = perf.to_dict()
            (strategy_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

        # Save market data (always save CSV, optionally parquet)
        if self._market_data:
            market_df = pd.DataFrame(self._market_data)
            market_df.to_csv(self.config.data_dir / "market_data.csv", index=False)
            try:
                market_df.to_parquet(self.config.data_dir / "market_data.parquet", index=False)
            except (ImportError, Exception):
                pass  # Parquet is optional

        # Save daemon state
        state = {
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "bars_processed": self._bars_processed,
            "symbol": self.config.symbol,
            "timeframe": self.config.timeframe,
            "num_strategies": len(self._strategies),
            "best_strategy": leaderboard[0].name if leaderboard else None,
            "best_return": leaderboard[0].return_pct if leaderboard else 0.0,
        }
        (self.config.data_dir / "daemon_state.json").write_text(json.dumps(state, indent=2))

        logger.info(f"Results saved to {self.config.data_dir}")
        logger.info(f"  - leaderboard.csv")
        logger.info(f"  - market_data.csv")
        logger.info(f"  - strategies/*/trades.csv")
        logger.info(f"  - strategies/*/equity_curve.csv")

    def run(self) -> None:
        """Run the multi-strategy daemon."""
        from trader.data.binance_live import BinanceLiveFeed

        setup_logging(level="INFO", file_level="DEBUG")

        self._started_at = datetime.now(timezone.utc)

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"{'MULTI-STRATEGY PAPER TRADING DAEMON':^80}")
        logger.info("=" * 80)
        logger.info(f"  Symbol:           {self.config.symbol}")
        logger.info(f"  Timeframe:        {self.config.timeframe}")
        logger.info(f"  Initial Equity:   ${self.config.initial_equity:,.2f} per strategy")
        logger.info(f"  Trade Size:       ${self.config.trade_size_usdt:,.2f}")
        logger.info(f"  Strategies:       {len(self._strategies)}")
        logger.info(f"  Testnet:          {self.config.testnet}")
        logger.info(f"  Data Dir:         {self.config.data_dir}")
        logger.info("=" * 80)
        logger.info("  Press Ctrl+C to stop and see final results")
        logger.info("=" * 80)
        logger.info("")

        # List all strategies
        logger.info("Strategies being tested:")
        for i, sc in enumerate(self.strategy_configs, 1):
            logger.info(f"  {i:2}. {sc.name} ({sc.strategy_type})")
        logger.info("")

        self._setup_signal_handlers()

        if self.config.prevent_sleep:
            prevent_sleep()

        self.config.data_dir.mkdir(parents=True, exist_ok=True)

        feed = BinanceLiveFeed(
            symbol=self.config.symbol,
            timeframe=self.config.timeframe,
            mode="websocket",
            testnet=self.config.testnet,
            bootstrap_history_bars=50,
        )

        last_leaderboard = time.monotonic()
        last_save = time.monotonic()
        leaderboard_interval = self.config.leaderboard_interval_minutes * 60
        save_interval = self.config.save_interval_minutes * 60

        try:
            logger.info("Connecting to Binance WebSocket feed...")
            logger.info("Waiting for first candle...")
            logger.info("")

            for live_bar in feed.iter_closed_bars(max_bars=None):
                if self._stop_event.is_set():
                    break

                bar = Bar(
                    timestamp=live_bar.timestamp,
                    open=live_bar.open,
                    high=live_bar.high,
                    low=live_bar.low,
                    close=live_bar.close,
                    volume=live_bar.volume,
                )

                # Store market data
                self._market_data.append({
                    "timestamp": str(bar.timestamp),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                })

                # Process all strategies
                for name in self._strategies:
                    self._process_strategy(name, bar)

                self._bars_processed += 1

                # Get current best
                leaderboard = self._get_leaderboard()
                best = leaderboard[0] if leaderboard else None

                # Log status
                active_positions = sum(1 for p in self._performances.values() if p.position_side != "flat")
                logger.info(
                    f"[{bar.timestamp}] ${bar.close:,.2f} | "
                    f"Bar #{self._bars_processed} | "
                    f"Positions: {active_positions}/{len(self._strategies)} | "
                    f"Best: {best.name if best else 'N/A'} ({best.return_pct:+.2f}%)" if best else ""
                )

                now = time.monotonic()

                # Periodic leaderboard
                if now - last_leaderboard >= leaderboard_interval:
                    self._print_leaderboard()
                    last_leaderboard = now

                # Periodic save
                if now - last_save >= save_interval:
                    self._save_results()
                    last_save = now

        except Exception as exc:
            logger.error(f"Daemon error: {exc}")
            import traceback
            traceback.print_exc()
        finally:
            logger.info("")
            logger.info("Shutting down...")

            # Final save
            self._save_results()

            try:
                feed.close()
            except Exception:
                pass

            if self.config.prevent_sleep:
                allow_sleep()

            # Print final results
            self._print_final_report()

    def _print_final_report(self) -> None:
        """Print comprehensive final report."""
        uptime = datetime.now(timezone.utc) - self._started_at if self._started_at else None
        leaderboard = self._get_leaderboard()

        logger.info("")
        logger.info("=" * 100)
        logger.info(f"{'FINAL REPORT - MULTI-STRATEGY COMPARISON':^100}")
        logger.info("=" * 100)

        if uptime:
            hours = uptime.total_seconds() / 3600
            logger.info(f"  Total Runtime:    {hours:.2f} hours")
        logger.info(f"  Bars Processed:   {self._bars_processed}")
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
            logger.info(f"  Profit Factor:    {best.profit_factor:.2f}")
            logger.info(f"  Max Drawdown:     {best.max_drawdown*100:.2f}%")
            logger.info(f"  Total Trades:     {best.total_trades}")
            logger.info("=" * 100)
            logger.info("")

            # Command to use this strategy
            logger.info("To run this strategy live:")
            params_str = " ".join([f"--{k}={v}" for k, v in best.params.items() if k not in ["stop_loss_pct", "take_profit_pct"]])
            logger.info(f"  python main.py run --mode live --strategy {best.strategy_type} {params_str}")
            logger.info("")


def run_multi_strategy(
    symbol: str = "BTC/USDT",
    timeframe: str = "1m",
    initial_equity: float = 10_000.0,
    testnet: bool = True,
    data_dir: str = "data/multi_strategy",
) -> None:
    """Run multi-strategy comparison daemon."""
    config = MultiStrategyConfig(
        symbol=symbol,
        timeframe=timeframe,
        initial_equity=initial_equity,
        testnet=testnet,
        data_dir=Path(data_dir),
    )

    daemon = MultiStrategyDaemon(config)
    daemon.run()

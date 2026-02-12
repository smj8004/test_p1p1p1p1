"""
24/7 Background Trading Daemon

Runs paper trading continuously, preventing system sleep and
accumulating real-time market data until manually stopped.
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
from typing import Any
from uuid import uuid4

import pandas as pd

from trader.logging import get_logger, setup_logging

logger = get_logger(__name__)

# Windows API constants for preventing sleep
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002


def prevent_sleep() -> bool:
    """Prevent Windows from entering sleep mode."""
    if sys.platform != "win32":
        logger.warning("Sleep prevention only supported on Windows")
        return False
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )
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
        logger.info("System sleep prevention disabled")
    except Exception:
        pass


@dataclass
class DaemonConfig:
    """Configuration for the trading daemon."""
    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT"])
    timeframe: str = "1m"
    strategy: str = "ema_cross"
    strategy_params: dict[str, Any] = field(default_factory=dict)
    initial_equity: float = 10_000.0
    fixed_notional_usdt: float = 1_000.0
    data_dir: Path = field(default_factory=lambda: Path("data"))
    save_interval_minutes: int = 5
    heartbeat_interval_minutes: int = 30
    testnet: bool = True
    prevent_sleep: bool = True


@dataclass
class DaemonState:
    """Runtime state of the daemon."""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    bars_processed: int = 0
    trades_executed: int = 0
    current_equity: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    position_qty: float = 0.0
    position_side: str = "flat"
    last_bar_time: str = ""
    last_price: float = 0.0
    last_save_time: datetime | None = None
    is_running: bool = False
    stop_requested: bool = False


class MarketDataAccumulator:
    """Accumulates and persists real-time market data."""

    def __init__(self, data_dir: Path, symbol: str, timeframe: str) -> None:
        self.data_dir = data_dir
        self.symbol = symbol
        self.timeframe = timeframe
        self._bars: list[dict[str, Any]] = []
        self._lock = threading.Lock()

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._load_existing_data()

    def _file_path(self) -> Path:
        safe_symbol = self.symbol.replace("/", "_")
        return self.data_dir / f"{safe_symbol}_{self.timeframe}_data.parquet"

    def _load_existing_data(self) -> None:
        """Load existing data from disk if available."""
        path = self._file_path()
        if path.exists():
            try:
                df = pd.read_parquet(path)
                self._bars = df.to_dict("records")
                logger.info(f"Loaded {len(self._bars)} existing bars for {self.symbol}")
            except Exception as exc:
                logger.warning(f"Failed to load existing data: {exc}")

    def add_bar(self, bar: dict[str, Any]) -> None:
        """Add a new bar to the accumulator."""
        with self._lock:
            bar_ts = str(bar.get("timestamp", ""))
            if self._bars and str(self._bars[-1].get("timestamp", "")) == bar_ts:
                return
            self._bars.append(bar)

    def save(self) -> None:
        """Save accumulated data to disk."""
        with self._lock:
            if not self._bars:
                return
            try:
                df = pd.DataFrame(self._bars)
                df.to_parquet(self._file_path(), index=False)
                logger.debug(f"Saved {len(self._bars)} bars to {self._file_path()}")
            except Exception as exc:
                logger.error(f"Failed to save data: {exc}")

    def get_dataframe(self) -> pd.DataFrame:
        """Get accumulated data as DataFrame."""
        with self._lock:
            return pd.DataFrame(self._bars) if self._bars else pd.DataFrame()

    @property
    def bar_count(self) -> int:
        with self._lock:
            return len(self._bars)


class TradingDaemon:
    """
    24/7 Trading Daemon that runs paper trading continuously.

    Features:
    - Prevents system sleep on Windows
    - Accumulates real-time market data
    - Runs until manually stopped (Ctrl+C)
    - Periodic state saving
    - Heartbeat logging
    """

    def __init__(self, config: DaemonConfig) -> None:
        self.config = config
        self.state = DaemonState(current_equity=config.initial_equity)
        self._accumulators: dict[str, MarketDataAccumulator] = {}
        self._stop_event = threading.Event()
        self._broker: Any = None
        self._storage: Any = None

        for symbol in config.symbols:
            self._accumulators[symbol] = MarketDataAccumulator(
                config.data_dir, symbol, config.timeframe
            )

    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown on Ctrl+C."""
        def handler(signum: int, frame: Any) -> None:
            logger.info("\nShutdown signal received, stopping daemon...")
            self.state.stop_requested = True
            self._stop_event.set()

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _build_strategy(self, strategy_name: str, params: dict[str, Any]) -> Any:
        """Build strategy instance."""
        from trader.strategy.bollinger import BollingerBandStrategy
        from trader.strategy.ema_cross import EMACrossStrategy
        from trader.strategy.macd import MACDStrategy
        from trader.strategy.rsi import RSIStrategy

        stop_loss = float(params.get("stop_loss_pct", 0.02))
        take_profit = float(params.get("take_profit_pct", 0.04))
        allow_short = bool(params.get("allow_short", True))

        if strategy_name == "ema_cross":
            return EMACrossStrategy(
                short_window=int(params.get("fast_len", 12)),
                long_window=int(params.get("slow_len", 26)),
                allow_short=allow_short,
                stop_loss_pct=stop_loss,
                take_profit_pct=take_profit,
            )
        if strategy_name == "rsi":
            return RSIStrategy(
                period=int(params.get("period", 14)),
                overbought=float(params.get("overbought", 70)),
                oversold=float(params.get("oversold", 30)),
                allow_short=allow_short,
                stop_loss_pct=stop_loss,
                take_profit_pct=take_profit,
            )
        if strategy_name == "macd":
            return MACDStrategy(
                fast_period=int(params.get("fast_period", 12)),
                slow_period=int(params.get("slow_period", 26)),
                signal_period=int(params.get("signal_period", 9)),
                allow_short=allow_short,
                stop_loss_pct=stop_loss,
                take_profit_pct=take_profit,
            )
        if strategy_name == "bollinger":
            return BollingerBandStrategy(
                period=int(params.get("period", 20)),
                std_dev=float(params.get("std_dev", 2.0)),
                mode=params.get("mode", "mean_reversion"),
                allow_short=allow_short,
                stop_loss_pct=stop_loss,
                take_profit_pct=take_profit,
            )
        raise ValueError(f"Unknown strategy: {strategy_name}")

    def _format_uptime(self, uptime: float) -> str:
        """Format uptime in seconds to human readable string."""
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        seconds = int(uptime % 60)
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"

    def _heartbeat(self) -> None:
        """Log periodic heartbeat with status."""
        uptime_sec = (datetime.now(timezone.utc) - self.state.started_at).total_seconds()
        uptime_str = self._format_uptime(uptime_sec)

        total_bars = sum(acc.bar_count for acc in self._accumulators.values())

        pnl_color = "+" if self.state.realized_pnl >= 0 else ""

        logger.info("=" * 70)
        logger.info(f"[HEARTBEAT] Uptime: {uptime_str}")
        logger.info(f"  Bars processed: {self.state.bars_processed} | Data accumulated: {total_bars}")
        logger.info(f"  Trades: {self.state.trades_executed} | Position: {self.state.position_side} ({self.state.position_qty:.6f})")
        logger.info(f"  Equity: ${self.state.current_equity:,.2f} | PnL: {pnl_color}${self.state.realized_pnl:,.2f}")
        logger.info(f"  Last price: ${self.state.last_price:,.2f} | Last bar: {self.state.last_bar_time}")
        logger.info("=" * 70)

    def _save_state(self) -> None:
        """Save daemon state and accumulated data."""
        for acc in self._accumulators.values():
            acc.save()

        state_file = self.config.data_dir / "daemon_state.json"
        state_data = {
            "started_at": self.state.started_at.isoformat(),
            "bars_processed": self.state.bars_processed,
            "trades_executed": self.state.trades_executed,
            "current_equity": self.state.current_equity,
            "realized_pnl": self.state.realized_pnl,
            "position_qty": self.state.position_qty,
            "position_side": self.state.position_side,
            "last_bar_time": self.state.last_bar_time,
            "last_price": self.state.last_price,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "symbols": self.config.symbols,
            "strategy": self.config.strategy,
            "timeframe": self.config.timeframe,
        }
        try:
            state_file.write_text(json.dumps(state_data, indent=2))
            logger.debug(f"State saved to {state_file}")
        except Exception as exc:
            logger.error(f"Failed to save state: {exc}")

        self.state.last_save_time = datetime.now(timezone.utc)

    def run(self) -> None:
        """Run the trading daemon."""
        from trader.broker.base import OrderRequest
        from trader.broker.paper import PaperBroker
        from trader.data.binance_live import BinanceLiveFeed
        from trader.storage import SQLiteStorage
        from trader.strategy.base import Bar, StrategyPosition

        setup_logging(level="INFO", file_level="DEBUG")

        logger.info("")
        logger.info("=" * 70)
        logger.info("        24/7 PAPER TRADING DAEMON")
        logger.info("=" * 70)
        logger.info(f"  Symbols:        {', '.join(self.config.symbols)}")
        logger.info(f"  Strategy:       {self.config.strategy}")
        logger.info(f"  Timeframe:      {self.config.timeframe}")
        logger.info(f"  Initial Equity: ${self.config.initial_equity:,.2f}")
        logger.info(f"  Trade Size:     ${self.config.fixed_notional_usdt:,.2f}")
        logger.info(f"  Testnet:        {self.config.testnet}")
        logger.info(f"  Data Dir:       {self.config.data_dir}")
        logger.info("=" * 70)
        logger.info("  Press Ctrl+C to stop gracefully")
        logger.info("=" * 70)
        logger.info("")

        self._setup_signal_handlers()

        if self.config.prevent_sleep:
            prevent_sleep()

        self.state.is_running = True
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

        self._broker = PaperBroker(
            starting_cash=self.config.initial_equity,
            slippage_bps=5,
            taker_fee_bps=4,
            maker_fee_bps=2,
        )

        self._storage = SQLiteStorage(str(self.config.data_dir / "daemon_trades.db"))
        run_id = uuid4().hex

        symbol = self.config.symbols[0]  # Primary symbol for now
        strategy = self._build_strategy(self.config.strategy, self.config.strategy_params)

        feed = BinanceLiveFeed(
            symbol=symbol,
            timeframe=self.config.timeframe,
            mode="websocket",
            testnet=self.config.testnet,
            bootstrap_history_bars=50,
        )

        last_heartbeat = time.monotonic()
        last_save = time.monotonic()
        heartbeat_interval = self.config.heartbeat_interval_minutes * 60
        save_interval = self.config.save_interval_minutes * 60

        position_qty = 0.0
        position_entry_price = 0.0
        trade_count = 0

        try:
            logger.info("Connecting to Binance WebSocket feed...")
            logger.info("Waiting for first closed candle (this may take up to 1 minute)...")
            logger.info("")

            for bar in feed.iter_closed_bars(max_bars=None):
                if self._stop_event.is_set():
                    break

                # Accumulate data
                bar_dict = {
                    "timestamp": str(bar.timestamp),
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                }
                self._accumulators[symbol].add_bar(bar_dict)

                # Update broker price
                self._broker.update_market_price(symbol, bar.close)

                # Get strategy signal
                position_side = "flat"
                if position_qty > 0:
                    position_side = "long"
                elif position_qty < 0:
                    position_side = "short"

                strategy_pos = StrategyPosition(
                    side=position_side,
                    qty=abs(position_qty),
                    entry_price=position_entry_price,
                )

                signal = strategy.on_bar(
                    Bar(
                        timestamp=bar.timestamp,
                        open=bar.open,
                        high=bar.high,
                        low=bar.low,
                        close=bar.close,
                        volume=bar.volume,
                    ),
                    position=strategy_pos,
                )

                # Execute trades based on signal
                trade_qty = self.config.fixed_notional_usdt / bar.close

                if signal == "long" and position_qty <= 0:
                    if position_qty < 0:  # Close short first
                        result = self._broker.place_order(OrderRequest(
                            symbol=symbol, side="buy", amount=abs(position_qty)
                        ))
                        if result.status == "FILLED":
                            pnl = abs(position_qty) * (position_entry_price - result.avg_price)
                            self.state.realized_pnl += pnl - result.fee
                            trade_count += 1
                            logger.info(f"[TRADE] Closed SHORT @ ${result.avg_price:.2f} | PnL: ${pnl - result.fee:.2f}")
                    # Open long
                    result = self._broker.place_order(OrderRequest(
                        symbol=symbol, side="buy", amount=trade_qty
                    ))
                    if result.status == "FILLED":
                        position_qty = result.filled_qty
                        position_entry_price = result.avg_price
                        trade_count += 1
                        logger.info(f"[TRADE] Opened LONG @ ${result.avg_price:.2f} | Qty: {result.filled_qty:.6f}")

                elif signal == "short" and position_qty >= 0:
                    if position_qty > 0:  # Close long first
                        result = self._broker.place_order(OrderRequest(
                            symbol=symbol, side="sell", amount=position_qty
                        ))
                        if result.status == "FILLED":
                            pnl = position_qty * (result.avg_price - position_entry_price)
                            self.state.realized_pnl += pnl - result.fee
                            trade_count += 1
                            logger.info(f"[TRADE] Closed LONG @ ${result.avg_price:.2f} | PnL: ${pnl - result.fee:.2f}")
                    # Open short
                    result = self._broker.place_order(OrderRequest(
                        symbol=symbol, side="sell", amount=trade_qty
                    ))
                    if result.status == "FILLED":
                        position_qty = -result.filled_qty
                        position_entry_price = result.avg_price
                        trade_count += 1
                        logger.info(f"[TRADE] Opened SHORT @ ${result.avg_price:.2f} | Qty: {result.filled_qty:.6f}")

                elif signal == "exit" and position_qty != 0:
                    side = "sell" if position_qty > 0 else "buy"
                    result = self._broker.place_order(OrderRequest(
                        symbol=symbol, side=side, amount=abs(position_qty)
                    ))
                    if result.status == "FILLED":
                        if position_qty > 0:
                            pnl = position_qty * (result.avg_price - position_entry_price)
                        else:
                            pnl = abs(position_qty) * (position_entry_price - result.avg_price)
                        self.state.realized_pnl += pnl - result.fee
                        trade_count += 1
                        logger.info(f"[TRADE] EXIT @ ${result.avg_price:.2f} | PnL: ${pnl - result.fee:.2f}")
                        position_qty = 0.0
                        position_entry_price = 0.0

                # Update state
                self.state.bars_processed += 1
                self.state.trades_executed = trade_count
                self.state.last_bar_time = str(bar.timestamp)
                self.state.last_price = bar.close
                self.state.position_qty = position_qty
                self.state.position_side = "long" if position_qty > 0 else ("short" if position_qty < 0 else "flat")

                balance = self._broker.get_balance()
                self.state.current_equity = balance.get("total", self.config.initial_equity)

                # Calculate unrealized PnL
                if position_qty != 0:
                    if position_qty > 0:
                        self.state.unrealized_pnl = position_qty * (bar.close - position_entry_price)
                    else:
                        self.state.unrealized_pnl = abs(position_qty) * (position_entry_price - bar.close)
                else:
                    self.state.unrealized_pnl = 0.0

                # Status line
                pos_str = f"{self.state.position_side.upper()}({abs(position_qty):.4f})" if position_qty != 0 else "FLAT"
                logger.info(
                    f"[{bar.timestamp}] {symbol} ${bar.close:,.2f} | "
                    f"Signal: {signal.upper():5} | Pos: {pos_str:15} | "
                    f"Equity: ${self.state.current_equity:,.2f} | PnL: ${self.state.realized_pnl:+,.2f}"
                )

                # Heartbeat
                now = time.monotonic()
                if now - last_heartbeat >= heartbeat_interval:
                    self._heartbeat()
                    last_heartbeat = now

                # Periodic save
                if now - last_save >= save_interval:
                    self._save_state()
                    last_save = now

        except Exception as exc:
            logger.error(f"Daemon error: {exc}")
            import traceback
            traceback.print_exc()
        finally:
            logger.info("")
            logger.info("Shutting down daemon...")

            # Final save
            self._save_state()

            # Close feed
            try:
                feed.close()
            except Exception:
                pass

            # Close storage
            if self._storage:
                self._storage.close()

            if self.config.prevent_sleep:
                allow_sleep()

            self.state.is_running = False

            # Print final summary
            uptime = datetime.now(timezone.utc) - self.state.started_at
            uptime_str = self._format_uptime(uptime.total_seconds())
            total_bars = sum(acc.bar_count for acc in self._accumulators.values())

            return_pct = ((self.state.current_equity - self.config.initial_equity) / self.config.initial_equity) * 100

            logger.info("")
            logger.info("=" * 70)
            logger.info("        DAEMON STOPPED - FINAL SUMMARY")
            logger.info("=" * 70)
            logger.info(f"  Total Uptime:     {uptime_str}")
            logger.info(f"  Bars Processed:   {self.state.bars_processed}")
            logger.info(f"  Data Accumulated: {total_bars} bars")
            logger.info(f"  Trades Executed:  {self.state.trades_executed}")
            logger.info(f"  Final Equity:     ${self.state.current_equity:,.2f}")
            logger.info(f"  Total Return:     {return_pct:+.2f}%")
            logger.info(f"  Realized PnL:     ${self.state.realized_pnl:+,.2f}")
            logger.info(f"  Data saved to:    {self.config.data_dir}")
            logger.info("=" * 70)
            logger.info("")


def run_daemon(
    symbols: list[str] | None = None,
    strategy: str = "ema_cross",
    timeframe: str = "1m",
    initial_equity: float = 10_000.0,
    testnet: bool = True,
    data_dir: str = "data",
) -> None:
    """
    Run the 24/7 trading daemon.

    Args:
        symbols: List of trading symbols (default: ["BTC/USDT"])
        strategy: Strategy name (ema_cross, rsi, macd, bollinger)
        timeframe: Candle timeframe (default: 1m)
        initial_equity: Starting paper trading equity
        testnet: Use Binance testnet (default: True)
        data_dir: Directory for data storage
    """
    config = DaemonConfig(
        symbols=symbols or ["BTC/USDT"],
        strategy=strategy,
        timeframe=timeframe,
        initial_equity=initial_equity,
        testnet=testnet,
        data_dir=Path(data_dir),
    )

    daemon = TradingDaemon(config)
    daemon.run()

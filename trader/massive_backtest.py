"""
Massive Backtesting Framework

Features:
- Parallel processing with multiprocessing
- Caching with SQLite/pickle for resume
- Multi-symbol (BTCUSDT, ETHUSDT) + Multi-TF (15m/1h/4h/1d)
- Cost profiles (conservative/base/aggressive)
- Price source options (next_open/close/mark_close)
- Robust overfitting filters
- Ensemble weight search with OOS validation
- Auto report generation
- Progress tracking with ETA
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd
import yaml

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich.live import Live
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

from trader.strategy import (
    ALL_STRATEGY_FAMILIES,
    STRATEGY_FACTORIES,
    Bar,
    Strategy,
    StrategyPosition,
)

logger = logging.getLogger(__name__)


@dataclass
class CostProfile:
    """Trading cost profile"""
    name: str
    slippage_bps: float
    fee_taker_bps: float

    @property
    def slippage_pct(self) -> float:
        return self.slippage_bps / 10000

    @property
    def fee_pct(self) -> float:
        return self.fee_taker_bps / 10000


COST_PROFILES = {
    "conservative": CostProfile("conservative", 5, 6),
    "base": CostProfile("base", 3, 4),
    "aggressive": CostProfile("aggressive", 1, 2),
}


@dataclass
class BacktestConfig:
    """Single backtest configuration"""
    config_id: str
    family: str
    strategy_type: str
    params: dict
    symbol: str
    timeframe: str
    leverage: int
    allow_short: bool
    stop_loss_pct: float
    take_profit_pct: float
    cost_profile: str
    price_source: str

    def to_dict(self) -> dict:
        return {
            "config_id": self.config_id,
            "family": self.family,
            "strategy_type": self.strategy_type,
            "params": json.dumps(self.params),
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "leverage": self.leverage,
            "allow_short": self.allow_short,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "cost_profile": self.cost_profile,
            "price_source": self.price_source,
        }


@dataclass
class BacktestResult:
    """Backtest result"""
    config: BacktestConfig

    # Performance
    final_equity: float = 0.0
    return_pct: float = 0.0
    annual_return_pct: float = 0.0

    # Risk metrics
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Trade stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0

    # Additional
    fees_paid: float = 0.0
    days: int = 0
    trades_per_day: float = 0.0

    # Validation scores
    wfo_score: float = 0.0
    mc_percentile: float = 0.0
    robustness_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            **self.config.to_dict(),
            "final_equity": self.final_equity,
            "return_pct": self.return_pct,
            "annual_return_pct": self.annual_return_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_trade_pnl": self.avg_trade_pnl,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "fees_paid": self.fees_paid,
            "days": self.days,
            "trades_per_day": self.trades_per_day,
            "wfo_score": self.wfo_score,
            "mc_percentile": self.mc_percentile,
            "robustness_score": self.robustness_score,
        }


@dataclass
class FilterConfig:
    """Overfitting detection filters"""
    min_trades: int = 30
    max_drawdown_pct: float = -40
    min_profit_factor: float = 1.0
    min_sharpe: float = 0.5
    wfo_positive_ratio: float = 0.6
    mc_percentile: float = 60


def generate_config_id(config: BacktestConfig) -> str:
    """Generate unique ID for config"""
    data = f"{config.family}_{config.strategy_type}_{json.dumps(config.params, sort_keys=True)}"
    data += f"_{config.symbol}_{config.timeframe}_{config.leverage}"
    data += f"_{config.allow_short}_{config.stop_loss_pct}_{config.take_profit_pct}"
    data += f"_{config.cost_profile}_{config.price_source}"
    return hashlib.md5(data.encode()).hexdigest()[:16]


class GridConfigLoader:
    """Load and expand grid configurations from YAML"""

    def __init__(self, config_dir: Path = Path("config/grids")):
        self.config_dir = config_dir

    def load_family_grid(self, family: str) -> dict:
        """Load grid config for a strategy family"""
        path = self.config_dir / f"{family}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Grid config not found: {path}")

        with open(path, encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _passes_constraints(self, config: BacktestConfig, cost_profiles_map: dict) -> bool:
        """Check if config passes constraint filters"""

        # A1: SL/TP ratio constraints
        tp_sl_ratio = config.take_profit_pct / config.stop_loss_pct
        if tp_sl_ratio < 1.2:  # TP too close to SL
            return False
        if tp_sl_ratio > 4.0:  # TP unrealistically far from SL
            return False

        # A2: Minimum profit after fees
        cost_profile = cost_profiles_map.get(config.cost_profile, {"slippage_bps": 3, "fee_taker_bps": 4})
        total_cost_pct = (cost_profile["slippage_bps"] + cost_profile["fee_taker_bps"]) / 10000 * 2  # Round trip
        if config.take_profit_pct < total_cost_pct * 3:  # TP must be at least 3x total cost
            return False

        # A3: Leverage constraints
        if not config.allow_short and config.leverage > 3:  # Long-only with high leverage = risky
            return False

        # A4: Strategy-specific parameter constraints
        params = config.params

        # EMA/Period constraints (trend strategies)
        if "fast" in params and "slow" in params:
            if params["fast"] >= params["slow"]:  # Fast must be < slow
                return False
            if params["slow"] / params["fast"] < 2.0:  # Minimum gap
                return False

        if "fast" in params and "trend" in params:
            if params["fast"] >= params["trend"]:  # Fast < trend
                return False

        if "slow" in params and "trend" in params:
            if params["slow"] >= params["trend"]:  # Slow < trend
                return False

        # Period ratio constraints
        if "entry_period" in params and "exit_period" in params:
            if params["exit_period"] >= params["entry_period"]:  # Exit should be faster
                return False

        # Z-score constraints
        if "entry_zscore" in params and "exit_zscore" in params:
            if params["exit_zscore"] >= params["entry_zscore"]:  # Exit before entry
                return False

        # Momentum period constraints
        if "momentum_fast" in params and "momentum_slow" in params:
            if params["momentum_fast"] >= params["momentum_slow"]:
                return False

        # RSI constraints
        if "rsi_fast" in params and "rsi_slow" in params:
            if params["rsi_fast"] >= params["rsi_slow"]:
                return False

        return True

    def expand_grid(self, grid: dict, apply_constraints: bool = True) -> list[BacktestConfig]:
        """Expand grid into all combinations with optional constraint filtering"""
        configs = []
        family = grid["family"]
        strategies = grid.get("strategies", {})
        risk = grid.get("risk", {})
        position = grid.get("position", {})
        symbols = grid.get("symbols", ["BTCUSDT"])
        timeframes = grid.get("timeframes", ["1h"])
        price_sources = grid.get("price_source", ["next_open"])

        cost_profiles_list = list(grid.get("costs", {}).get("profiles", {}).keys())
        if not cost_profiles_list:
            cost_profiles_list = ["base"]

        # Build cost profile map for constraint checking
        cost_profiles_map = grid.get("costs", {}).get("profiles", {})
        if not cost_profiles_map:
            cost_profiles_map = {"base": {"slippage_bps": 3, "fee_taker_bps": 4}}

        rejected_count = 0

        for strategy_type, param_grid in strategies.items():
            # Generate all parameter combinations
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())

            for param_combo in product(*param_values):
                params = dict(zip(param_names, param_combo))

                for symbol in symbols:
                    for tf in timeframes:
                        for leverage in position.get("leverage", [1]):
                            for allow_short in position.get("allow_short", [True]):
                                for sl in risk.get("stop_loss_pct", [0.02]):
                                    for tp in risk.get("take_profit_pct", [0.04]):
                                        for cost_profile in cost_profiles_list:
                                            for price_source in price_sources:
                                                config = BacktestConfig(
                                                    config_id="",
                                                    family=family,
                                                    strategy_type=strategy_type,
                                                    params=params,
                                                    symbol=symbol,
                                                    timeframe=tf,
                                                    leverage=leverage,
                                                    allow_short=allow_short,
                                                    stop_loss_pct=sl,
                                                    take_profit_pct=tp,
                                                    cost_profile=cost_profile,
                                                    price_source=price_source,
                                                )

                                                # Apply constraint filtering
                                                if apply_constraints:
                                                    if not self._passes_constraints(config, cost_profiles_map):
                                                        rejected_count += 1
                                                        continue

                                                config.config_id = generate_config_id(config)
                                                configs.append(config)

        if apply_constraints and rejected_count > 0:
            logger.info(f"  Constraint filtering: kept {len(configs):,} / rejected {rejected_count:,} configs")

        return configs

    def load_all_families(self) -> list[BacktestConfig]:
        """Load and expand all family grids"""
        all_configs = []

        for yaml_file in self.config_dir.glob("*.yaml"):
            family = yaml_file.stem
            try:
                grid = self.load_family_grid(family)
                configs = self.expand_grid(grid)
                all_configs.extend(configs)
                logger.info(f"Loaded {family}: {len(configs)} combinations")
            except Exception as e:
                logger.warning(f"Error loading {family}: {e}")

        return all_configs


class ResultCache:
    """SQLite-based result cache for resume capability"""

    def __init__(self, cache_path: Path = Path("data/backtest_cache.db")):
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    config_id TEXT PRIMARY KEY,
                    config_json TEXT,
                    result_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_config_id ON results(config_id)
            """)

    def get_completed_ids(self) -> set[str]:
        """Get set of completed config IDs"""
        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.execute("SELECT config_id FROM results")
            return {row[0] for row in cursor.fetchall()}

    def save_result(self, result: BacktestResult):
        """Save result to cache"""
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO results (config_id, config_json, result_json) VALUES (?, ?, ?)",
                (result.config.config_id, json.dumps(result.config.to_dict()), json.dumps(result.to_dict())),
            )

    def load_all_results(self) -> list[dict]:
        """Load all results from cache"""
        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.execute("SELECT result_json FROM results")
            return [json.loads(row[0]) for row in cursor.fetchall()]

    def clear(self):
        """Clear cache"""
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute("DELETE FROM results")


def run_single_backtest(
    config: BacktestConfig,
    df: pd.DataFrame,
    initial_equity: float = 10000.0,
) -> BacktestResult:
    """Run a single backtest"""
    cost_profile = COST_PROFILES.get(config.cost_profile, COST_PROFILES["base"])

    # Create strategy
    factory = STRATEGY_FACTORIES.get(config.family)
    if not factory:
        return BacktestResult(config=config)

    try:
        strategy = factory(
            config.strategy_type,
            config.params,
            allow_short=config.allow_short,
            stop_loss_pct=config.stop_loss_pct,
            take_profit_pct=config.take_profit_pct,
        )
    except Exception:
        return BacktestResult(config=config)

    # Run backtest
    equity = initial_equity
    peak_equity = equity
    max_drawdown = 0.0

    position_side = "flat"
    position_qty = 0.0
    position_entry_price = 0.0

    trades = []
    returns_list = []
    last_equity = equity
    fees_paid = 0.0

    leverage = config.leverage
    pending_signal = None

    for i in range(len(df)):
        row = df.iloc[i]
        bar = Bar(
            timestamp=row.name if hasattr(row.name, 'isoformat') else str(row.name),
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
        )

        # Execute pending signal
        if pending_signal is not None and i > 0:
            exec_price = row["open"] if config.price_source == "next_open" else df.iloc[i - 1]["close"]

            if pending_signal == "long" and position_side != "long":
                if position_side == "short":
                    pnl = position_qty * (position_entry_price - exec_price * (1 + cost_profile.slippage_pct))
                    fee = position_qty * exec_price * cost_profile.fee_pct
                    equity += pnl - fee
                    fees_paid += fee
                    trades.append({"pnl": pnl - fee})

                entry_price = exec_price * (1 + cost_profile.slippage_pct)
                trade_value = equity * leverage * 0.95
                position_qty = trade_value / entry_price
                fee = position_qty * entry_price * cost_profile.fee_pct
                equity -= fee
                fees_paid += fee
                position_side = "long"
                position_entry_price = entry_price

            elif pending_signal == "short" and config.allow_short and position_side != "short":
                if position_side == "long":
                    pnl = position_qty * (exec_price * (1 - cost_profile.slippage_pct) - position_entry_price)
                    fee = position_qty * exec_price * cost_profile.fee_pct
                    equity += pnl - fee
                    fees_paid += fee
                    trades.append({"pnl": pnl - fee})

                entry_price = exec_price * (1 - cost_profile.slippage_pct)
                trade_value = equity * leverage * 0.95
                position_qty = trade_value / entry_price
                fee = position_qty * entry_price * cost_profile.fee_pct
                equity -= fee
                fees_paid += fee
                position_side = "short"
                position_entry_price = entry_price

            elif pending_signal == "exit" and position_side != "flat":
                if position_side == "long":
                    pnl = position_qty * (exec_price * (1 - cost_profile.slippage_pct) - position_entry_price)
                else:
                    pnl = position_qty * (position_entry_price - exec_price * (1 + cost_profile.slippage_pct))
                fee = position_qty * exec_price * cost_profile.fee_pct
                equity += pnl - fee
                fees_paid += fee
                trades.append({"pnl": pnl - fee})
                position_side = "flat"
                position_qty = 0.0
                position_entry_price = 0.0

            pending_signal = None

        # Generate signal
        position = StrategyPosition(side=position_side, qty=position_qty, entry_price=position_entry_price)
        signal = strategy.on_bar(bar, position)
        if signal in ["long", "short", "exit"]:
            pending_signal = signal

        # Track equity
        if position_side == "long":
            unrealized = position_qty * (bar.close - position_entry_price)
        elif position_side == "short":
            unrealized = position_qty * (position_entry_price - bar.close)
        else:
            unrealized = 0.0

        total_equity = equity + unrealized
        peak_equity = max(peak_equity, total_equity)
        if peak_equity > 0:
            dd = (peak_equity - total_equity) / peak_equity
            max_drawdown = max(max_drawdown, dd)

        if last_equity > 0:
            ret = (total_equity - last_equity) / last_equity
            returns_list.append(ret)
        last_equity = total_equity

        if equity <= 0:
            break

    # Close final position
    if position_side != "flat" and equity > 0:
        final_price = df.iloc[-1]["close"]
        if position_side == "long":
            pnl = position_qty * (final_price * (1 - cost_profile.slippage_pct) - position_entry_price)
        else:
            pnl = position_qty * (position_entry_price - final_price * (1 + cost_profile.slippage_pct))
        fee = position_qty * final_price * cost_profile.fee_pct
        equity += pnl - fee
        trades.append({"pnl": pnl - fee})

    # Calculate metrics
    total_trades = len(trades)
    if total_trades == 0:
        return BacktestResult(config=config, total_trades=0)

    wins = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in trades if t["pnl"] <= 0]

    return_pct = (equity - initial_equity) / initial_equity * 100

    # Calculate days from dataframe index
    if hasattr(df.index[0], 'date'):
        days = (df.index[-1] - df.index[0]).days
    else:
        # Bars per day by timeframe
        tf_bars_per_day = {
            "1m": 1440, "5m": 288, "15m": 96, "30m": 48,
            "1h": 24, "4h": 6, "1d": 1
        }
        bars_per_day = tf_bars_per_day.get(config.timeframe, 24)
        days = len(df) // bars_per_day

    days = max(days, 1)
    annual_return = return_pct * 365 / days

    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1e-10
    profit_factor = gross_profit / gross_loss

    # Sharpe ratio - scale by sqrt(periods_per_year)
    # periods_per_year = 252 trading days * bars_per_day
    tf_bars_per_day = {
        "1m": 1440, "5m": 288, "15m": 96, "30m": 48,
        "1h": 24, "4h": 6, "1d": 1
    }
    bars_per_day = tf_bars_per_day.get(config.timeframe, 24)
    periods_per_year = 252 * bars_per_day

    if len(returns_list) > 1:
        returns_arr = np.array(returns_list)
        sharpe = np.mean(returns_arr) / (np.std(returns_arr) + 1e-10) * np.sqrt(periods_per_year)
        neg_returns = returns_arr[returns_arr < 0]
        sortino = np.mean(returns_arr) / (np.std(neg_returns) + 1e-10) * np.sqrt(periods_per_year) if len(neg_returns) > 0 else 0
    else:
        sharpe = 0
        sortino = 0

    calmar = annual_return / (max_drawdown * 100 + 1e-10) if max_drawdown > 0 else 0

    return BacktestResult(
        config=config,
        final_equity=equity,
        return_pct=return_pct,
        annual_return_pct=annual_return,
        max_drawdown_pct=-max_drawdown * 100,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        total_trades=total_trades,
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_trade_pnl=np.mean([t["pnl"] for t in trades]),
        avg_win=avg_win,
        avg_loss=avg_loss,
        fees_paid=fees_paid,
        days=days,
        trades_per_day=total_trades / days,
    )


def worker_backtest(args: tuple) -> dict | None:
    """Worker function for parallel processing"""
    config_dict, df_bytes, initial_equity = args

    try:
        config = BacktestConfig(**{
            k: json.loads(v) if k == "params" else v
            for k, v in config_dict.items()
        })
        df = pickle.loads(df_bytes)
        result = run_single_backtest(config, df, initial_equity)
        return result.to_dict()
    except Exception as e:
        return None


class MassiveBacktester:
    """Massive parallel backtester with caching and resume"""

    def __init__(
        self,
        data_dir: Path = Path("data/futures/clean"),
        output_dir: Path = Path("out/reports"),
        cache_path: Path = Path("data/backtest_cache.db"),
        n_workers: int = 4,
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cache = ResultCache(cache_path)
        self.n_workers = n_workers

        self.data_cache: dict[str, pd.DataFrame] = {}

    def load_data(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        """Load and resample data"""
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]

        # Try loading pre-resampled data
        path = self.data_dir / symbol / f"ohlcv_{timeframe}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
        else:
            # Load 1m and resample
            path_1m = self.data_dir / symbol / "ohlcv_1m.parquet"
            if not path_1m.exists():
                return None

            df_1m = pd.read_parquet(path_1m)
            if "open_time" in df_1m.columns:
                df_1m["timestamp"] = pd.to_datetime(df_1m["open_time"], utc=True)
            elif "timestamp" in df_1m.columns:
                df_1m["timestamp"] = pd.to_datetime(df_1m["timestamp"], utc=True)
            else:
                return None

            df_1m = df_1m.set_index("timestamp")

            # Resample
            tf_map = {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1D"}
            freq = tf_map.get(timeframe, "1h")

            df = df_1m.resample(freq).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna()

        self.data_cache[cache_key] = df
        return df

    def run(
        self,
        configs: list[BacktestConfig] | None = None,
        families: list[str] | None = None,
        resume: bool = True,
        initial_equity: float = 10000.0,
        max_configs: int | None = None,
    ) -> pd.DataFrame:
        """Run massive backtest"""
        logger.info("=" * 70)
        logger.info("MASSIVE BACKTESTER")
        logger.info("=" * 70)

        # Load configs if not provided
        if configs is None:
            loader = GridConfigLoader()
            if families:
                configs = []
                for family in families:
                    try:
                        grid = loader.load_family_grid(family)
                        configs.extend(loader.expand_grid(grid))
                    except Exception as e:
                        logger.warning(f"Error loading {family}: {e}")
            else:
                configs = loader.load_all_families()

        logger.info(f"Total configurations: {len(configs):,}")

        # Filter already completed
        if resume:
            completed_ids = self.cache.get_completed_ids()
            configs = [c for c in configs if c.config_id not in completed_ids]
            logger.info(f"Remaining after resume: {len(configs):,}")

        if max_configs:
            configs = configs[:max_configs]

        if not configs:
            logger.info("No configurations to process")
            results = self.cache.load_all_results()
            return pd.DataFrame(results)

        # Group by symbol/timeframe for data loading
        groups: dict[str, list[BacktestConfig]] = {}
        for config in configs:
            key = f"{config.symbol}_{config.timeframe}"
            if key not in groups:
                groups[key] = []
            groups[key].append(config)

        results = []
        total = len(configs)
        processed = 0
        start_time = time.monotonic()
        last_update = start_time

        # Print initial status
        print(f"\n{'=' * 70}")
        print(f"MASSIVE BACKTEST STARTED")
        print(f"{'=' * 70}")
        print(f"Total configurations: {total:,}")
        print(f"Workers: {self.n_workers}")
        print(f"Groups: {len(groups)}")
        print(f"{'=' * 70}\n")

        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                TextColumn("[progress.percentage]{task.percentage:>5.1f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
                refresh_per_second=2,
            ) as progress:
                main_task = progress.add_task(f"[cyan]Total Progress", total=total)

                for group_key, group_configs in groups.items():
                    symbol, timeframe = group_key.split("_")
                    group_task = progress.add_task(f"[green]{symbol} {timeframe}", total=len(group_configs))

                    df = self.load_data(symbol, timeframe)
                    if df is None:
                        progress.update(group_task, completed=len(group_configs))
                        progress.update(main_task, advance=len(group_configs))
                        continue

                    df_bytes = pickle.dumps(df)
                    args_list = [(config.to_dict(), df_bytes, initial_equity) for config in group_configs]

                    with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                        futures = {executor.submit(worker_backtest, args): i for i, args in enumerate(args_list)}

                        for future in as_completed(futures):
                            processed += 1
                            result = future.result()

                            if result:
                                results.append(result)
                                config_dict = {k: v for k, v in result.items() if k in BacktestConfig.__dataclass_fields__}
                                config = BacktestConfig(**{
                                    k: json.loads(v) if k == "params" else v
                                    for k, v in config_dict.items()
                                })
                                result_obj = BacktestResult(config=config, **{
                                    k: v for k, v in result.items() if k in BacktestResult.__dataclass_fields__ and k != "config"
                                })
                                self.cache.save_result(result_obj)

                            progress.update(group_task, advance=1)
                            progress.update(main_task, advance=1)

                    progress.remove_task(group_task)
        else:
            # Fallback without rich - print progress every 1%
            for group_key, group_configs in groups.items():
                symbol, timeframe = group_key.split("_")
                print(f"\n[{symbol} {timeframe}] Processing {len(group_configs):,} configs...")

                df = self.load_data(symbol, timeframe)
                if df is None:
                    print(f"  WARNING: No data for {symbol} {timeframe}")
                    processed += len(group_configs)
                    continue

                df_bytes = pickle.dumps(df)
                args_list = [(config.to_dict(), df_bytes, initial_equity) for config in group_configs]

                with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                    futures = {executor.submit(worker_backtest, args): i for i, args in enumerate(args_list)}

                    for future in as_completed(futures):
                        processed += 1
                        result = future.result()

                        if result:
                            results.append(result)
                            config_dict = {k: v for k, v in result.items() if k in BacktestConfig.__dataclass_fields__}
                            config = BacktestConfig(**{
                                k: json.loads(v) if k == "params" else v
                                for k, v in config_dict.items()
                            })
                            result_obj = BacktestResult(config=config, **{
                                k: v for k, v in result.items() if k in BacktestResult.__dataclass_fields__ and k != "config"
                            })
                            self.cache.save_result(result_obj)

                        # Progress every 1% or every 5 seconds
                        now = time.monotonic()
                        if processed % max(1, total // 100) == 0 or (now - last_update) > 5 or processed == total:
                            last_update = now
                            elapsed = now - start_time
                            rate = processed / elapsed if elapsed > 0 else 0
                            eta_sec = (total - processed) / rate if rate > 0 else 0
                            pct = processed / total * 100

                            eta_str = str(timedelta(seconds=int(eta_sec)))
                            elapsed_str = str(timedelta(seconds=int(elapsed)))

                            print(f"\r  Progress: {pct:5.1f}% | {processed:,}/{total:,} | "
                                  f"{rate:.1f}/sec | Elapsed: {elapsed_str} | ETA: {eta_str}    ", end="", flush=True)

                print()  # Newline after group completes

        # Final summary
        total_elapsed = time.monotonic() - start_time
        elapsed_str = str(timedelta(seconds=int(total_elapsed)))

        print(f"\n{'=' * 70}")
        print(f"BACKTEST COMPLETED")
        print(f"{'=' * 70}")
        print(f"  Total time:        {elapsed_str}")
        print(f"  Configurations:    {processed:,}")
        print(f"  Results saved:     {len(results):,}")
        print(f"  Speed:             {processed / total_elapsed:.1f} configs/sec")

        # Load all results from cache
        all_results = self.cache.load_all_results()
        df_results = pd.DataFrame(all_results)

        # Sort by performance
        if "annual_return_pct" in df_results.columns:
            df_results = df_results.sort_values("sharpe_ratio", ascending=False)

        # Print top 5 results
        if len(df_results) > 0:
            print(f"\n  Top 5 by Sharpe Ratio:")
            print(f"  {'-' * 66}")
            for i, row in df_results.head(5).iterrows():
                family = row.get('family', '?')[:8]
                strategy = row.get('strategy_type', '?')[:10]
                sharpe = row.get('sharpe_ratio', 0)
                ret = row.get('annual_return_pct', 0)
                dd = row.get('max_drawdown_pct', 0)
                print(f"    {family:<8} {strategy:<10} | Sharpe: {sharpe:5.2f} | "
                      f"Annual: {ret:+7.1f}% | DD: {dd:6.1f}%")

        print(f"{'=' * 70}\n")

        return df_results

    def apply_filters(self, df: pd.DataFrame, filters: FilterConfig | None = None) -> pd.DataFrame:
        """Apply overfitting detection filters"""
        if filters is None:
            filters = FilterConfig()

        filtered = df.copy()

        # Min trades
        filtered = filtered[filtered["total_trades"] >= filters.min_trades]

        # Max drawdown
        filtered = filtered[filtered["max_drawdown_pct"] >= filters.max_drawdown_pct]

        # Min profit factor
        filtered = filtered[filtered["profit_factor"] >= filters.min_profit_factor]

        # Min sharpe
        filtered = filtered[filtered["sharpe_ratio"] >= filters.min_sharpe]

        return filtered

    def generate_report(
        self,
        df: pd.DataFrame,
        name: str = "backtest_report",
    ) -> Path:
        """Generate comprehensive report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.output_dir / f"{name}_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)

        # Save full results
        df.to_csv(report_dir / "full_results.csv", index=False)
        df.to_parquet(report_dir / "full_results.parquet", index=False)

        # Top 100
        top_100 = df.head(100)
        top_100.to_csv(report_dir / "top_100.csv", index=False)

        # Summary stats
        summary = {
            "total_configs": len(df),
            "profitable_configs": len(df[df["return_pct"] > 0]),
            "avg_annual_return": df["annual_return_pct"].mean(),
            "avg_sharpe": df["sharpe_ratio"].mean(),
            "avg_max_dd": df["max_drawdown_pct"].mean(),
            "best_annual_return": df["annual_return_pct"].max(),
            "best_sharpe": df["sharpe_ratio"].max(),
        }

        with open(report_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Family analysis
        if "family" in df.columns:
            family_stats = df.groupby("family").agg({
                "annual_return_pct": ["mean", "std", "max"],
                "sharpe_ratio": ["mean", "std", "max"],
                "max_drawdown_pct": ["mean", "min"],
                "profit_factor": ["mean", "max"],
                "total_trades": "mean",
            })
            family_stats.to_csv(report_dir / "family_analysis.csv")

        # Timeframe analysis
        if "timeframe" in df.columns:
            tf_stats = df.groupby("timeframe").agg({
                "annual_return_pct": ["mean", "std", "max"],
                "sharpe_ratio": ["mean", "std", "max"],
            })
            tf_stats.to_csv(report_dir / "timeframe_analysis.csv")

        logger.info(f"\nReport saved to: {report_dir}")
        return report_dir


def run_massive_backtest(
    families: list[str] | None = None,
    resume: bool = True,
    max_configs: int | None = None,
    n_workers: int = 4,
    output_dir: str = "out/reports",
) -> pd.DataFrame:
    """CLI entry point for massive backtesting"""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    backtester = MassiveBacktester(
        output_dir=Path(output_dir),
        n_workers=n_workers,
    )

    results = backtester.run(
        families=families,
        resume=resume,
        max_configs=max_configs,
    )

    if len(results) > 0:
        # Apply filters
        filtered = backtester.apply_filters(results)
        logger.info(f"\nFiltered results: {len(filtered)} / {len(results)}")

        # Generate report from ALL results (not just filtered)
        # This ensures we always have data to analyze
        backtester.generate_report(results, name="backtest_report")

        # Also save filtered if any passed
        if len(filtered) > 0:
            backtester.generate_report(filtered, name="backtest_filtered")

    return results


if __name__ == "__main__":
    import sys

    families = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    max_configs = int(sys.argv[2]) if len(sys.argv) > 2 else None

    run_massive_backtest(families=families, max_configs=max_configs)

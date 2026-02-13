from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from trader.backtest.engine import BacktestConfig, BacktestEngine
from trader.backtest.metrics import summarize_performance
from trader.backtest.report import print_backtest_report
from trader.broker.base import OrderRequest
from trader.broker.live_binance import LiveBinanceBroker
from trader.broker.paper import PaperBroker
from trader.config import AppConfig
from trader.data.binance import BinanceDataClient
from trader.data.binance_live import BinanceLiveFeed
from trader.logging import setup_logging
from trader.notify import Notifier
from trader.optimize import (
    Optimizer,
    export_results,
    generate_parameter_grid,
    load_grid_yaml,
    load_result_file,
    run_candidate_backtest,
    select_parameter_sets,
)
from trader.storage import SQLiteStorage
from trader.runtime import RuntimeConfig, RuntimeEngine, RuntimeOrchestrator
from trader.risk.guards import RiskGuard
from trader.strategy.base import Bar, Strategy
from trader.strategy.bollinger import BollingerBandStrategy
from trader.strategy.ema_cross import EMACrossStrategy
from trader.strategy.macd import MACDStrategy
from trader.strategy.rsi import RSIStrategy

AVAILABLE_STRATEGIES = ["ema_cross", "rsi", "macd", "bollinger"]

app = typer.Typer(help="Binance trader CLI (backtest / optimize / replay / run / paper / live)")
console = Console()


def _parse_symbols(symbols: str) -> list[str]:
    parsed = [x.strip() for x in symbols.split(",") if x.strip()]
    if not parsed:
        raise typer.BadParameter("At least one symbol is required")
    return parsed


def _build_base_backtest_config(cfg: AppConfig) -> BacktestConfig:
    return BacktestConfig(
        symbol=cfg.symbol,
        timeframe=cfg.timeframe,
        initial_equity=cfg.initial_equity,
        leverage=cfg.leverage,
        order_type=cfg.order_type,
        execution_price_source=cfg.execution_price_source,
        slippage_bps=cfg.slippage_bps,
        maker_fee_bps=cfg.maker_fee_bps,
        taker_fee_bps=cfg.taker_fee_bps,
        sizing_mode=cfg.sizing_mode,
        fixed_notional_usdt=cfg.fixed_notional_usdt,
        equity_pct=cfg.equity_pct,
        atr_period=cfg.atr_period,
        atr_risk_pct=cfg.atr_risk_pct,
        atr_stop_multiple=cfg.atr_stop_multiple,
        enable_funding=cfg.enable_funding,
        db_path=cfg.db_path,
    )


def _is_testnet(cfg: AppConfig) -> bool:
    return cfg.binance_env == "testnet"


def _timeframe_seconds(timeframe: str) -> float:
    if timeframe.endswith("m"):
        return float(int(timeframe[:-1]) * 60)
    if timeframe.endswith("h"):
        return float(int(timeframe[:-1]) * 3600)
    if timeframe.endswith("d"):
        return float(int(timeframe[:-1]) * 86400)
    return 60.0


def _pct_text(value: float) -> str:
    return f"{float(value) * 100:.2f}%"


def _print_runtime_banner(*, cfg: AppConfig, runtime_cfg: RuntimeConfig) -> None:
    table = Table(title="Runtime Profile")
    table.add_column("key")
    table.add_column("value")
    table.add_row("mode", str(runtime_cfg.mode))
    table.add_row("BINANCE_ENV", str(runtime_cfg.binance_env))
    table.add_row("LIVE_TRADING", str(runtime_cfg.live_trading_enabled))
    table.add_row("dry_run", str(runtime_cfg.dry_run))
    table.add_row("preset", str(runtime_cfg.preset_name or "-"))
    table.add_row("sleep_mode", str(runtime_cfg.sleep_mode_enabled))
    table.add_row("allocation_pct", _pct_text(runtime_cfg.account_allocation_pct))
    table.add_row("leverage", str(cfg.leverage))
    table.add_row("daily_loss_limit", _pct_text(runtime_cfg.daily_loss_limit_pct))
    table.add_row("max_dd", _pct_text(cfg.max_drawdown_pct))
    table.add_row("risk_per_trade", _pct_text(runtime_cfg.risk_per_trade_pct))
    table.add_row("max_position_notional", f"{runtime_cfg.max_position_notional_usdt:.2f} USDT")
    table.add_row("protective_mode", str(runtime_cfg.protective_missing_policy))
    table.add_row(
        "sl/tp",
        (
            f"SL({runtime_cfg.sl_mode})={runtime_cfg.protective_stop_loss_pct:.4f} "
            f"TP({runtime_cfg.tp_mode})={runtime_cfg.protective_take_profit_pct:.4f}"
        ),
    )
    console.print(table)


def _sleep_mode_warnings(cfg: AppConfig) -> list[str]:
    warnings: list[str] = []
    if cfg.leverage > 2:
        warnings.append(f"leverage {cfg.leverage} > 2")
    if cfg.daily_loss_limit_pct > 0.02:
        warnings.append(f"daily_loss_limit_pct {_pct_text(cfg.daily_loss_limit_pct)} > 2%")
    if cfg.account_allocation_pct > 0.30:
        warnings.append(f"allocation {_pct_text(cfg.account_allocation_pct)} > 30%")
    if cfg.live_trading and cfg.binance_env == "mainnet":
        warnings.append("LIVE_TRADING=true on mainnet")
    return warnings


def _print_optimize_top(df: pd.DataFrame, title: str, metric: str, top: int) -> None:
    top_df = df.head(top)
    table = Table(title=title)
    table.add_column("rank")
    table.add_column("run_id")
    table.add_column("symbol")
    table.add_column(metric)
    table.add_column("objective")
    table.add_column("params")
    for _, row in top_df.iterrows():
        params = row.get("params_json", "-")
        metric_val = row.get(metric, row.get("metric_value", 0.0))
        table.add_row(
            str(row.get("rank", "-")),
            str(row.get("candidate_run_id", "-")),
            str(row.get("symbol", "-")),
            f"{float(metric_val):.6f}" if pd.notna(metric_val) else "-",
            f"{float(row.get('objective', 0.0)):.6f}" if pd.notna(row.get("objective")) else "-",
            str(params)[:120],
        )
    console.print(table)


def _parse_params_from_row(row: pd.Series) -> dict[str, Any]:
    raw = row.get("params_json")
    if isinstance(raw, str) and raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            out: dict[str, Any] = {}
            for token in raw.split(";"):
                if ":" in token:
                    k, v = token.split(":", 1)
                    out[k.strip()] = v.strip()
            return out
    params: dict[str, Any] = {}
    for col, val in row.items():
        if col.startswith("param_"):
            params[col.removeprefix("param_")] = val
    return params


def _coerce_param_types(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, str):
            text = v.strip()
            if text.lower() in {"true", "false"}:
                out[k] = text.lower() == "true"
                continue
            try:
                if "." in text:
                    out[k] = float(text)
                else:
                    out[k] = int(text)
                continue
            except ValueError:
                out[k] = text
        else:
            out[k] = v
    return out


def _load_strategy_params_from_source(
    *,
    params_from: str | None,
    params_rank: int,
    storage: SQLiteStorage,
) -> tuple[dict[str, Any], str | None]:
    if not params_from:
        return {}, None

    p = Path(params_from)
    if p.exists():
        df = load_result_file(p)
        if df.empty:
            raise typer.BadParameter(f"No rows in params file: {params_from}")
        row: pd.Series
        if "rank" in df.columns:
            ranked = df[df["rank"] == params_rank]
            row = ranked.iloc[0] if not ranked.empty else df.sort_values("rank").iloc[0]
        else:
            row = df.iloc[max(0, params_rank - 1)] if len(df) >= params_rank else df.iloc[0]
        strategy_name = str(row.get("strategy", "ema_cross"))
        return _coerce_param_types(_parse_params_from_row(row)), strategy_name

    row = storage.get_optimize_result_by_candidate_run_id(params_from)
    if row is not None:
        return _coerce_param_types(json.loads(row["params_json"])), row.get("strategy")

    run_cfg = storage.get_backtest_run_config(params_from)
    if run_cfg is not None:
        params = run_cfg.get("strategy_params") or {}
        strategy_name = run_cfg.get("strategy_name")
        if isinstance(params, dict):
            return _coerce_param_types(params), strategy_name

    raise typer.BadParameter(f"Unable to resolve params-from source: {params_from}")


def _build_strategy(
    *,
    strategy_name: str,
    params: dict[str, Any],
    cfg: AppConfig,
) -> Strategy:
    """
    Build a strategy instance based on name and parameters.

    Supported strategies: ema_cross, rsi, macd, bollinger
    """
    stop_loss_pct = float(params.get("stop_loss_pct", cfg.ema_stop_loss_pct))
    take_profit_pct = float(params.get("take_profit_pct", cfg.ema_take_profit_pct))
    allow_short = bool(params.get("allow_short", True))

    if strategy_name == "ema_cross":
        fast_len = int(params.get("fast_len", params.get("short_window", cfg.short_window)))
        slow_len = int(params.get("slow_len", params.get("long_window", cfg.long_window)))
        return EMACrossStrategy(
            short_window=fast_len,
            long_window=slow_len,
            allow_short=allow_short,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )

    if strategy_name == "rsi":
        period = int(params.get("period", 14))
        overbought = float(params.get("overbought", 70.0))
        oversold = float(params.get("oversold", 30.0))
        return RSIStrategy(
            period=period,
            overbought=overbought,
            oversold=oversold,
            allow_short=allow_short,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )

    if strategy_name == "macd":
        fast_period = int(params.get("fast_period", 12))
        slow_period = int(params.get("slow_period", 26))
        signal_period = int(params.get("signal_period", 9))
        use_histogram = bool(params.get("use_histogram", False))
        histogram_threshold = float(params.get("histogram_threshold", 0.0))
        return MACDStrategy(
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
            allow_short=allow_short,
            use_histogram=use_histogram,
            histogram_threshold=histogram_threshold,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )

    if strategy_name == "bollinger":
        period = int(params.get("period", 20))
        std_dev = float(params.get("std_dev", 2.0))
        mode = str(params.get("mode", "mean_reversion"))
        if mode not in ("mean_reversion", "breakout"):
            mode = "mean_reversion"
        return BollingerBandStrategy(
            period=period,
            std_dev=std_dev,
            mode=mode,  # type: ignore[arg-type]
            allow_short=allow_short,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )

    raise typer.BadParameter(f"Unsupported strategy: {strategy_name}. Available: {', '.join(AVAILABLE_STRATEGIES)}")


def _get_strategy_params(strategy_name: str, cfg: AppConfig) -> dict[str, Any]:
    """Get default parameters for a strategy based on config."""
    if strategy_name == "ema_cross":
        return {
            "fast_len": cfg.short_window,
            "slow_len": cfg.long_window,
            "stop_loss_pct": cfg.ema_stop_loss_pct,
            "take_profit_pct": cfg.ema_take_profit_pct,
        }
    if strategy_name == "rsi":
        return {
            "period": 14,
            "overbought": 70.0,
            "oversold": 30.0,
            "stop_loss_pct": cfg.ema_stop_loss_pct,
            "take_profit_pct": cfg.ema_take_profit_pct,
        }
    if strategy_name == "macd":
        return {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
            "use_histogram": False,
            "stop_loss_pct": cfg.ema_stop_loss_pct,
            "take_profit_pct": cfg.ema_take_profit_pct,
        }
    if strategy_name == "bollinger":
        return {
            "period": 20,
            "std_dev": 2.0,
            "mode": "mean_reversion",
            "stop_loss_pct": cfg.ema_stop_loss_pct,
            "take_profit_pct": cfg.ema_take_profit_pct,
        }
    return {}


@app.command()
def backtest(
    symbol: str = typer.Option("BTC/USDT", help="Market symbol, e.g. BTC/USDT"),
    timeframe: str = typer.Option("1h", help="Candle timeframe"),
    limit: int = typer.Option(300, min=100, help="Number of candles to fetch"),
    strategy: str = typer.Option("ema_cross", help="Strategy: ema_cross, rsi, macd, bollinger"),
) -> None:
    setup_logging()
    if strategy not in AVAILABLE_STRATEGIES:
        raise typer.BadParameter(f"Unknown strategy: {strategy}. Available: {', '.join(AVAILABLE_STRATEGIES)}")
    cfg = AppConfig.from_env().model_copy(update={"symbol": symbol, "timeframe": timeframe})
    strategy_params = _get_strategy_params(strategy, cfg)
    strategy_obj = _build_strategy(strategy_name=strategy, params=strategy_params, cfg=cfg)
    engine = BacktestEngine()
    backtest_config = replace(
        _build_base_backtest_config(cfg),
        strategy_name=strategy,
        strategy_params=strategy_params,
    )
    client = BinanceDataClient(testnet=_is_testnet(cfg))
    try:
        candles = client.fetch_ohlcv(symbol=cfg.symbol, timeframe=cfg.timeframe, limit=limit)
    finally:
        client.close()
    result = engine.run(candles=candles, strategy=strategy_obj, config=backtest_config)
    metrics = result.summary or summarize_performance(result.equity_curve, result.trades, cfg.initial_equity)
    print_backtest_report(result, metrics)


@app.command()
def optimize(
    strategy: str = typer.Option("ema_cross", help="Strategy name"),
    symbols: str = typer.Option(..., help="Comma-separated symbols, e.g. BTC/USDT,ETH/USDT"),
    timeframe: str = typer.Option("1h", help="Candle timeframe"),
    start: str = typer.Option(..., help="Backtest start, e.g. 2023-01-01"),
    end: str = typer.Option(..., help="Backtest end, e.g. 2025-01-01"),
    search: str = typer.Option("grid", help="Search type: grid | random"),
    grid: str = typer.Option(..., help="YAML grid file path"),
    metric: str = typer.Option("sharpe_like", help="Primary metric"),
    top: int = typer.Option(20, min=1, help="Top N to display"),
    export: str | None = typer.Option(None, help="Export path (.csv or .parquet)"),
    jobs: int = typer.Option(1, min=1, help="Parallel workers"),
    random_samples: int = typer.Option(100, min=1, help="Sample size for random search"),
    random_seed: int = typer.Option(42, help="Seed for random search"),
    walk_forward: bool = typer.Option(False, "--walk-forward", help="Enable rolling walk-forward"),
    train_days: int = typer.Option(180, min=1, help="Walk-forward train window days"),
    test_days: int = typer.Option(60, min=1, help="Walk-forward test window days"),
    top_per_train: int = typer.Option(10, min=1, help="Top K train params to evaluate on test"),
    constraints: str | None = typer.Option(None, help="Constraints, e.g. max_drawdown<=0.15,trades>=50"),
    score: str | None = typer.Option(None, help="Score expression, e.g. 0.6*win_rate + 0.4*profit_factor"),
) -> None:
    setup_logging()
    cfg = AppConfig.from_env().model_copy(update={"timeframe": timeframe})
    parsed_symbols = _parse_symbols(symbols)
    grid_map = load_grid_yaml(grid)
    all_params = generate_parameter_grid(grid_map)
    parameter_sets = select_parameter_sets(
        search_mode=search,
        all_params=all_params,
        random_samples=random_samples,
        random_seed=random_seed,
    )

    client = BinanceDataClient(testnet=_is_testnet(cfg))
    candles_by_symbol: dict[str, pd.DataFrame] = {}
    try:
        for sym in parsed_symbols:
            candles_by_symbol[sym] = client.fetch_ohlcv_range(
                symbol=sym, timeframe=timeframe, start=start, end=end
            )
    finally:
        client.close()

    base_bt_cfg = _build_base_backtest_config(cfg)
    output = Optimizer().run(
        strategy_name=strategy,
        symbols=parsed_symbols,
        timeframe=timeframe,
        candles_by_symbol=candles_by_symbol,
        parameter_sets=parameter_sets,
        metric=metric,
        top_n=top,
        base_backtest_config=base_bt_cfg,
        search_mode=search,
        jobs=jobs,
        constraints=constraints,
        score_expr=score,
        walk_forward=walk_forward,
        start=start,
        end=end,
        train_days=train_days,
        test_days=test_days,
        top_per_train=top_per_train,
    )

    console.print(f"optimize_run_id: {output.optimize_run_id}")
    if walk_forward:
        _print_optimize_top(output.train_top_results, "Walk-forward Train Top", metric, top)
        _print_optimize_top(output.test_top_results, "Walk-forward Test Top", metric, top)
    else:
        _print_optimize_top(output.top_results, "Optimize Top Results", metric, top)

    if export:
        export_results(output.results, export)
        console.print(f"Exported optimization results: {export}")


@app.command()
def run(
    mode: str = typer.Option("paper", help="paper | live"),
    symbol: str = typer.Option("BTC/USDT", help="Market symbol"),
    symbols: str | None = typer.Option(None, "--symbols", help="Comma-separated symbols, e.g. BTC/USDT,ETH/USDT"),
    timeframe: str = typer.Option("1m", help="Runtime timeframe"),
    env: str | None = typer.Option(None, "--env", help="Binance env override: mainnet | testnet"),
    preset: str | None = typer.Option(None, "--preset", help="Preset file/name (e.g. sleep_mode.yaml)"),
    sleep_mode: bool = typer.Option(False, "--sleep-mode", help="Apply sleep_mode preset defaults"),
    strategy: str = typer.Option("ema_cross", help="Strategy name"),
    data_mode: str = typer.Option("rest", help="Data mode: rest | websocket"),
    params_from: str | None = typer.Option(None, help="csv/parquet path or run_id source"),
    params_rank: int = typer.Option(1, min=1, help="Rank to select from params file"),
    max_bars: int = typer.Option(0, min=0, help="Stop after N closed bars (0 = infinite)"),
    poll_interval_sec: float | None = typer.Option(None, min=0.1, help="REST polling interval"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Log order payload only, do not send"),
    one_shot: bool = typer.Option(False, "--one-shot", help="Process one closed bar then exit"),
    halt_on_error: bool = typer.Option(False, "--halt-on-error", help="Halt immediately on runtime exception"),
    resume: bool = typer.Option(False, "--resume", help="Resume from saved runtime_state"),
    resume_run_id: str | None = typer.Option(None, "--resume-run-id", help="Specific run_id to resume"),
    state_save_every_n_bars: int = typer.Option(1, min=1, help="Persist runtime state every N bars"),
    feed_stall_seconds: float | None = typer.Option(None, min=0.0, help="Halt when closed-bar receive gap exceeds this many seconds"),
    bar_staleness_warn_seconds: float | None = typer.Option(None, min=0.0, help="Warn when bar timestamp staleness exceeds this many seconds"),
    bar_staleness_halt: bool | None = typer.Option(None, "--bar-staleness-halt/--no-bar-staleness-halt", help="Halt on stale bars when threshold is exceeded"),
    bar_staleness_halt_seconds: float | None = typer.Option(None, min=0.0, help="Optional halt threshold for stale bars (default: warn threshold)"),
    api_error_halt_threshold: int | None = typer.Option(None, min=1, help="Consecutive API errors before halt"),
    auto_protective: bool = typer.Option(True, "--auto-protective/--no-auto-protective", help="Auto-create SL/TP orders after entry"),
    run_stop_loss_pct: float | None = typer.Option(None, min=0.0, help="Protective stop loss pct"),
    run_take_profit_pct: float | None = typer.Option(None, min=0.0, help="Protective take profit pct"),
    capital: float | None = typer.Option(None, "--capital", min=1.0, help="Optional fixed budget cap (USDT)"),
    yes_i_understand_live_risk: bool = typer.Option(
        False,
        "--yes-i-understand-live-risk",
        help="Required when --mode live",
    ),
) -> None:
    if mode not in {"paper", "live"}:
        raise typer.BadParameter("mode must be one of: paper, live")
    if data_mode not in {"rest", "websocket"}:
        raise typer.BadParameter("data-mode must be one of: rest, websocket")
    if mode == "live" and not yes_i_understand_live_risk:
        raise typer.BadParameter("--yes-i-understand-live-risk is required for live mode")
    if env is not None and env.strip().lower() not in {"mainnet", "testnet"}:
        raise typer.BadParameter("--env must be one of: mainnet, testnet")

    parsed_symbols = _parse_symbols(symbols) if symbols else [symbol]
    primary_symbol = parsed_symbols[0]

    setup_logging()
    selected_preset = "sleep_mode" if sleep_mode else preset
    cfg = AppConfig.from_env(preset=selected_preset).model_copy(update={"symbol": primary_symbol, "timeframe": timeframe})
    update_payload: dict[str, Any] = {"sleep_mode": sleep_mode}
    if env is not None:
        env_norm = env.strip().lower()
        update_payload["binance_env"] = env_norm
        update_payload["binance_testnet"] = env_norm == "testnet"
    if capital is not None:
        update_payload["capital_limit_usdt"] = float(capital)
    cfg = cfg.model_copy(update=update_payload)
    storage = SQLiteStorage(cfg.db_path)

    loaded_params, loaded_strategy = _load_strategy_params_from_source(
        params_from=params_from,
        params_rank=params_rank,
        storage=storage,
    )
    strategy_name = loaded_strategy or strategy
    strategy_obj = _build_strategy(strategy_name=strategy_name, params=loaded_params, cfg=cfg)

    effective_sl_pct = float(
        run_stop_loss_pct
        if run_stop_loss_pct is not None
        else (
            loaded_params.get("stop_loss_pct", cfg.sl_pct)
            if cfg.sl_mode == "pct"
            else cfg.run_stop_loss_pct
        )
    )
    effective_tp_pct = float(
        run_take_profit_pct
        if run_take_profit_pct is not None
        else (
            loaded_params.get("take_profit_pct", cfg.tp_pct)
            if cfg.tp_mode == "pct"
            else cfg.run_take_profit_pct
        )
    )

    runtime_cfg = RuntimeConfig(
        mode=mode,  # type: ignore[arg-type]
        symbol=primary_symbol,
        timeframe=timeframe,
        fixed_notional_usdt=float(loaded_params.get("fixed_notional_usdt", cfg.run_fixed_notional_usdt)),
        atr_period=cfg.atr_period,
        max_bars=max_bars,
        dry_run=dry_run,
        one_shot=one_shot,
        halt_on_error=halt_on_error,
        resume=resume,
        resume_run_id=resume_run_id,
        state_save_every_n_bars=state_save_every_n_bars if state_save_every_n_bars > 0 else cfg.run_state_save_every_n_bars,
        enable_protective_orders=auto_protective and cfg.enable_protective_orders,
        require_protective_orders=cfg.require_protective_orders,
        protective_missing_policy=cfg.protective_missing_policy,
        api_error_halt_threshold=int(api_error_halt_threshold or cfg.api_error_halt_threshold),
        feed_stall_timeout_sec=(
            float(feed_stall_seconds)
            if feed_stall_seconds is not None
            else (float(cfg.feed_stall_seconds) if cfg.feed_stall_seconds > 0 else _timeframe_seconds(timeframe) * 3.0)
        ),
        bar_staleness_warn_sec=(
            float(bar_staleness_warn_seconds)
            if bar_staleness_warn_seconds is not None
            else float(cfg.bar_staleness_warn_seconds)
        ),
        bar_staleness_halt=(cfg.bar_staleness_halt if bar_staleness_halt is None else bool(bar_staleness_halt)),
        bar_staleness_halt_sec=(
            float(bar_staleness_halt_seconds)
            if bar_staleness_halt_seconds is not None
            else float(cfg.bar_staleness_halt_seconds)
        ),
        preflight_max_time_drift_ms=cfg.preflight_max_time_drift_ms,
        preflight_expected_leverage=cfg.leverage,
        preflight_expected_margin_mode=cfg.expected_margin_mode,
        protective_stop_loss_pct=effective_sl_pct,
        protective_take_profit_pct=effective_tp_pct,
        binance_env=cfg.binance_env,
        live_trading_enabled=cfg.live_trading,
        preset_name=cfg.preset_name,
        sleep_mode_enabled=sleep_mode,
        account_allocation_pct=cfg.account_allocation_pct,
        max_position_notional_usdt=cfg.max_position_notional_usdt,
        risk_per_trade_pct=cfg.risk_per_trade_pct,
        daily_loss_limit_pct=cfg.daily_loss_limit_pct,
        capital_limit_usdt=cfg.capital_limit_usdt,
        consec_loss_limit=cfg.consec_loss_limit,
        sl_mode=cfg.sl_mode,
        sl_atr_mult=cfg.sl_atr_mult,
        tp_mode=cfg.tp_mode,
        tp_atr_mult=cfg.tp_atr_mult,
        trailing_stop_enabled=cfg.trailing_stop_enabled,
        trail_pct=cfg.trail_pct,
        trail_atr_mult=cfg.trail_atr_mult,
        cooldown_bars_after_halt=cfg.cooldown_bars_after_halt,
        quiet_hours=(cfg.quiet_hours if sleep_mode else None),
        heartbeat_enabled=cfg.heartbeat_enabled,
        heartbeat_interval_minutes=cfg.heartbeat_interval_minutes,
    )
    risk_guard = RiskGuard(
        max_order_notional=cfg.max_order_notional,
        max_position_notional=cfg.max_position_notional_usdt,
        max_daily_loss=cfg.max_daily_loss,
        max_drawdown_pct=cfg.max_drawdown_pct,
        max_atr_pct=cfg.max_atr_pct,
        account_allocation_pct=cfg.account_allocation_pct,
        risk_per_trade_pct=cfg.risk_per_trade_pct,
        daily_loss_limit_pct=cfg.daily_loss_limit_pct,
        consec_loss_limit=cfg.consec_loss_limit,
        quiet_hours=(cfg.quiet_hours if sleep_mode else None),
        capital_limit_usdt=cfg.capital_limit_usdt,
    )
    notifier = Notifier(
        telegram_bot_token=cfg.telegram_bot_token.get_secret_value() if cfg.telegram_bot_token else None,
        telegram_chat_id=cfg.telegram_chat_id.get_secret_value() if cfg.telegram_chat_id else None,
        discord_webhook_url=cfg.discord_webhook_url.get_secret_value() if cfg.discord_webhook_url else None,
    )
    _print_runtime_banner(cfg=cfg, runtime_cfg=runtime_cfg)

    if mode == "paper":
        broker = PaperBroker(
            starting_cash=cfg.initial_equity,
            slippage_bps=cfg.slippage_bps,
            taker_fee_bps=cfg.taker_fee_bps,
            maker_fee_bps=cfg.maker_fee_bps,
        )
    else:
        if not cfg.binance_api_key or not cfg.binance_api_secret:
            raise typer.BadParameter("BINANCE_API_KEY and BINANCE_API_SECRET are required for live mode")
        broker = LiveBinanceBroker(
            api_key=cfg.binance_api_key.get_secret_value(),
            api_secret=cfg.binance_api_secret.get_secret_value(),
            testnet=_is_testnet(cfg),
            live_trading=cfg.live_trading,
            use_user_stream=cfg.use_user_stream,
            listenkey_renew_secs=cfg.listenkey_renew_secs,
        )

    feeds: dict[str, BinanceLiveFeed] = {}
    engines: dict[str, RuntimeEngine] = {}
    shared_run_id = uuid4().hex
    try:
        for sym in parsed_symbols:
            per_cfg = replace(runtime_cfg, symbol=sym)
            per_strategy = _build_strategy(strategy_name=strategy_name, params=loaded_params, cfg=cfg)
            per_feed = BinanceLiveFeed(
                symbol=sym,
                timeframe=timeframe,
                mode=data_mode,
                poll_interval_sec=poll_interval_sec or cfg.run_poll_interval_sec,
                testnet=_is_testnet(cfg),
                bootstrap_history_bars=(max_bars if (data_mode == "websocket" and max_bars > 0) else 0),
            )
            feeds[sym] = per_feed
            engines[sym] = RuntimeEngine(
                config=per_cfg,
                strategy=per_strategy,
                broker=broker,
                feed=per_feed,
                storage=storage,
                risk_guard=risk_guard,
                notifier=notifier,
                initial_equity=cfg.initial_equity,
                run_id=shared_run_id,
            )

        if len(parsed_symbols) == 1:
            result = engines[primary_symbol].run()
        else:
            orchestrator = RuntimeOrchestrator(
                engines=engines,
                feeds=feeds,
                max_bars=(max_bars if max_bars > 0 else None),
                account_risk_guard=risk_guard,
                account_initial_equity=cfg.initial_equity,
            )
            result = orchestrator.run()
        console.print(result)
    finally:
        for feed in feeds.values():
            feed.close()
        if hasattr(broker, "close"):
            broker.close()  # type: ignore[attr-defined]
        storage.close()


@app.command()
def arm_sleep(
    preset: str = typer.Option("sleep_mode", "--preset", help="Preset to validate for unattended operation"),
    env: str | None = typer.Option(None, "--env", help="mainnet | testnet"),
) -> None:
    setup_logging()
    cfg = AppConfig.from_env(preset=preset)
    if env is not None:
        env_norm = env.strip().lower()
        if env_norm not in {"mainnet", "testnet"}:
            raise typer.BadParameter("--env must be one of: mainnet, testnet")
        cfg = cfg.model_copy(update={"binance_env": env_norm, "binance_testnet": env_norm == "testnet"})

    checklist = Table(title="Sleep Mode Checklist")
    checklist.add_column("item")
    checklist.add_column("value")
    checklist.add_row("preset", str(cfg.preset_name or preset))
    checklist.add_row("BINANCE_ENV", str(cfg.binance_env))
    checklist.add_row("LIVE_TRADING", str(cfg.live_trading))
    checklist.add_row("allocation_pct", _pct_text(cfg.account_allocation_pct))
    checklist.add_row("leverage", str(cfg.leverage))
    checklist.add_row("daily_loss_limit_pct", _pct_text(cfg.daily_loss_limit_pct))
    checklist.add_row("max_drawdown_pct", _pct_text(cfg.max_drawdown_pct))
    checklist.add_row("risk_per_trade_pct", _pct_text(cfg.risk_per_trade_pct))
    checklist.add_row("max_position_notional", f"{cfg.max_position_notional_usdt:.2f}")
    checklist.add_row("protective_mode", str(cfg.protective_missing_policy))
    checklist.add_row("quiet_hours", str(cfg.quiet_hours or "-"))
    console.print(checklist)

    warnings = _sleep_mode_warnings(cfg)
    if warnings:
        warning_table = Table(title="Strong Warnings")
        warning_table.add_column("warning")
        for w in warnings:
            warning_table.add_row(w)
        console.print(warning_table)
    else:
        console.print("[green]No high-risk warning detected for current sleep profile.[/green]")


@app.command()
def doctor(
    env: str = typer.Option("testnet", "--env", help="mainnet | testnet"),
    symbol: str | None = typer.Option(None, help="Symbol for filter validation (default: SYMBOL env)"),
) -> None:
    requested_env = env.strip().lower()
    if requested_env not in {"mainnet", "testnet"}:
        raise typer.BadParameter("--env must be one of: mainnet, testnet")

    setup_logging()
    cfg = AppConfig.from_env()
    target_symbol = symbol or cfg.symbol
    is_testnet = requested_env == "testnet"

    broker = LiveBinanceBroker(
        api_key=cfg.binance_api_key.get_secret_value() if cfg.binance_api_key else "",
        api_secret=cfg.binance_api_secret.get_secret_value() if cfg.binance_api_secret else "",
        testnet=is_testnet,
        live_trading=False,
        use_user_stream=False,
    )
    try:
        ok, checks = broker.preflight_check(
            symbol=target_symbol,
            max_time_drift_ms=cfg.preflight_max_time_drift_ms,
            expected_leverage=cfg.leverage,
            expected_margin_mode=cfg.expected_margin_mode,
            include_leverage_margin=False,
        )
    finally:
        broker.close()

    table = Table(title=f"Doctor ({requested_env}) - auth/time/symbol checks")
    table.add_column("event")
    table.add_column("check")
    table.add_column("ok")
    table.add_column("detail")
    for row in checks:
        if not isinstance(row, dict):
            table.add_row("preflight_check", "-", "-", str(row))
            continue
        event_type = str(row.get("event_type", "preflight_check"))
        check_name = str(row.get("check", "-"))
        ok_text = "yes" if bool(row.get("ok", False)) else "no"
        detail = str(row.get("detail", "-"))
        if event_type == "preflight_environment":
            detail = (
                f"BINANCE_ENV={row.get('binance_env')} "
                f"base_url={row.get('base_url')} ws_url={row.get('ws_url')}"
            )
        elif event_type == "preflight_credentials":
            detail = (
                f"api_key_present={row.get('api_key_present')} api_key_len={row.get('api_key_len')} "
                f"api_secret_present={row.get('api_secret_present')} api_secret_len={row.get('api_secret_len')}"
            )
        elif event_type == "preflight_endpoint":
            endpoint = row.get("endpoint", "-")
            status = row.get("http_status", "unknown")
            detail = f"endpoint={endpoint} http_status={status}"
            if row.get("error_code") is not None:
                detail += f" error_code={row.get('error_code')}"
            base_detail = str(row.get("detail", "")).strip()
            if base_detail:
                detail += f" ({base_detail})"
        elif event_type == "preflight_auth_guidance":
            guide = row.get("guide")
            if isinstance(guide, list):
                detail = " | ".join(str(item) for item in guide)
        table.add_row(event_type, check_name, ok_text, detail)
    console.print(table)

    if not ok:
        console.print("[bold red]Doctor failed. Check endpoint/auth diagnostics above.[/bold red]")
        raise typer.Exit(code=1)
    console.print("[green]Doctor passed. No orders were sent.[/green]")


@app.command()
def replay(
    run_id: str | None = typer.Option(None, help="Candidate run_id to replay"),
    from_opt: str | None = typer.Option(None, help="Optimization result file (.csv/.parquet)"),
    top: int = typer.Option(20, min=1, help="Top N rows from --from-opt"),
    export: str | None = typer.Option(None, help="Export directory or report file path"),
) -> None:
    if not run_id and not from_opt:
        raise typer.BadParameter("Either --run-id or --from-opt is required")

    setup_logging()
    cfg = AppConfig.from_env()
    base_bt_cfg = _build_base_backtest_config(cfg)
    storage = SQLiteStorage(cfg.db_path)
    client = BinanceDataClient(testnet=_is_testnet(cfg))
    cache: dict[tuple[str, str, str, str], pd.DataFrame] = {}

    def fetch_cached(symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
        key = (symbol, timeframe, start, end)
        if key not in cache:
            cache[key] = client.fetch_ohlcv_range(symbol=symbol, timeframe=timeframe, start=start, end=end)
        return cache[key]

    try:
        if run_id:
            row = storage.get_optimize_result_by_candidate_run_id(run_id)
            if row is None:
                raise typer.BadParameter(f"run_id not found in optimize_results: {run_id}")
            params = json.loads(row["params_json"])
            symbol = row["symbol"]
            timeframe = row["timeframe"] or row.get("run_timeframe") or cfg.timeframe
            start = row["window_start"]
            end = row["window_end"]
            strategy_name = row.get("strategy", "ema_cross")
            candles = fetch_cached(symbol, timeframe, start, end)
            result = run_candidate_backtest(
                candles=candles,
                symbol=symbol,
                timeframe=timeframe,
                strategy_name=strategy_name,
                params=params,
                base_backtest_config=base_bt_cfg,
            )
            print_backtest_report(result, result.summary)
            if export:
                export_dir = Path(export)
                export_dir.mkdir(parents=True, exist_ok=True)
                trades_df = pd.DataFrame([t.__dict__ for t in result.trades])
                equity_df = pd.DataFrame({"equity": result.equity_curve})
                trades_df.to_csv(export_dir / f"{run_id}_replay_trades.csv", index=False)
                equity_df.to_csv(export_dir / f"{run_id}_replay_equity.csv", index=False)
                console.print(f"Exported replay artifacts to {export_dir}")
            return

        source_df = load_result_file(from_opt)
        if source_df.empty:
            console.print("No rows in optimization result file.")
            return
        if "rank" in source_df.columns:
            source_df = source_df.sort_values("rank", ascending=True)
        elif "objective" in source_df.columns:
            source_df = source_df.sort_values("objective", ascending=False)
        source_df = source_df.head(top).reset_index(drop=True)

        comparisons: list[dict[str, Any]] = []
        for _, row in source_df.iterrows():
            params = _parse_params_from_row(row)
            symbol = str(row.get("symbol", cfg.symbol))
            timeframe = str(row.get("timeframe", cfg.timeframe))
            window_start = str(row.get("window_start", ""))
            window_end = str(row.get("window_end", ""))
            if not window_start or not window_end or window_start == "nan" or window_end == "nan":
                continue
            strategy_name = str(row.get("strategy", "ema_cross"))
            primary_metric = str(row.get("primary_metric", "sharpe_like"))
            candles = fetch_cached(symbol, timeframe, window_start, window_end)
            replay_result = run_candidate_backtest(
                candles=candles,
                symbol=symbol,
                timeframe=timeframe,
                strategy_name=strategy_name,
                params=params,
                base_backtest_config=base_bt_cfg,
            )
            orig_obj = float(row.get("objective", float("nan")))
            replay_obj = float(replay_result.summary.get(primary_metric, 0.0))
            comparisons.append(
                {
                    "candidate_run_id": row.get("candidate_run_id", "-"),
                    "symbol": symbol,
                    "metric": primary_metric,
                    "orig_objective": orig_obj,
                    "replay_metric_value": replay_obj,
                    "delta": replay_obj - orig_obj if pd.notna(orig_obj) else float("nan"),
                }
            )

        report_df = pd.DataFrame(comparisons)
        console.print(report_df.to_string(index=False))
        if export:
            export_path = Path(export)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            if export_path.suffix.lower() == ".parquet":
                report_df.to_parquet(export_path, index=False)
            else:
                report_df.to_csv(export_path, index=False)
            console.print(f"Exported replay comparison: {export_path}")
    finally:
        client.close()
        storage.close()


@app.command()
def paper(
    symbol: str = typer.Option("BTC/USDT", help="Market symbol, e.g. BTC/USDT"),
    timeframe: str = typer.Option("1h", help="Candle timeframe"),
    limit: int = typer.Option(300, min=100, help="Number of candles to fetch"),
    starting_cash: float = typer.Option(10_000.0, min=100.0),
    trade_notional: float = typer.Option(1_000.0, min=10.0),
) -> None:
    setup_logging()
    cfg = AppConfig.from_env().model_copy(update={"symbol": symbol, "timeframe": timeframe})
    strategy = EMACrossStrategy(
        short_window=cfg.short_window,
        long_window=cfg.long_window,
        allow_short=False,
        stop_loss_pct=cfg.ema_stop_loss_pct,
        take_profit_pct=cfg.ema_take_profit_pct,
    )
    broker = PaperBroker(starting_cash=starting_cash)
    client = BinanceDataClient(testnet=_is_testnet(cfg))
    try:
        candles = client.fetch_ohlcv(symbol=cfg.symbol, timeframe=cfg.timeframe, limit=limit)
    finally:
        client.close()

    for row in candles.itertuples(index=False):
        close_price = float(row.close)
        broker.update_market_price(cfg.symbol, close_price)
        signal = strategy.on_bar(
            Bar(
                timestamp=row.timestamp,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=close_price,
                volume=float(row.volume),
            )
        )
        if signal in {"long", "buy"}:
            qty = trade_notional / close_price
            broker.place_order(OrderRequest(symbol=cfg.symbol, side="buy", amount=qty))
        elif signal in {"exit", "sell"}:
            pos = broker.get_position(cfg.symbol)
            qty = abs(pos.qty)
            if qty > 0:
                broker.place_order(OrderRequest(symbol=cfg.symbol, side="sell", amount=qty))

    console.print(broker.get_balance())


@app.command()
def live(
    symbol: str = typer.Option("BTC/USDT", help="Market symbol, e.g. BTC/USDT"),
    side: str = typer.Option("buy", help="Order side: buy/sell"),
    amount: float = typer.Option(..., min=0.0001, help="Order amount"),
    price: float | None = typer.Option(None, help="If provided, submit a limit order"),
    yes_i_understand_live_risk: bool = typer.Option(
        False,
        "--yes-i-understand-live-risk",
        help="Explicit confirmation before live trading",
    ),
) -> None:
    if side not in {"buy", "sell"}:
        raise typer.BadParameter("side must be one of: buy, sell")

    if not yes_i_understand_live_risk:
        console.print(
            "[bold red]Live trading is blocked.[/bold red] "
            "Add --yes-i-understand-live-risk after reviewing your settings."
        )
        raise typer.Exit(code=1)

    cfg = AppConfig.from_env()
    if not cfg.binance_api_key or not cfg.binance_api_secret:
        console.print("[bold red]BINANCE_API_KEY / BINANCE_API_SECRET are required for live mode.[/bold red]")
        raise typer.Exit(code=1)

    broker = LiveBinanceBroker(
        api_key=cfg.binance_api_key.get_secret_value(),
        api_secret=cfg.binance_api_secret.get_secret_value(),
        testnet=_is_testnet(cfg),
        live_trading=cfg.live_trading,
    )
    try:
        result = broker.place_order(
            OrderRequest(
                symbol=symbol,
                side=side,
                amount=amount,
                order_type="limit" if price is not None else "market",
                price=price,
            )
        )
        console.print(result)
    finally:
        broker.close()


@app.command()
def status(
    run_id: str | None = typer.Option(None, help="Runtime run_id"),
    latest: bool = typer.Option(False, "--latest", help="Use latest runtime/backtest run_id"),
    db_path: str | None = typer.Option(None, help="SQLite DB path (default: config DB_PATH)"),
    events: int = typer.Option(10, min=1, help="Recent events to display"),
    errors: int = typer.Option(5, min=1, help="Recent errors to display"),
) -> None:
    def _as_symbol_map(payload: Any, *, default_symbol: str) -> dict[str, dict[str, Any]]:
        if not isinstance(payload, dict):
            return {}
        if "symbol" in payload and isinstance(payload.get("symbol"), str):
            return {str(payload["symbol"]): dict(payload)}
        has_symbol_like_keys = any("/" in str(k) and isinstance(v, dict) for k, v in payload.items())
        if has_symbol_like_keys:
            return {str(k): dict(v) for k, v in payload.items() if isinstance(v, dict)}
        return {default_symbol: dict(payload)}

    cfg = AppConfig.from_env()
    storage = SQLiteStorage(db_path or cfg.db_path)
    try:
        target_run_id = run_id
        if latest or not target_run_id:
            target_run_id = storage.get_latest_run_id()
        if not target_run_id:
            raise typer.BadParameter("No run found. Provide --run-id or run trader first.")

        summary = storage.get_run_status(target_run_id)
        pos_map = _as_symbol_map(summary.get("open_positions") or {}, default_symbol=cfg.symbol)
        risk_map = _as_symbol_map(summary.get("risk_state") or {}, default_symbol=cfg.symbol)
        open_orders_raw = summary.get("open_orders") or {}
        open_orders_map = _as_symbol_map(open_orders_raw, default_symbol=cfg.symbol)
        default_symbol = next(iter(pos_map.keys()), cfg.symbol)
        pos = pos_map.get(default_symbol, {})
        risk_state = risk_map.get(default_symbol, {})

        overview = Table(title=f"Runtime Status: {target_run_id}")
        overview.add_column("key")
        overview.add_column("value")
        overview.add_row("updated_at", str(summary.get("updated_at", "-")))
        overview.add_row("last_bar_ts", str(summary.get("last_bar_ts", "-")))
        overview.add_row("symbols", ",".join(sorted(pos_map.keys())))
        overview.add_row("halted", str(risk_state.get("halted", False)))
        overview.add_row("halt_reason", str(risk_state.get("halt_reason", "")))
        overview.add_row("preset", str(risk_state.get("preset", "-")))
        overview.add_row("sleep_mode", str(risk_state.get("sleep_mode", False)))
        overview.add_row("env", str(risk_state.get("env", "-")))
        overview.add_row("live_trading", str(risk_state.get("live_trading", False)))
        overview.add_row("dry_run", str(risk_state.get("dry_run", False)))
        overview.add_row("budget_usdt", str(risk_state.get("budget_usdt", "-")))
        overview.add_row("allocation_pct", str(risk_state.get("allocation_pct", "-")))
        overview.add_row("current_exposure_notional", str(risk_state.get("current_exposure_notional", "-")))
        overview.add_row("daily_loss_remaining_usdt", str(risk_state.get("daily_loss_remaining_usdt", "-")))
        overview.add_row("drawdown_pct", str(risk_state.get("drawdown_pct", "-")))
        overview.add_row("max_dd_limit", str(risk_state.get("max_drawdown_pct_limit", "-")))
        overview.add_row("quiet_hours", str(risk_state.get("quiet_hours", "-")))
        overview.add_row("quiet_hours_active", str(risk_state.get("quiet_hours_active", False)))
        overview.add_row("trades", str(summary.get("trades_count", 0)))
        overview.add_row("orders", str(summary.get("orders_count", 0)))
        overview.add_row("fills", str(summary.get("fills_count", 0)))
        overview.add_row("net_pnl", f"{float(summary.get('trades_net_pnl', 0.0)):.4f}")
        console.print(overview)

        if risk_map:
            total_exposure = 0.0
            total_realized = 0.0
            min_daily_remaining: float | None = None
            halted_symbols = 0
            for sym, r in risk_map.items():
                if not isinstance(r, dict):
                    continue
                total_exposure += float(r.get("current_exposure_notional", 0.0) or 0.0)
                total_realized += float(r.get("realized_pnl", 0.0) or 0.0)
                remaining = float(r.get("daily_loss_remaining_usdt", 0.0) or 0.0)
                min_daily_remaining = remaining if min_daily_remaining is None else min(min_daily_remaining, remaining)
                if bool(r.get("halted", False)):
                    halted_symbols += 1
            account_table = Table(title="Account Risk Summary")
            account_table.add_column("key")
            account_table.add_column("value")
            account_table.add_row("symbols_total", str(len(risk_map)))
            account_table.add_row("symbols_halted", str(halted_symbols))
            account_table.add_row("exposure_notional_total", f"{total_exposure:.4f}")
            account_table.add_row("realized_pnl_total", f"{total_realized:.4f}")
            account_table.add_row(
                "daily_loss_remaining_min",
                f"{(min_daily_remaining if min_daily_remaining is not None else 0.0):.4f}",
            )
            console.print(account_table)

        if pos_map:
            sym_table = Table(title="Per-Symbol Summary")
            sym_table.add_column("symbol")
            sym_table.add_column("position_qty")
            sym_table.add_column("entry_price")
            sym_table.add_column("open_orders")
            sym_table.add_column("halted")
            sym_table.add_column("halt_reason")
            sym_table.add_column("signal")
            for sym in sorted(pos_map.keys()):
                p = pos_map.get(sym, {})
                r = risk_map.get(sym, {})
                oo = open_orders_map.get(sym, {})
                if not isinstance(oo, dict):
                    oo = {}
                open_count = len([k for k, v in oo.items() if not str(k).startswith("_") and isinstance(v, dict)])
                sym_table.add_row(
                    sym,
                    str(p.get("qty", 0.0)),
                    str(p.get("entry_price", 0.0)),
                    str(open_count),
                    str(r.get("halted", False)),
                    str(r.get("halt_reason", "")),
                    str(r.get("last_signal", "-")),
                )
            console.print(sym_table)

        events_rows = storage.list_recent_events_for_run(target_run_id, limit=events)
        if events_rows:
            event_table = Table(title=f"Recent Events ({events})")
            event_table.add_column("ts")
            event_table.add_column("event_type")
            event_table.add_column("summary")
            for row in events_rows:
                payload = row.get("payload", {})
                if isinstance(payload, dict):
                    slim = {k: payload[k] for k in payload if k not in {"run_id"}}
                    summary_text = json.dumps(slim, default=str)[:120]
                else:
                    summary_text = "-"
                event_table.add_row(str(row.get("ts", "-")), str(row.get("event_type", "-")), summary_text)
            console.print(event_table)

        error_rows = storage.list_recent_errors_for_run(target_run_id, limit=errors)
        if error_rows:
            error_table = Table(title=f"Recent Errors ({errors})")
            error_table.add_column("ts")
            error_table.add_column("event_type")
            error_table.add_column("summary")
            for row in error_rows:
                payload = row.get("payload", {})
                if isinstance(payload, dict):
                    summary_text = json.dumps(payload, default=str)[:140]
                else:
                    summary_text = "-"
                error_table.add_row(str(row.get("ts", "-")), str(row.get("event_type", "-")), summary_text)
            console.print(error_table)
    finally:
        storage.close()


@app.command()
def daemon(
    symbols: str = typer.Option("BTC/USDT", help="Comma-separated symbols, e.g. BTC/USDT,ETH/USDT"),
    strategy: str = typer.Option("ema_cross", help="Strategy: ema_cross, rsi, macd, bollinger"),
    timeframe: str = typer.Option("1m", help="Candle timeframe"),
    initial_equity: float = typer.Option(10_000.0, help="Starting paper equity (USDT)"),
    testnet: bool = typer.Option(True, help="Use Binance testnet"),
    data_dir: str = typer.Option("data", help="Directory for data storage"),
    no_prevent_sleep: bool = typer.Option(False, "--no-prevent-sleep", help="Don't prevent system sleep"),
) -> None:
    """
    Run 24/7 paper trading daemon.

    Continuously monitors real-time data and executes paper trades.
    Prevents system sleep and accumulates market data.
    Press Ctrl+C to stop gracefully.
    """
    from trader.daemon import DaemonConfig, TradingDaemon
    from pathlib import Path

    if strategy not in AVAILABLE_STRATEGIES:
        raise typer.BadParameter(f"Unknown strategy: {strategy}. Available: {', '.join(AVAILABLE_STRATEGIES)}")

    parsed_symbols = _parse_symbols(symbols)

    config = DaemonConfig(
        symbols=parsed_symbols,
        strategy=strategy,
        timeframe=timeframe,
        initial_equity=initial_equity,
        testnet=testnet,
        data_dir=Path(data_dir),
        prevent_sleep=not no_prevent_sleep,
    )

    daemon_instance = TradingDaemon(config)
    daemon_instance.run()


@app.command("compare")
def compare_strategies(
    symbol: str = typer.Option("BTC/USDT", help="Market symbol"),
    timeframe: str = typer.Option("1m", help="Candle timeframe"),
    initial_equity: float = typer.Option(10_000.0, help="Starting equity per strategy (USDT)"),
    testnet: bool = typer.Option(True, help="Use Binance testnet"),
    data_dir: str = typer.Option("data/multi_strategy", help="Directory for results"),
    no_prevent_sleep: bool = typer.Option(False, "--no-prevent-sleep", help="Don't prevent system sleep"),
    leaderboard_interval: int = typer.Option(10, help="Leaderboard display interval (minutes)"),
    save_interval: int = typer.Option(5, help="Data save interval (minutes)"),
) -> None:
    """
    Run multiple strategies simultaneously for comparison.

    Tests a matrix of strategy configurations (EMA, RSI, MACD, Bollinger)
    with different parameters to find the best performer.

    Results are saved to data_dir with:
    - leaderboard.csv: Ranked strategy performance
    - strategies/: Detailed results per strategy
    - market_data.parquet: Collected price data
    """
    from trader.multi_strategy_daemon import MultiStrategyConfig, MultiStrategyDaemon
    from pathlib import Path

    config = MultiStrategyConfig(
        symbol=symbol,
        timeframe=timeframe,
        initial_equity=initial_equity,
        testnet=testnet,
        data_dir=Path(data_dir),
        prevent_sleep=not no_prevent_sleep,
        leaderboard_interval_minutes=leaderboard_interval,
        save_interval_minutes=save_interval,
    )

    daemon = MultiStrategyDaemon(config)
    daemon.run()


@app.command("backtest-compare")
def backtest_compare(
    symbol: str = typer.Option("BTC/USDT", help="Market symbol"),
    timeframe: str = typer.Option("1m", help="Candle timeframe"),
    days: int = typer.Option(365, help="Number of days of historical data"),
    initial_equity: float = typer.Option(10_000.0, help="Starting equity per strategy (USDT)"),
    data_dir: str = typer.Option("data/backtest", help="Directory for results"),
) -> None:
    """
    Backtest multiple strategies on historical data.

    Downloads historical data from Binance and tests all 44 strategy
    configurations to find the best performer. Much faster than real-time
    testing - can test years of data in minutes.

    Examples:
        # Test 1 year of data
        python main.py backtest-compare --days 365

        # Test 3 years of ETH data
        python main.py backtest-compare --symbol ETH/USDT --days 1095

    Results are saved to data_dir with:
    - leaderboard.csv: Ranked strategy performance
    - strategies/: Detailed results per strategy
    """
    from trader.backtest_compare import BacktestConfig, MultiStrategyBacktester
    from pathlib import Path

    config = BacktestConfig(
        symbol=symbol,
        timeframe=timeframe,
        days=days,
        initial_equity=initial_equity,
        data_dir=Path(data_dir),
    )

    backtester = MultiStrategyBacktester(config)
    backtester.run()


@app.command("download-data")
def download_data(
    symbols: str = typer.Option("BTC/USDT,ETH/USDT,XRP/USDT", help="Comma-separated symbols"),
    timeframe: str = typer.Option("1m", help="Candle timeframe"),
    days: int = typer.Option(1095, help="Number of days to download (default: 3 years)"),
    cache_dir: str = typer.Option("data/historical", help="Directory to save data"),
) -> None:
    """
    Download and cache historical data from Binance.

    Downloads historical kline data for multiple symbols and saves to CSV cache.
    This data is used by backtest-compare for fast offline backtesting.

    Examples:
        # Download 3 years of BTC, ETH, XRP data (default)
        python main.py download-data

        # Download 1 year of specific symbols
        python main.py download-data --symbols BTC/USDT,ETH/USDT --days 365

        # Download 5 years of BTC only
        python main.py download-data --symbols BTC/USDT --days 1825
    """
    from trader.data.historical import download_multiple_symbols

    parsed_symbols = [s.strip() for s in symbols.split(",") if s.strip()]

    if not parsed_symbols:
        raise typer.BadParameter("At least one symbol is required")

    download_multiple_symbols(
        symbols=parsed_symbols,
        timeframe=timeframe,
        days=days,
        cache_dir=cache_dir,
    )


@app.command("futures-backtest")
def futures_backtest(
    symbol: str = typer.Option("BTCUSDT", help="Futures symbol"),
    data_dir: str = typer.Option("data/futures", help="Data directory"),
    output_dir: str = typer.Option("data/futures_backtest", help="Output directory"),
    initial_equity: float = typer.Option(10_000.0, help="Initial equity (USDT)"),
    timeframes: str = typer.Option("5m,15m,1h,4h", help="Comma-separated timeframes"),
    leverages: str = typer.Option("1,2,3,5,10", help="Comma-separated leverage values"),
) -> None:
    """
    Run comprehensive futures backtesting.

    Tests ALL combinations of:
    - 4 strategies (EMA, RSI, MACD, Bollinger)
    - Multiple parameters per strategy
    - Multiple timeframes (5m, 15m, 1h, 4h)
    - Multiple leverages (1x, 2x, 3x, 5x, 10x)
    - Long-only vs Long+Short
    - Multiple SL/TP combinations

    Includes futures-specific features:
    - Funding rate costs (8h intervals)
    - Leverage and margin simulation
    - Liquidation simulation

    Results saved to output_dir/results.csv

    Example:
        python main.py futures-backtest --timeframes 1h,4h --leverages 1,3,5
    """
    from trader.futures_backtest import run_futures_backtest

    run_futures_backtest(
        symbol=symbol,
        data_dir=data_dir,
        output_dir=output_dir,
        initial_equity=initial_equity,
        timeframes=timeframes,
        leverages=leverages,
    )


@app.command("mtf-backtest")
def mtf_backtest(
    symbol: str = typer.Option("BTCUSDT", help="Futures symbol"),
    days: int = typer.Option(365, help="Days of data to backtest"),
    data_dir: str = typer.Option("data/futures", help="Data directory"),
    output_dir: str = typer.Option("data/futures/mtf_results", help="Output directory"),
    leverages: str = typer.Option("1,3,5,10", help="Comma-separated leverage values"),
) -> None:
    """
    Run Multi-Timeframe (MTF) futures backtesting.

    This is the MOST REALISTIC backtesting approach:
    - Uses 1m as base, calculates 5m/15m/1h/4h in real-time
    - Multiple timeframe confirmation for entries/exits
    - Higher timeframe for trend, lower for entry timing

    MTF Strategies included:
    - TrendFollow_MTF: 4h trend + 1h pullback + 15m entry
    - MomentumBreakout_MTF: BB squeeze + volume breakout
    - MACDDivergence_MTF: 1h divergence + 15m confirmation
    - RSIMeanReversion_MTF: Extreme RSI + mean reversion
    - AdaptiveTrend_MTF: ADX-based mode switching

    Features:
    - Next-bar execution (no lookahead bias)
    - Stop loss / Take profit / Trailing stop
    - Funding rate costs
    - Liquidation simulation

    Example:
        python main.py mtf-backtest --days 365 --leverages 3,5,10
    """
    from trader.mtf_backtest import run_mtf_backtest

    leverage_list = [int(x.strip()) for x in leverages.split(",") if x.strip()]

    run_mtf_backtest(
        symbol=symbol,
        days=days,
        leverages=leverage_list,
        data_dir=data_dir,
        output_dir=output_dir,
    )


@app.command("mtf-ml")
def mtf_ml(
    symbol: str = typer.Option("BTCUSDT", help="Futures symbol"),
    days: int = typer.Option(90, help="Days of data"),
    strategy: str = typer.Option("TrendFollow", help="Strategy: TrendFollow, MACDDivergence, MomentumBreakout, RSIMeanReversion"),
    trials: int = typer.Option(50, help="Number of optimization trials"),
    leverage: int = typer.Option(3, help="Leverage"),
    data_dir: str = typer.Option("data/futures", help="Data directory"),
    output_dir: str = typer.Option("data/futures/ml_optimization", help="Output directory"),
) -> None:
    """
    Machine Learning optimization for MTF strategies.

    Uses Bayesian optimization to find optimal parameters:
    - Explores parameter space intelligently
    - Balances exploration vs exploitation
    - Composite objective: Sharpe + WinRate - Drawdown

    Example:
        python main.py mtf-ml --strategy TrendFollow --trials 50
    """
    from trader.mtf_advanced import run_ml_optimization

    run_ml_optimization(
        symbol=symbol,
        days=days,
        strategy_name=strategy,
        n_trials=trials,
        leverage=leverage,
        data_dir=data_dir,
        output_dir=output_dir,
    )


@app.command("mtf-walkforward")
def mtf_walkforward(
    symbol: str = typer.Option("BTCUSDT", help="Futures symbol"),
    strategy: str = typer.Option("TrendFollow", help="Strategy name"),
    train_days: int = typer.Option(60, help="Training window days"),
    test_days: int = typer.Option(30, help="Testing window days"),
    trials: int = typer.Option(30, help="Optimization trials per window"),
    leverage: int = typer.Option(3, help="Leverage"),
    data_dir: str = typer.Option("data/futures", help="Data directory"),
    output_dir: str = typer.Option("data/futures/walk_forward", help="Output directory"),
) -> None:
    """
    Walk-forward validation to prevent overfitting.

    Process:
    1. Split data into rolling train/test windows
    2. Optimize on train, validate on test (out-of-sample)
    3. Roll forward and repeat
    4. Aggregate results across all windows

    This is the GOLD STANDARD for validating trading strategies.
    If a strategy works in walk-forward, it's more likely to work live.

    Example:
        python main.py mtf-walkforward --strategy TrendFollow --train-days 60 --test-days 30
    """
    from trader.mtf_advanced import run_walk_forward

    run_walk_forward(
        symbol=symbol,
        strategy_name=strategy,
        train_days=train_days,
        test_days=test_days,
        n_trials=trials,
        leverage=leverage,
        data_dir=data_dir,
        output_dir=output_dir,
    )


@app.command("mtf-optimize")
def mtf_optimize(
    symbol: str = typer.Option("BTCUSDT", help="Futures symbol"),
    days: int = typer.Option(90, help="Days of data to optimize on"),
    data_dir: str = typer.Option("data/futures", help="Data directory"),
    output_dir: str = typer.Option("data/futures/optimization", help="Output directory"),
    leverages: str = typer.Option("3,5", help="Comma-separated leverage values"),
) -> None:
    """
    Optimize MTF strategies with grid search.

    Features:
    1. Grid search over strategy parameters
    2. Market regime detection (trending/ranging/volatile)
    3. Best strategy selection per regime
    4. Risk parameter optimization (SL/TP/holding period)

    Tests ALL combinations of:
    - Strategy parameters (ADX thresholds, RSI levels, etc.)
    - Risk parameters (stop loss, take profit, min holding)
    - Leverage levels

    Output:
    - optimization_*.csv: All results ranked by Sharpe ratio
    - regime_best_*.json: Best strategy for each market regime

    Example:
        python main.py mtf-optimize --days 90 --leverages 3,5
    """
    from trader.mtf_optimizer import run_mtf_optimization

    leverage_list = [int(x.strip()) for x in leverages.split(",") if x.strip()]

    run_mtf_optimization(
        symbol=symbol,
        days=days,
        leverages=leverage_list,
        data_dir=data_dir,
        output_dir=output_dir,
    )


@app.command("download-futures")
def download_futures(
    symbols: str = typer.Option("BTCUSDT,ETHUSDT", help="Comma-separated futures symbols (no slash)"),
    days: int = typer.Option(365, help="Number of days to download"),
    base_dir: str = typer.Option("data/futures", help="Output directory"),
    delay: float = typer.Option(0.25, help="Delay between requests in seconds (increase if rate limited)"),
    force: bool = typer.Option(False, "--force", help="Force re-download even if cache exists"),
    include_trades: bool = typer.Option(False, "--include-trades", help="Include aggTrades (very heavy)"),
    skip_ohlcv: bool = typer.Option(False, "--skip-ohlcv", help="Skip OHLCV download"),
    skip_funding: bool = typer.Option(False, "--skip-funding", help="Skip funding rate"),
    skip_mark: bool = typer.Option(False, "--skip-mark", help="Skip mark price"),
    skip_index: bool = typer.Option(False, "--skip-index", help="Skip index price"),
    skip_oi: bool = typer.Option(False, "--skip-oi", help="Skip open interest"),
    skip_ratio: bool = typer.Option(False, "--skip-ratio", help="Skip long/short ratio"),
) -> None:
    """
    Download USDT-M Futures data from Binance FAPI.

    Downloads comprehensive futures data for realistic backtesting:
    - OHLCV (1m klines, auto-resampled to 5m/15m/1h/4h)
    - Funding Rate (8h intervals)
    - Mark Price Klines (for liquidation simulation)
    - Index Price Klines (weighted spot average)
    - Open Interest History (market sentiment)
    - Long/Short Ratio (positioning data)
    - Aggregated Trades (optional, for slippage modeling)

    Data is saved in 3-tier structure:
    - raw/: Original API responses (CSV)
    - clean/: Validated & processed (Parquet)
    - meta/: Exchange info & manifests (JSON)

    Examples:
        # Download 1 year of BTC and ETH futures data
        python main.py download-futures --days 365

        # Download 6 months with aggregated trades
        python main.py download-futures --days 180 --include-trades

        # Quick download (OHLCV + funding only)
        python main.py download-futures --skip-mark --skip-index --skip-oi --skip-ratio
    """
    from pathlib import Path
    from trader.data.futures_data import FuturesDataConfig, FuturesDataDownloader

    parsed_symbols = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    if not parsed_symbols:
        raise typer.BadParameter("At least one symbol is required")

    config = FuturesDataConfig(
        symbols=parsed_symbols,
        days=days,
        base_dir=Path(base_dir),
        request_delay=delay,
        force_download=force,
        download_ohlcv=not skip_ohlcv,
        download_funding=not skip_funding,
        download_mark_price=not skip_mark,
        download_index_price=not skip_index,
        download_open_interest=not skip_oi,
        download_long_short_ratio=not skip_ratio,
        download_exchange_info=True,
        download_agg_trades=include_trades,
    )

    downloader = FuturesDataDownloader(config)
    downloader.download_all()


def main() -> None:
    app()


if __name__ == "__main__":
    main()

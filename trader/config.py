from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, SecretStr, model_validator


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _as_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _pct_as_fraction(value: str | float | int | None, default: float) -> float:
    if value is None:
        return default
    raw = float(value)
    if raw > 1.0:
        return raw / 100.0
    return raw


def _key_to_env_style(key: str) -> str:
    out = re.sub(r"[^A-Za-z0-9]+", "_", key.strip()).strip("_")
    return out.upper()


def _strip_inline_comment(value: str) -> str:
    in_single = False
    in_double = False
    for i, ch in enumerate(value):
        if ch == "'" and not in_double:
            in_single = not in_single
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            continue
        if ch == "#" and not in_single and not in_double:
            return value[:i].strip()
    return value.strip()


def _unquote_env_value(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and ((text[0] == '"' and text[-1] == '"') or (text[0] == "'" and text[-1] == "'")):
        return text[1:-1]
    return text


def _load_dotenv_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists() or not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        env_key = _key_to_env_style(key)
        parsed = _unquote_env_value(_strip_inline_comment(value))
        out[env_key] = parsed
    return out


class AppConfig(BaseModel):
    mode: Literal["backtest", "paper", "live"] = "backtest"
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    db_path: Path = Path("data/trader.db")
    initial_equity: float = Field(default=10_000.0, gt=0)
    leverage: float = Field(default=3.0, ge=1.0, le=125.0)
    order_type: Literal[
        "MARKET",
        "LIMIT",
        "STOP_MARKET",
        "TAKE_PROFIT_MARKET",
        "market",
        "limit",
        "stop_market",
        "take_profit_market",
    ] = "MARKET"
    execution_price_source: Literal["close", "next_open"] = "next_open"
    slippage_bps: float = Field(default=1.0, ge=0.0)
    taker_fee_bps: float = Field(default=5.0, ge=0.0)
    maker_fee_bps: float = Field(default=2.0, ge=0.0)
    sizing_mode: Literal["fixed_usdt", "percent_equity", "atr"] = "fixed_usdt"
    fixed_notional_usdt: float = Field(default=1_000.0, ge=0.0)
    equity_pct: float = Field(default=0.1, ge=0.0, le=1.0)
    atr_period: int = Field(default=14, ge=2)
    atr_risk_pct: float = Field(default=0.01, ge=0.0, le=1.0)
    atr_stop_multiple: float = Field(default=2.0, gt=0.0)
    enable_funding: bool = False
    short_window: int = Field(default=12, ge=2)
    long_window: int = Field(default=26, ge=3)
    ema_stop_loss_pct: float = Field(default=0.0, ge=0.0, le=1.0)
    ema_take_profit_pct: float = Field(default=0.0, ge=0.0, le=10.0)
    live_trading: bool = False
    use_user_stream: bool = False
    listenkey_renew_secs: int = Field(default=1800, ge=60)
    enable_protective_orders: bool = True
    run_stop_loss_pct: float = Field(default=0.0, ge=0.0, le=1.0)
    run_take_profit_pct: float = Field(default=0.0, ge=0.0, le=10.0)
    run_poll_interval_sec: float = Field(default=1.0, ge=0.1)
    run_state_save_every_n_bars: int = Field(default=1, ge=1)
    run_fixed_notional_usdt: float = Field(default=100.0, gt=0.0)
    max_order_notional: float = Field(default=1_000.0, gt=0.0)
    max_position_notional: float = Field(default=10_000.0, gt=0.0)
    max_daily_loss: float = Field(default=500.0, gt=0.0)
    max_drawdown_pct: float = Field(default=0.2, ge=0.0, le=1.0)
    max_atr_pct: float = Field(default=0.05, ge=0.0, le=1.0)
    api_error_halt_threshold: int = Field(default=3, ge=1)
    feed_stall_seconds: float = Field(default=0.0, ge=0.0)
    bar_staleness_warn_seconds: float = Field(default=0.0, ge=0.0)
    bar_staleness_halt: bool = False
    bar_staleness_halt_seconds: float = Field(default=0.0, ge=0.0)
    require_protective_orders: bool = True
    protective_missing_policy: Literal["halt", "recreate"] = "halt"
    preflight_max_time_drift_ms: int = Field(default=5_000, ge=0)
    expected_margin_mode: Literal["cross", "isolated"] | None = None
    telegram_bot_token: SecretStr | None = None
    telegram_chat_id: SecretStr | None = None
    discord_webhook_url: SecretStr | None = None
    binance_env: Literal["mainnet", "testnet"] = "testnet"
    binance_testnet: bool = True
    binance_api_key: SecretStr | None = None
    binance_api_secret: SecretStr | None = None

    # Sleep Mode related controls
    preset_name: str | None = None
    sleep_mode: bool = False
    account_allocation_pct: float = Field(default=0.2, ge=0.0, le=1.0)
    max_position_notional_usdt: float = Field(default=2_000.0, gt=0.0)
    risk_per_trade_pct: float = Field(default=0.005, ge=0.0, le=1.0)
    daily_loss_limit_pct: float = Field(default=0.02, ge=0.0, le=1.0)
    consec_loss_limit: int = Field(default=5, ge=1)
    sl_mode: Literal["pct", "atr"] = "pct"
    sl_pct: float = Field(default=0.01, ge=0.0, le=1.0)
    sl_atr_mult: float = Field(default=1.5, ge=0.0)
    tp_mode: Literal["pct", "atr"] = "pct"
    tp_pct: float = Field(default=0.02, ge=0.0, le=10.0)
    tp_atr_mult: float = Field(default=2.0, ge=0.0)
    trailing_stop_enabled: bool = False
    trail_pct: float = Field(default=0.0, ge=0.0, le=1.0)
    trail_atr_mult: float = Field(default=0.0, ge=0.0)
    cooldown_bars_after_halt: int = Field(default=0, ge=0)
    quiet_hours: str | None = None
    heartbeat_enabled: bool = False
    heartbeat_interval_minutes: int = Field(default=30, ge=1)
    capital_limit_usdt: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _validate_windows(self) -> "AppConfig":
        if self.long_window <= self.short_window:
            raise ValueError("LONG_WINDOW must be greater than SHORT_WINDOW")
        self.binance_testnet = self.binance_env == "testnet"
        self.protective_missing_policy = str(self.protective_missing_policy).lower()  # type: ignore[assignment]
        self.sl_mode = str(self.sl_mode).lower()  # type: ignore[assignment]
        self.tp_mode = str(self.tp_mode).lower()  # type: ignore[assignment]
        return self

    @classmethod
    def _resolve_preset_path(cls, preset: str | None) -> Path | None:
        if not preset:
            return None
        raw = Path(preset)
        candidates: list[Path] = []
        if raw.exists():
            candidates.append(raw)
        if raw.suffix:
            candidates.append(Path("config/presets") / raw.name)
        else:
            candidates.append(Path("config/presets") / f"{raw.name}.yaml")
            candidates.append(Path("config/presets") / f"{raw.name}.yml")
            candidates.append(Path("config/presets") / raw.name)
        for cand in candidates:
            if cand.exists() and cand.is_file():
                return cand
        return None

    @classmethod
    def _load_yaml_as_env_map(cls, path: Path | None) -> dict[str, Any]:
        if path is None or not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            return {}
        out: dict[str, Any] = {}
        for k, v in loaded.items():
            key = _key_to_env_style(str(k))
            out[key] = v
            if key == "TRAILING_STOP" and isinstance(v, dict):
                out["TRAILING_STOP_ENABLED"] = bool(v.get("enabled", False))
                out["TRAIL_PCT"] = v.get("trail_pct", 0.0)
                out["TRAIL_ATR_MULT"] = v.get("trail_atr_mult", 0.0)
        return out

    @classmethod
    def _get_with_env_override(cls, merged_defaults: dict[str, Any], env_key: str) -> Any:
        if env_key in os.environ:
            return os.environ[env_key]
        return merged_defaults.get(env_key)

    @classmethod
    def from_env(cls, *, preset: str | None = None) -> "AppConfig":
        builtins: dict[str, Any] = {
            "APP_MODE": "backtest",
            "SYMBOL": "BTC/USDT",
            "TIMEFRAME": "1h",
            "DB_PATH": "data/trader.db",
            "INITIAL_EQUITY": 10000.0,
            "LEVERAGE": 3.0,
            "ORDER_TYPE": "market",
            "EXECUTION_PRICE_SOURCE": "next_open",
            "SLIPPAGE_BPS": 1.0,
            "TAKER_FEE_BPS": 5.0,
            "MAKER_FEE_BPS": 2.0,
            "SIZING_MODE": "fixed_usdt",
            "FIXED_NOTIONAL_USDT": 1000.0,
            "EQUITY_PCT": 0.1,
            "ATR_PERIOD": 14,
            "ATR_RISK_PCT": 0.01,
            "ATR_STOP_MULTIPLE": 2.0,
            "ENABLE_FUNDING": False,
            "SHORT_WINDOW": 12,
            "LONG_WINDOW": 26,
            "EMA_STOP_LOSS_PCT": 0.0,
            "EMA_TAKE_PROFIT_PCT": 0.0,
            "LIVE_TRADING": False,
            "USE_USER_STREAM": False,
            "LISTENKEY_RENEW_SECS": 1800,
            "ENABLE_PROTECTIVE_ORDERS": True,
            "RUN_STOP_LOSS_PCT": 0.0,
            "RUN_TAKE_PROFIT_PCT": 0.0,
            "RUN_POLL_INTERVAL_SEC": 1.0,
            "RUN_STATE_SAVE_EVERY_N_BARS": 1,
            "RUN_FIXED_NOTIONAL_USDT": 100.0,
            "MAX_ORDER_NOTIONAL": 1000.0,
            "MAX_POSITION_NOTIONAL": 10000.0,
            "MAX_DAILY_LOSS": 500.0,
            "MAX_DRAWDOWN_PCT": 0.2,
            "MAX_ATR_PCT": 0.05,
            "API_ERROR_HALT_THRESHOLD": 3,
            "FEED_STALL_SECONDS": 0.0,
            "BAR_STALENESS_WARN_SECONDS": 0.0,
            "BAR_STALENESS_HALT": False,
            "BAR_STALENESS_HALT_SECONDS": 0.0,
            "REQUIRE_PROTECTIVE_ORDERS": True,
            "PROTECTIVE_MISSING_POLICY": "halt",
            "PREFLIGHT_MAX_TIME_DRIFT_MS": 5000,
            "EXPECTED_MARGIN_MODE": "",
            "BINANCE_ENV": "testnet",
            "BINANCE_TESTNET": True,
            # Sleep mode defaults
            "ACCOUNT_ALLOCATION_PCT": 0.2,
            "MAX_POSITION_NOTIONAL_USDT": 2000.0,
            "RISK_PER_TRADE_PCT": 0.005,
            "DAILY_LOSS_LIMIT_PCT": 0.02,
            "CONSEC_LOSS_LIMIT": 5,
            "PROTECTIVE_MODE": "HALT",
            "SL_MODE": "PCT",
            "SL_PCT": 0.01,
            "SL_ATR_MULT": 1.5,
            "TP_MODE": "PCT",
            "TP_PCT": 0.02,
            "TP_ATR_MULT": 2.0,
            "TRAILING_STOP_ENABLED": False,
            "TRAIL_PCT": 0.0,
            "TRAIL_ATR_MULT": 0.0,
            "COOLDOWN_BARS_AFTER_HALT": 0,
            "QUIET_HOURS": "",
            "HEARTBEAT_ENABLED": False,
            "HEARTBEAT_INTERVAL_MINUTES": 30,
            "CAPITAL_LIMIT_USDT": "",
        }

        requested_preset = preset or os.getenv("PRESET")
        preset_path = cls._resolve_preset_path(requested_preset)
        preset_defaults = cls._load_yaml_as_env_map(preset_path)

        config_file_raw = os.getenv("CONFIG_FILE")
        config_file_path = Path(config_file_raw) if config_file_raw else Path("config/config.yaml")
        config_defaults = cls._load_yaml_as_env_map(config_file_path if config_file_path.exists() else None)
        dotenv_file_raw = os.getenv("ENV_FILE")
        dotenv_path: Path | None = None
        if dotenv_file_raw:
            p = Path(dotenv_file_raw)
            dotenv_path = p if p.exists() and p.is_file() else None
        else:
            for cand in (Path(".env"), Path(".env.example")):
                if cand.exists() and cand.is_file():
                    dotenv_path = cand
                    break
        dotenv_defaults = _load_dotenv_file(dotenv_path) if dotenv_path is not None else {}

        merged = dict(builtins)
        merged.update(preset_defaults)
        merged.update(config_defaults)
        merged.update(dotenv_defaults)

        def v(name: str) -> Any:
            return cls._get_with_env_override(merged, name)

        env_mode = str(v("BINANCE_ENV") or "").strip().lower()
        if env_mode in {"mainnet", "testnet"}:
            binance_env = env_mode
        else:
            binance_env = "testnet" if _as_bool(str(v("BINANCE_TESTNET")), default=True) else "mainnet"

        key = str(v("BINANCE_API_KEY") or "").strip()
        secret = str(v("BINANCE_API_SECRET") or "").strip()
        telegram_bot_token = str(v("TELEGRAM_BOT_TOKEN") or "").strip()
        telegram_chat_id = str(v("TELEGRAM_CHAT_ID") or "").strip()
        discord_webhook_url = str(v("DISCORD_WEBHOOK_URL") or "").strip()

        protective_policy = str(v("PROTECTIVE_MODE") or v("PROTECTIVE_MISSING_POLICY") or "halt").strip().lower()
        if protective_policy not in {"halt", "recreate"}:
            protective_policy = "halt"

        capital_limit_raw = str(v("CAPITAL_LIMIT_USDT") or "").strip()
        capital_limit_usdt = float(capital_limit_raw) if capital_limit_raw else None

        return cls(
            mode=str(v("APP_MODE") or "backtest"),
            symbol=str(v("SYMBOL") or "BTC/USDT"),
            timeframe=str(v("TIMEFRAME") or "1h"),
            db_path=Path(str(v("DB_PATH") or "data/trader.db")),
            initial_equity=float(v("INITIAL_EQUITY") or 10000.0),
            leverage=float(v("LEVERAGE") or 3.0),
            order_type=str(v("ORDER_TYPE") or "market"),
            execution_price_source=str(v("EXECUTION_PRICE_SOURCE") or "next_open"),
            slippage_bps=float(v("SLIPPAGE_BPS") or 1.0),
            taker_fee_bps=float(v("TAKER_FEE_BPS") or 5.0),
            maker_fee_bps=float(v("MAKER_FEE_BPS") or 2.0),
            sizing_mode=str(v("SIZING_MODE") or "fixed_usdt"),
            fixed_notional_usdt=float(v("FIXED_NOTIONAL_USDT") or 1000.0),
            equity_pct=float(v("EQUITY_PCT") or 0.1),
            atr_period=int(v("ATR_PERIOD") or 14),
            atr_risk_pct=_pct_as_fraction(v("ATR_RISK_PCT"), 0.01),
            atr_stop_multiple=float(v("ATR_STOP_MULTIPLE") or 2.0),
            enable_funding=_as_bool(str(v("ENABLE_FUNDING")), default=False),
            short_window=int(v("SHORT_WINDOW") or 12),
            long_window=int(v("LONG_WINDOW") or 26),
            ema_stop_loss_pct=_pct_as_fraction(v("EMA_STOP_LOSS_PCT"), 0.0),
            ema_take_profit_pct=_pct_as_fraction(v("EMA_TAKE_PROFIT_PCT"), 0.0),
            live_trading=_as_bool(str(v("LIVE_TRADING")), default=False),
            use_user_stream=_as_bool(str(v("USE_USER_STREAM")), default=False),
            listenkey_renew_secs=int(v("LISTENKEY_RENEW_SECS") or 1800),
            enable_protective_orders=_as_bool(str(v("ENABLE_PROTECTIVE_ORDERS")), default=True),
            run_stop_loss_pct=_pct_as_fraction(v("RUN_STOP_LOSS_PCT"), 0.0),
            run_take_profit_pct=_pct_as_fraction(v("RUN_TAKE_PROFIT_PCT"), 0.0),
            run_poll_interval_sec=float(v("RUN_POLL_INTERVAL_SEC") or 1.0),
            run_state_save_every_n_bars=int(v("RUN_STATE_SAVE_EVERY_N_BARS") or 1),
            run_fixed_notional_usdt=float(v("RUN_FIXED_NOTIONAL_USDT") or 100.0),
            max_order_notional=float(v("MAX_ORDER_NOTIONAL") or 1000.0),
            max_position_notional=float(v("MAX_POSITION_NOTIONAL") or v("MAX_POSITION_NOTIONAL_USDT") or 10000.0),
            max_daily_loss=float(v("MAX_DAILY_LOSS") or 500.0),
            max_drawdown_pct=_pct_as_fraction(v("MAX_DRAWDOWN_PCT"), 0.2),
            max_atr_pct=_pct_as_fraction(v("MAX_ATR_PCT"), 0.05),
            api_error_halt_threshold=int(v("API_ERROR_HALT_THRESHOLD") or 3),
            feed_stall_seconds=float(v("FEED_STALL_SECONDS") or 0.0),
            bar_staleness_warn_seconds=float(v("BAR_STALENESS_WARN_SECONDS") or 0.0),
            bar_staleness_halt=_as_bool(str(v("BAR_STALENESS_HALT")), default=False),
            bar_staleness_halt_seconds=float(v("BAR_STALENESS_HALT_SECONDS") or 0.0),
            require_protective_orders=_as_bool(str(v("REQUIRE_PROTECTIVE_ORDERS")), default=True),
            protective_missing_policy=protective_policy,
            preflight_max_time_drift_ms=int(v("PREFLIGHT_MAX_TIME_DRIFT_MS") or 5000),
            expected_margin_mode=(str(v("EXPECTED_MARGIN_MODE") or "").strip().lower() or None),
            telegram_bot_token=SecretStr(telegram_bot_token) if telegram_bot_token else None,
            telegram_chat_id=SecretStr(telegram_chat_id) if telegram_chat_id else None,
            discord_webhook_url=SecretStr(discord_webhook_url) if discord_webhook_url else None,
            binance_env=binance_env,  # type: ignore[arg-type]
            binance_testnet=(binance_env == "testnet"),
            binance_api_key=SecretStr(key) if key else None,
            binance_api_secret=SecretStr(secret) if secret else None,
            preset_name=preset_path.name if preset_path else (str(requested_preset) if requested_preset else None),
            sleep_mode=_as_bool(str(v("SLEEP_MODE")), default=False),
            account_allocation_pct=_pct_as_fraction(v("ACCOUNT_ALLOCATION_PCT"), 0.2),
            max_position_notional_usdt=float(v("MAX_POSITION_NOTIONAL_USDT") or v("MAX_POSITION_NOTIONAL") or 2000.0),
            risk_per_trade_pct=_pct_as_fraction(v("RISK_PER_TRADE_PCT"), 0.005),
            daily_loss_limit_pct=_pct_as_fraction(v("DAILY_LOSS_LIMIT_PCT"), 0.02),
            consec_loss_limit=int(v("CONSEC_LOSS_LIMIT") or 5),
            sl_mode=str(v("SL_MODE") or "PCT").lower(),  # type: ignore[arg-type]
            sl_pct=_pct_as_fraction(v("SL_PCT"), 0.01),
            sl_atr_mult=float(v("SL_ATR_MULT") or 1.5),
            tp_mode=str(v("TP_MODE") or "PCT").lower(),  # type: ignore[arg-type]
            tp_pct=_pct_as_fraction(v("TP_PCT"), 0.02),
            tp_atr_mult=float(v("TP_ATR_MULT") or 2.0),
            trailing_stop_enabled=_as_bool(str(v("TRAILING_STOP_ENABLED")), default=False),
            trail_pct=_pct_as_fraction(v("TRAIL_PCT"), 0.0),
            trail_atr_mult=float(v("TRAIL_ATR_MULT") or 0.0),
            cooldown_bars_after_halt=int(v("COOLDOWN_BARS_AFTER_HALT") or 0),
            quiet_hours=(str(v("QUIET_HOURS")).strip() or None) if v("QUIET_HOURS") is not None else None,
            heartbeat_enabled=_as_bool(str(v("HEARTBEAT_ENABLED")), default=False),
            heartbeat_interval_minutes=int(v("HEARTBEAT_INTERVAL_MINUTES") or 30),
            capital_limit_usdt=capital_limit_usdt,
        )

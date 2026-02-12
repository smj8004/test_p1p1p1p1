from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from trader.backtest.engine import BacktestConfig, BacktestEngine
from trader.strategy.base import Bar, Strategy, StrategyPosition


class SequenceStrategy(Strategy):
    def __init__(self, signals: list[str]) -> None:
        self._signals = signals
        self._idx = 0

    def on_bar(self, bar: Bar, position: StrategyPosition | None = None) -> str:
        if self._idx < len(self._signals):
            signal = self._signals[self._idx]
        else:
            signal = "hold"
        self._idx += 1
        return signal


def _candles(
    closes: list[float],
    *,
    opens: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    funding_rates: list[float] | None = None,
) -> pd.DataFrame:
    opens = opens or closes
    highs = highs or [max(o, c) + 1.0 for o, c in zip(opens, closes)]
    lows = lows or [min(o, c) - 1.0 for o, c in zip(opens, closes)]
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=len(closes), freq="h", tz="UTC"),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [10_000.0 for _ in closes],
        }
    )
    if funding_rates is not None:
        frame["funding_rate"] = funding_rates
    return frame


def _base_config() -> BacktestConfig:
    return BacktestConfig(
        symbol="BTC/USDT",
        timeframe="1h",
        initial_equity=1_000.0,
        leverage=1.0,
        order_type="market",
        execution_price_source="close",
        slippage_bps=0.0,
        taker_fee_bps=0.0,
        maker_fee_bps=0.0,
        sizing_mode="fixed_usdt",
        fixed_notional_usdt=1_000.0,
        persist_to_db=False,
    )


def test_backtest_engine_returns_equity_curve_length() -> None:
    engine = BacktestEngine()
    result = engine.run(
        candles=_candles([100, 101, 102, 103]),
        strategy=SequenceStrategy(["hold", "hold", "hold", "hold"]),
        config=_base_config(),
    )
    assert len(result.equity_curve) == 4


def test_fee_is_reflected_in_pnl() -> None:
    engine = BacktestEngine()
    config = replace(_base_config(), taker_fee_bps=10.0)  # 0.10%
    result = engine.run(
        candles=_candles([100, 110]),
        strategy=SequenceStrategy(["long", "exit"]),
        config=config,
    )
    assert result.equity_curve[-1] == pytest.approx(1097.9)
    assert result.trades[0].fee_paid == pytest.approx(2.1)
    assert result.trades[0].net_pnl == pytest.approx(97.9)


def test_slippage_bps_reduces_performance() -> None:
    engine = BacktestEngine()
    no_slip = engine.run(
        candles=_candles([100, 110]),
        strategy=SequenceStrategy(["long", "exit"]),
        config=_base_config(),
    )
    slip = engine.run(
        candles=_candles([100, 110]),
        strategy=SequenceStrategy(["long", "exit"]),
        config=replace(_base_config(), slippage_bps=100.0),
    )
    assert no_slip.equity_curve[-1] == pytest.approx(1100.0)
    assert slip.equity_curve[-1] < no_slip.equity_curve[-1]


def test_leverage_scales_pnl_with_percent_equity_sizing() -> None:
    engine = BacktestEngine()
    cfg_x1 = replace(_base_config(), sizing_mode="percent_equity", equity_pct=0.5, leverage=1.0)
    cfg_x3 = replace(_base_config(), sizing_mode="percent_equity", equity_pct=0.5, leverage=3.0)
    result_x1 = engine.run(
        candles=_candles([100, 110]),
        strategy=SequenceStrategy(["long", "exit"]),
        config=cfg_x1,
    )
    result_x3 = engine.run(
        candles=_candles([100, 110]),
        strategy=SequenceStrategy(["long", "exit"]),
        config=cfg_x3,
    )
    assert result_x1.equity_curve[-1] == pytest.approx(1050.0)
    assert result_x3.equity_curve[-1] == pytest.approx(1150.0)


def test_long_position_loses_when_price_drops() -> None:
    engine = BacktestEngine()
    result = engine.run(
        candles=_candles([100, 90]),
        strategy=SequenceStrategy(["long", "exit"]),
        config=_base_config(),
    )
    assert result.equity_curve[-1] == pytest.approx(900.0)


def test_short_position_wins_when_price_drops() -> None:
    engine = BacktestEngine()
    result = engine.run(
        candles=_candles([100, 90]),
        strategy=SequenceStrategy(["short", "exit"]),
        config=_base_config(),
    )
    assert result.equity_curve[-1] == pytest.approx(1100.0)


def test_fixed_usdt_sizing_uses_expected_quantity() -> None:
    engine = BacktestEngine()
    result = engine.run(
        candles=_candles([100, 110]),
        strategy=SequenceStrategy(["long", "exit"]),
        config=replace(_base_config(), fixed_notional_usdt=250.0),
    )
    assert result.trades[0].qty == pytest.approx(2.5)
    assert result.equity_curve[-1] == pytest.approx(1025.0)


def test_percent_equity_sizing_uses_expected_quantity() -> None:
    engine = BacktestEngine()
    result = engine.run(
        candles=_candles([100, 110]),
        strategy=SequenceStrategy(["long", "exit"]),
        config=replace(_base_config(), sizing_mode="percent_equity", equity_pct=0.2, leverage=2.0),
    )
    assert result.trades[0].qty == pytest.approx(4.0)
    assert result.equity_curve[-1] == pytest.approx(1040.0)


def test_atr_sizing_computes_quantity_from_risk_budget() -> None:
    engine = BacktestEngine()
    candles = _candles(
        [100, 100, 110],
        highs=[105, 105, 115],
        lows=[95, 95, 105],
    )
    result = engine.run(
        candles=candles,
        strategy=SequenceStrategy(["hold", "long", "exit"]),
        config=replace(
            _base_config(),
            sizing_mode="atr",
            atr_period=2,
            atr_risk_pct=0.02,
            atr_stop_multiple=2.0,
        ),
    )
    assert result.trades[0].qty == pytest.approx(1.0)
    assert result.equity_curve[-1] == pytest.approx(1010.0)


def test_funding_fee_is_applied_when_enabled() -> None:
    engine = BacktestEngine()
    result = engine.run(
        candles=_candles([100, 100, 100], funding_rates=[0.0, 0.01, 0.0]),
        strategy=SequenceStrategy(["long", "hold", "exit"]),
        config=replace(_base_config(), enable_funding=True),
    )
    assert result.trades[0].funding_paid == pytest.approx(10.0)
    assert result.equity_curve[-1] == pytest.approx(990.0)


def test_sqlite_persistence_creates_tables_and_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "backtest.sqlite"
    engine = BacktestEngine()
    result = engine.run(
        candles=_candles([100, 110]),
        strategy=SequenceStrategy(["long", "exit"]),
        config=replace(_base_config(), persist_to_db=True, db_path=db_path),
    )
    conn = sqlite3.connect(db_path)
    try:
        run_count = conn.execute("SELECT COUNT(*) FROM backtest_runs WHERE run_id = ?", (result.run_id,)).fetchone()[0]
        order_count = conn.execute("SELECT COUNT(*) FROM orders WHERE run_id = ?", (result.run_id,)).fetchone()[0]
        fill_count = conn.execute("SELECT COUNT(*) FROM fills WHERE run_id = ?", (result.run_id,)).fetchone()[0]
        trade_count = conn.execute("SELECT COUNT(*) FROM trades WHERE run_id = ?", (result.run_id,)).fetchone()[0]
    finally:
        conn.close()

    assert run_count == 1
    assert order_count >= 2
    assert fill_count >= 2
    assert trade_count == 1


def test_summary_metrics_are_populated_from_equity_and_trades() -> None:
    engine = BacktestEngine()
    result = engine.run(
        candles=_candles([100, 110]),
        strategy=SequenceStrategy(["long", "exit"]),
        config=_base_config(),
    )
    assert {"final_equity", "total_return", "max_drawdown", "win_rate", "profit_factor", "sharpe_like"} <= set(
        result.summary.keys()
    )

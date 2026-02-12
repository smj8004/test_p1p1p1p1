from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from trader.backtest.engine import BacktestConfig
from trader.optimize import Optimizer, generate_parameter_grid


def _candles(rows: int = 80) -> pd.DataFrame:
    base = [100 + (i * 0.2) for i in range(rows)]
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC"),
            "open": base,
            "high": [x + 1.0 for x in base],
            "low": [x - 1.0 for x in base],
            "close": [x + (0.5 if i % 3 == 0 else -0.2) for i, x in enumerate(base)],
            "volume": [1000.0 for _ in range(rows)],
        }
    )


def test_generate_parameter_grid_count() -> None:
    grid = {
        "fast_len": [5, 8, 13],
        "slow_len": [50, 100],
        "leverage": [1, 3],
    }
    combos = generate_parameter_grid(grid)
    assert len(combos) == 12


def test_optimize_persists_results_to_db(tmp_path: Path) -> None:
    db_path = tmp_path / "optimize.sqlite"
    optimizer = Optimizer()
    base_cfg = BacktestConfig(
        symbol="BTC/USDT",
        timeframe="1h",
        initial_equity=10_000.0,
        leverage=1.0,
        order_type="market",
        execution_price_source="close",
        slippage_bps=0.0,
        taker_fee_bps=0.0,
        maker_fee_bps=0.0,
        persist_to_db=False,
        db_path=db_path,
    )
    params = [
        {"fast_len": 5, "slow_len": 30, "leverage": 1},
        {"fast_len": 8, "slow_len": 50, "leverage": 3},
    ]
    output = optimizer.run(
        strategy_name="ema_cross",
        symbols=["BTC/USDT"],
        timeframe="1h",
        candles_by_symbol={"BTC/USDT": _candles()},
        parameter_sets=params,
        metric="sharpe_like",
        top_n=5,
        base_backtest_config=base_cfg,
        search_mode="grid",
        jobs=1,
        start="2024-01-01",
        end="2024-02-01",
    )

    conn = sqlite3.connect(db_path)
    try:
        run_count = conn.execute(
            "SELECT COUNT(*) FROM optimize_runs WHERE optimize_run_id = ?",
            (output.optimize_run_id,),
        ).fetchone()[0]
        result_count = conn.execute(
            "SELECT COUNT(*) FROM optimize_results WHERE optimize_run_id = ?",
            (output.optimize_run_id,),
        ).fetchone()[0]
        candidate_run_id_count = conn.execute(
            "SELECT COUNT(*) FROM optimize_results WHERE optimize_run_id = ? AND candidate_run_id <> ''",
            (output.optimize_run_id,),
        ).fetchone()[0]
    finally:
        conn.close()

    assert run_count == 1
    assert result_count == 2
    assert candidate_run_id_count == 2

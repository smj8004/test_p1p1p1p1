from __future__ import annotations

from rich.console import Console
from rich.table import Table

from .engine import BacktestResult


def print_backtest_report(result: BacktestResult, metrics: dict[str, float]) -> None:
    table = Table(title="Backtest Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Trades", str(len(result.trades)))
    table.add_row("Final Equity", f"{metrics['final_equity']:.2f}")
    table.add_row("Total Return", f"{metrics['total_return'] * 100:.2f}%")
    table.add_row("Max Drawdown", f"{metrics['max_drawdown'] * 100:.2f}%")
    table.add_row("Win Rate", f"{metrics.get('win_rate', 0.0) * 100:.2f}%")
    profit_factor = metrics.get("profit_factor", 0.0)
    table.add_row("Profit Factor", "inf" if profit_factor == float("inf") else f"{profit_factor:.3f}")
    table.add_row("Sharpe-like", f"{metrics['sharpe_like']:.3f}")
    Console().print(table)

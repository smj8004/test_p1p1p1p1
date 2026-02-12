from __future__ import annotations

import numpy as np

from trader.backtest.engine import Trade


def summarize_performance(
    equity_curve: list[float],
    trades: list[Trade] | None = None,
    initial_equity: float | None = None,
) -> dict[str, float]:
    if not equity_curve:
        return {
            "final_equity": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe_like": 0.0,
            "trades": 0.0,
        }

    trades = trades or []
    arr = np.asarray(equity_curve, dtype=float)
    final_equity = float(arr[-1])
    start_equity = initial_equity if initial_equity is not None else float(arr[0])
    total_return = float((arr[-1] / start_equity) - 1.0) if start_equity != 0 else 0.0

    peaks = np.maximum.accumulate(arr)
    drawdowns = np.where(peaks > 0, (arr - peaks) / peaks, 0.0)
    max_drawdown = float(drawdowns.min())

    if len(arr) > 1:
        period_returns = np.diff(arr) / arr[:-1]
        std = float(period_returns.std())
        sharpe_like = float((period_returns.mean() / std) * np.sqrt(252)) if std > 0 else 0.0
    else:
        sharpe_like = 0.0

    net_pnls = [trade.net_pnl for trade in trades]
    if net_pnls:
        winning = [p for p in net_pnls if p > 0]
        losing = [p for p in net_pnls if p < 0]
        win_rate = len(winning) / len(net_pnls)
        gross_profit = float(sum(winning))
        gross_loss = float(sum(losing))
        if gross_loss < 0:
            profit_factor = gross_profit / abs(gross_loss)
        elif gross_profit > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0
    else:
        win_rate = 0.0
        profit_factor = 0.0

    return {
        "final_equity": final_equity,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "sharpe_like": sharpe_like,
        "trades": float(len(net_pnls)),
    }

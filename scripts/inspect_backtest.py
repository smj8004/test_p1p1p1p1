from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

try:
    from rich.console import Console
    from rich.table import Table

    HAS_RICH = True
except Exception:
    HAS_RICH = False
    Console = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]


def _default_db_path() -> str:
    try:
        from trader.config import AppConfig

        return str(AppConfig().db_path)
    except Exception:
        return "data/trader.db"


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(r[1]) for r in rows}


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _print_table(title: str, columns: list[str], rows: list[list[Any]]) -> None:
    if HAS_RICH:
        console = Console()
        table = Table(title=title)
        for col in columns:
            table.add_column(col)
        for row in rows:
            table.add_row(*[_format_value(v) for v in row])
        console.print(table)
        return

    print(f"\n{title}")
    widths = [len(c) for c in columns]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(_format_value(cell)))
    header = " | ".join(c.ljust(widths[i]) for i, c in enumerate(columns))
    sep = "-+-".join("-" * w for w in widths)
    print(header)
    print(sep)
    for row in rows:
        print(" | ".join(_format_value(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def _safe_json_loads(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _compact_params(config_json: dict[str, Any]) -> str:
    keys = [
        "leverage",
        "sizing_mode",
        "fixed_notional_usdt",
        "equity_pct",
        "slippage_bps",
        "taker_fee_bps",
    ]
    parts = []
    for key in keys:
        if key in config_json:
            parts.append(f"{key}={config_json[key]}")
    return ", ".join(parts) if parts else "-"


def show_latest_runs(conn: sqlite3.Connection, limit: int = 5) -> None:
    if not _table_exists(conn, "backtest_runs"):
        print("No table: backtest_runs")
        return

    columns = _table_columns(conn, "backtest_runs")
    ts_col = "started_at" if "started_at" in columns else "created_at"
    rows = conn.execute(
        f"""
        SELECT run_id, {ts_col}, symbol, timeframe, config_json
        FROM backtest_runs
        ORDER BY {ts_col} DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    out_rows: list[list[Any]] = []
    for row in rows:
        cfg = _safe_json_loads(row["config_json"])
        strategy = cfg.get("strategy") or cfg.get("strategy_name") or "-"
        notes = cfg.get("notes") or _compact_params(cfg)
        out_rows.append([row["run_id"], row[ts_col], row["symbol"], row["timeframe"], strategy, notes])

    _print_table(
        title=f"Recent Backtest Runs (latest {limit})",
        columns=["run_id", "started_at", "symbol", "timeframe", "strategy", "notes/params"],
        rows=out_rows,
    )


def _fetch_rows(conn: sqlite3.Connection, table: str, run_id: str, order_by: str) -> list[sqlite3.Row]:
    if not _table_exists(conn, table):
        return []
    return conn.execute(
        f"SELECT * FROM {table} WHERE run_id = ? ORDER BY {order_by}",
        (run_id,),
    ).fetchall()


def _row_get(row: sqlite3.Row, names: list[str], default: Any = None) -> Any:
    for name in names:
        if name in row.keys():
            return row[name]
    return default


def show_run_details(conn: sqlite3.Connection, run_id: str, export_dir: Path | None = None) -> None:
    trades_rows = _fetch_rows(conn, "trades", run_id, "id ASC")
    orders_rows = _fetch_rows(conn, "orders", run_id, "id ASC")
    fills_rows = _fetch_rows(conn, "fills", run_id, "id ASC")

    trades_out: list[dict[str, Any]] = []
    for row in trades_rows:
        pnl_pct = _row_get(row, ["pnl_pct"], None)
        if pnl_pct is None:
            return_pct = _row_get(row, ["return_pct"], 0.0)
            pnl_pct = float(return_pct) * 100.0
        trades_out.append(
            {
                "entry_ts": _row_get(row, ["entry_ts"]),
                "exit_ts": _row_get(row, ["exit_ts"]),
                "direction": _row_get(row, ["direction", "side"]),
                "entry_price": _row_get(row, ["entry_price"]),
                "exit_price": _row_get(row, ["exit_price"]),
                "qty": _row_get(row, ["qty"]),
                "fee": _row_get(row, ["fee", "fee_paid"], 0.0),
                "pnl": _row_get(row, ["pnl", "net_pnl", "gross_pnl"], 0.0),
                "pnl_pct": pnl_pct,
                "reason": _row_get(row, ["reason"], "-"),
            }
        )

    orders_out: list[dict[str, Any]] = []
    for row in orders_rows:
        orders_out.append(
            {
                "ts": _row_get(row, ["ts", "created_at"]),
                "side": _row_get(row, ["side"]),
                "type": _row_get(row, ["type", "order_type"]),
                "qty": _row_get(row, ["qty"]),
                "price": _row_get(row, ["price", "requested_price"]),
                "status": _row_get(row, ["status"]),
                "client_order_id": _row_get(row, ["client_order_id"], "-"),
            }
        )

    fills_out: list[dict[str, Any]] = []
    for row in fills_rows:
        fills_out.append(
            {
                "ts": _row_get(row, ["ts"]),
                "order_id": _row_get(row, ["order_id"]),
                "price": _row_get(row, ["price"]),
                "qty": _row_get(row, ["qty"]),
                "fee": _row_get(row, ["fee"]),
            }
        )

    _print_table(
        title=f"Trades for run {run_id}",
        columns=["entry_ts", "exit_ts", "direction", "entry_price", "exit_price", "qty", "fee", "pnl", "pnl_pct", "reason"],
        rows=[[d[c] for c in ["entry_ts", "exit_ts", "direction", "entry_price", "exit_price", "qty", "fee", "pnl", "pnl_pct", "reason"]] for d in trades_out],
    )
    _print_table(
        title=f"Orders for run {run_id}",
        columns=["ts", "side", "type", "qty", "price", "status", "client_order_id"],
        rows=[[d[c] for c in ["ts", "side", "type", "qty", "price", "status", "client_order_id"]] for d in orders_out],
    )
    _print_table(
        title=f"Fills for run {run_id}",
        columns=["ts", "order_id", "price", "qty", "fee"],
        rows=[[d[c] for c in ["ts", "order_id", "price", "qty", "fee"]] for d in fills_out],
    )

    if export_dir is not None:
        export_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(export_dir / f"{run_id}_trades.csv", trades_out)
        _write_csv(export_dir / f"{run_id}_orders.csv", orders_out)
        _write_csv(export_dir / f"{run_id}_fills.csv", fills_out)
        print(f"Exported CSV files to: {export_dir}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def resolve_latest_run_id(conn: sqlite3.Connection) -> str | None:
    if not _table_exists(conn, "backtest_runs"):
        return None
    columns = _table_columns(conn, "backtest_runs")
    ts_col = "started_at" if "started_at" in columns else "created_at"
    row = conn.execute(
        f"SELECT run_id FROM backtest_runs ORDER BY {ts_col} DESC LIMIT 1"
    ).fetchone()
    return None if row is None else str(row["run_id"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect backtest runs, orders, fills, and trades from SQLite.")
    parser.add_argument("--db-path", default=_default_db_path(), help="Path to sqlite DB")
    parser.add_argument("--latest", action="store_true", help="Show recent backtest runs")
    parser.add_argument("--limit", type=int, default=5, help="Number of latest runs to show")
    parser.add_argument("--run-id", help="Run ID to inspect detailed trades/orders/fills")
    parser.add_argument("--export-csv", help="Directory to export trades/orders/fills CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"DB not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        if args.run_id:
            export_dir = Path(args.export_csv) if args.export_csv else None
            show_run_details(conn, args.run_id, export_dir=export_dir)
            return

        show_latest_runs(conn, limit=args.limit)
        if args.latest:
            return

        latest_run_id = resolve_latest_run_id(conn)
        if latest_run_id:
            print(f"\nTip: use --run-id {latest_run_id} to inspect details.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

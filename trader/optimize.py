from __future__ import annotations

import ast
import itertools
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import yaml

from trader.backtest.engine import BacktestConfig, BacktestEngine
from trader.storage import SQLiteStorage
from trader.strategy.ema_cross import EMACrossStrategy


@dataclass(frozen=True)
class OptimizeRunOutput:
    optimize_run_id: str
    results: pd.DataFrame
    top_results: pd.DataFrame
    train_top_results: pd.DataFrame
    test_top_results: pd.DataFrame


_CONSTRAINT_PATTERN = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(<=|>=|==|!=|<|>)\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$"
)
_ALLOWED_FUNCS = {"abs": abs, "min": min, "max": max}
_ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.Constant,
    ast.Load,
    ast.Name,
    ast.Call,
)


def _to_utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def load_grid_yaml(path: str | Path) -> dict[str, list[Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Grid YAML must be a mapping of parameter -> list")
    grid: dict[str, list[Any]] = {}
    for key, value in loaded.items():
        if isinstance(value, list):
            grid[str(key)] = value
        else:
            grid[str(key)] = [value]
    return grid


def generate_parameter_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def select_parameter_sets(
    *,
    search_mode: str,
    all_params: list[dict[str, Any]],
    random_samples: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    if search_mode == "grid":
        return all_params
    if search_mode != "random":
        raise ValueError(f"Unsupported search mode: {search_mode}")
    if random_samples <= 0:
        raise ValueError("random_samples must be > 0 for random search")
    rng = random.Random(random_seed)
    k = min(random_samples, len(all_params))
    return rng.sample(all_params, k)


def parse_constraints(constraints: str | None) -> list[tuple[str, str, float]]:
    if constraints is None or not constraints.strip():
        return []
    parsed: list[tuple[str, str, float]] = []
    for token in constraints.split(","):
        token = token.strip()
        if not token:
            continue
        match = _CONSTRAINT_PATTERN.match(token)
        if match is None:
            raise ValueError(f"Invalid constraint: {token}")
        metric, op, threshold = match.groups()
        parsed.append((metric, op, float(threshold)))
    return parsed


def _check_constraint(value: float, op: str, threshold: float) -> bool:
    if op == "<=":
        return value <= threshold
    if op == ">=":
        return value >= threshold
    if op == "<":
        return value < threshold
    if op == ">":
        return value > threshold
    if op == "==":
        return value == threshold
    if op == "!=":
        return value != threshold
    raise ValueError(f"Unsupported operator: {op}")


def constraints_pass(metrics: dict[str, float], parsed_constraints: list[tuple[str, str, float]]) -> bool:
    for metric, op, threshold in parsed_constraints:
        value = metrics.get(metric)
        if value is None:
            return False
        if not _check_constraint(float(value), op, threshold):
            return False
    return True


def _validate_score_expression(expression: str) -> ast.Expression:
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise ValueError(f"Unsupported token in score expression: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
                raise ValueError("Only abs/min/max calls are allowed in score expression")
    return tree


def eval_score_expression(expression: str, metrics: dict[str, float]) -> float:
    tree = _validate_score_expression(expression)
    env: dict[str, Any] = {}
    env.update(_ALLOWED_FUNCS)
    env.update(metrics)
    value = eval(compile(tree, "<score>", "eval"), {"__builtins__": {}}, env)
    return float(value)


def _build_strategy(strategy_name: str, params: dict[str, Any]) -> Any:
    if strategy_name != "ema_cross":
        raise ValueError(f"Unsupported strategy: {strategy_name}")
    fast_len = int(params.get("fast_len", params.get("short_window", 12)))
    slow_len = int(params.get("slow_len", params.get("long_window", 26)))
    stop_loss_pct = float(params.get("stop_loss_pct", 0.0))
    take_profit_pct = float(params.get("take_profit_pct", 0.0))
    return EMACrossStrategy(
        short_window=fast_len,
        long_window=slow_len,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )


def _build_backtest_config(
    *,
    base: BacktestConfig,
    symbol: str,
    timeframe: str,
    strategy_name: str,
    params: dict[str, Any],
) -> BacktestConfig:
    taker_fee_bps = float(params.get("fee_taker_bps", params.get("taker_fee_bps", base.taker_fee_bps)))
    maker_fee_bps = float(params.get("fee_maker_bps", params.get("maker_fee_bps", base.maker_fee_bps)))
    slippage_bps = float(params.get("slippage_bps", base.slippage_bps))
    leverage = float(params.get("leverage", base.leverage))
    return replace(
        base,
        symbol=symbol,
        timeframe=timeframe,
        taker_fee_bps=taker_fee_bps,
        maker_fee_bps=maker_fee_bps,
        slippage_bps=slippage_bps,
        leverage=leverage,
        strategy_name=strategy_name,
        persist_to_db=False,
    )


def run_candidate_backtest(
    *,
    candles: pd.DataFrame,
    symbol: str,
    timeframe: str,
    strategy_name: str,
    params: dict[str, Any],
    base_backtest_config: BacktestConfig,
):
    strategy = _build_strategy(strategy_name, params)
    config = _build_backtest_config(
        base=base_backtest_config,
        symbol=symbol,
        timeframe=timeframe,
        strategy_name=strategy_name,
        params=params,
    )
    return BacktestEngine().run(candles=candles, strategy=strategy, config=config)


def _slice_candles(candles: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if candles.empty:
        return candles.copy()
    df = candles.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if start is not None:
        start_ts = _to_utc_timestamp(start)
        df = df[df["timestamp"] >= start_ts]
    if end is not None:
        end_ts = _to_utc_timestamp(end)
        df = df[df["timestamp"] < end_ts]
    return df.reset_index(drop=True)


def _split_walk_forward_windows(
    *,
    start: str,
    end: str,
    train_days: int,
    test_days: int,
) -> list[dict[str, str]]:
    if train_days <= 0 or test_days <= 0:
        raise ValueError("train_days and test_days must be > 0")
    start_ts = _to_utc_timestamp(start)
    end_ts = _to_utc_timestamp(end)
    windows: list[dict[str, str]] = []
    cursor = start_ts
    idx = 0
    while True:
        train_end = cursor + pd.Timedelta(days=train_days)
        test_end = train_end + pd.Timedelta(days=test_days)
        if test_end > end_ts:
            break
        windows.append(
            {
                "window_index": idx,
                "train_start": str(cursor),
                "train_end": str(train_end),
                "test_start": str(train_end),
                "test_end": str(test_end),
            }
        )
        idx += 1
        cursor = cursor + pd.Timedelta(days=test_days)
    return windows


def _flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in row.items() if k not in {"params", "metrics"}}
    params = row["params"]
    metrics = row["metrics"]
    out["params_json"] = yaml.safe_dump(params, sort_keys=True).strip().replace("\n", "; ")
    out["metrics_json"] = yaml.safe_dump(metrics, sort_keys=True).strip().replace("\n", "; ")
    for k, v in params.items():
        out[f"param_{k}"] = v
    for k, v in metrics.items():
        out[k] = v
    return out


class Optimizer:
    def __init__(self, storage: SQLiteStorage | None = None) -> None:
        self.storage = storage

    def _evaluate_one(
        self,
        *,
        optimize_run_id: str,
        symbol: str,
        timeframe: str,
        candles: pd.DataFrame,
        strategy_name: str,
        params: dict[str, Any],
        base_backtest_config: BacktestConfig,
        metric: str,
        parsed_constraints: list[tuple[str, str, float]],
        score_expr: str | None,
        window_role: str,
        window_index: int | None,
        window_start: str | None,
        window_end: str | None,
    ) -> dict[str, Any]:
        created_at = datetime.now(timezone.utc).isoformat()
        strategy = _build_strategy(strategy_name, params)
        config = _build_backtest_config(
            base=base_backtest_config,
            symbol=symbol,
            timeframe=timeframe,
            strategy_name=strategy_name,
            params=params,
        )
        result = BacktestEngine().run(candles=candles, strategy=strategy, config=config)
        metrics = result.summary
        metric_value = float(metrics.get(metric, float("-inf")))
        passed = constraints_pass(metrics, parsed_constraints)
        if score_expr:
            score_value = eval_score_expression(score_expr, metrics)
        else:
            score_value = metric_value
        objective = score_value if passed else float("-inf")
        return {
            "optimize_run_id": optimize_run_id,
            "candidate_run_id": result.run_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "primary_metric": metric,
            "window_role": window_role,
            "window_index": window_index,
            "window_start": window_start,
            "window_end": window_end,
            "params": params,
            "metrics": metrics,
            "metric_value": metric_value,
            "score": score_value,
            "objective": objective,
            "passed_constraints": passed,
            "created_at": created_at,
        }

    def _run_tasks(
        self,
        *,
        tasks: list[dict[str, Any]],
        jobs: int,
    ) -> list[dict[str, Any]]:
        if jobs <= 1:
            return [self._evaluate_one(**task) for task in tasks]

        with ThreadPoolExecutor(max_workers=jobs) as executor:
            return list(executor.map(lambda t: self._evaluate_one(**t), tasks))

    def _rank_dataframe(self, df: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        if df.empty:
            return df, df
        ranked = df.copy()
        ranked["rank"] = pd.NA
        valid_sorted = ranked[ranked["passed_constraints"] == True].sort_values(  # noqa: E712
            by=["objective"], ascending=False
        )
        valid_indices = list(valid_sorted.index)
        for i, idx in enumerate(valid_indices, start=1):
            ranked.at[idx, "rank"] = i
        top = ranked.loc[valid_indices].head(top_n).reset_index(drop=True)
        return ranked, top

    def run(
        self,
        *,
        strategy_name: str,
        symbols: list[str],
        timeframe: str,
        candles_by_symbol: dict[str, pd.DataFrame],
        parameter_sets: list[dict[str, Any]],
        metric: str,
        top_n: int,
        base_backtest_config: BacktestConfig,
        search_mode: str,
        jobs: int = 1,
        constraints: str | None = None,
        score_expr: str | None = None,
        walk_forward: bool = False,
        start: str | None = None,
        end: str | None = None,
        train_days: int = 180,
        test_days: int = 60,
        top_per_train: int = 10,
    ) -> OptimizeRunOutput:
        if strategy_name != "ema_cross":
            raise ValueError("Only ema_cross strategy is currently supported")
        optimize_run_id = uuid4().hex
        parsed_constraints = parse_constraints(constraints)

        created_storage = False
        storage = self.storage
        if storage is None:
            storage = SQLiteStorage(base_backtest_config.db_path)
            created_storage = True
        storage.start_optimize_run(
            optimize_run_id=optimize_run_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            strategy=strategy_name,
            symbols=symbols,
            timeframe=timeframe,
            start_ts=start,
            end_ts=end,
            search_mode=search_mode,
            metric=metric,
            constraints=constraints,
            score_expr=score_expr,
            top_n=top_n,
            walk_forward=walk_forward,
            train_days=train_days if walk_forward else None,
            test_days=test_days if walk_forward else None,
            top_per_train=top_per_train if walk_forward else None,
            config={"parameter_set_count": len(parameter_sets), "jobs": jobs},
        )

        rows: list[dict[str, Any]] = []
        try:
            if walk_forward:
                if start is None or end is None:
                    raise ValueError("start/end are required when walk_forward is enabled")
                windows = _split_walk_forward_windows(start=start, end=end, train_days=train_days, test_days=test_days)
                for symbol in symbols:
                    candles = _slice_candles(candles_by_symbol[symbol], start, end)
                    for window in windows:
                        train_candles = _slice_candles(candles, window["train_start"], window["train_end"])
                        test_candles = _slice_candles(candles, window["test_start"], window["test_end"])
                        if train_candles.empty or test_candles.empty:
                            continue
                        train_tasks = [
                            {
                                "optimize_run_id": optimize_run_id,
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "candles": train_candles,
                                "strategy_name": strategy_name,
                                "params": params,
                                "base_backtest_config": base_backtest_config,
                                "metric": metric,
                                "parsed_constraints": parsed_constraints,
                                "score_expr": score_expr,
                                "window_role": "train",
                                "window_index": window["window_index"],
                                "window_start": window["train_start"],
                                "window_end": window["train_end"],
                            }
                            for params in parameter_sets
                        ]
                        train_rows = self._run_tasks(tasks=train_tasks, jobs=jobs)
                        rows.extend(train_rows)
                        train_df = pd.DataFrame(train_rows)
                        if train_df.empty:
                            continue
                        selected = (
                            train_df[train_df["passed_constraints"] == True]  # noqa: E712
                            .sort_values("objective", ascending=False)
                            .head(top_per_train)
                        )
                        storage.save_wfo_window(
                            optimize_run_id=optimize_run_id,
                            window_index=int(window["window_index"]),
                            symbol=symbol,
                            train_start=window["train_start"],
                            train_end=window["train_end"],
                            test_start=window["test_start"],
                            test_end=window["test_end"],
                            top_per_train=top_per_train,
                            selected_count=int(len(selected)),
                            created_at=datetime.now(timezone.utc).isoformat(),
                        )
                        if selected.empty:
                            continue
                        test_tasks = [
                            {
                                "optimize_run_id": optimize_run_id,
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "candles": test_candles,
                                "strategy_name": strategy_name,
                                "params": row["params"],
                                "base_backtest_config": base_backtest_config,
                                "metric": metric,
                                "parsed_constraints": parsed_constraints,
                                "score_expr": score_expr,
                                "window_role": "test",
                                "window_index": window["window_index"],
                                "window_start": window["test_start"],
                                "window_end": window["test_end"],
                            }
                            for _, row in selected.iterrows()
                        ]
                        rows.extend(self._run_tasks(tasks=test_tasks, jobs=jobs))
            else:
                tasks: list[dict[str, Any]] = []
                for symbol in symbols:
                    candles = _slice_candles(candles_by_symbol[symbol], start, end)
                    if candles.empty:
                        continue
                    for params in parameter_sets:
                        tasks.append(
                            {
                                "optimize_run_id": optimize_run_id,
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "candles": candles,
                                "strategy_name": strategy_name,
                                "params": params,
                                "base_backtest_config": base_backtest_config,
                                "metric": metric,
                                "parsed_constraints": parsed_constraints,
                                "score_expr": score_expr,
                                "window_role": "normal",
                                "window_index": None,
                                "window_start": start,
                                "window_end": end,
                            }
                        )
                rows = self._run_tasks(tasks=tasks, jobs=jobs)

            for row in rows:
                storage.save_optimize_result(row)

            flat_rows = [_flatten_row(r) for r in rows]
            df = pd.DataFrame(flat_rows)
            ranked, top = self._rank_dataframe(df, top_n=top_n)
            train_top = ranked[ranked["window_role"] == "train"].sort_values("objective", ascending=False).head(top_n)
            test_top = ranked[ranked["window_role"] == "test"].sort_values("objective", ascending=False).head(top_n)
            return OptimizeRunOutput(
                optimize_run_id=optimize_run_id,
                results=ranked,
                top_results=top,
                train_top_results=train_top,
                test_top_results=test_top,
            )
        finally:
            if created_storage:
                storage.close()


def export_results(df: pd.DataFrame, export_path: str | Path) -> None:
    path = Path(export_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
        return
    raise ValueError("export path must end with .csv or .parquet")


def load_result_file(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    raise ValueError("Result file must be .csv or .parquet")

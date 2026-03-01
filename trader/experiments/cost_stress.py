"""
Cost Stress Test Experiment

Tests strategy robustness under varying execution costs:
- Fee multipliers (0.5x, 1x, 2x, 3x)
- Slippage (fixed BPS + ATR-based)
- Latency (execution delay in bars)
- Order type effects (market vs limit)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from trader.experiments.core import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentType,
    ScenarioResult,
)
from trader.strategy import STRATEGY_FACTORIES
from trader.strategy.base import Bar, Signal, Strategy

logger = logging.getLogger(__name__)


@dataclass
class CostScenario:
    """Single cost scenario configuration"""
    fee_multiplier: float
    slippage_bps: float
    slippage_type: str  # "fixed" or "atr"
    latency_bars: int
    order_type: str  # "market" or "limit"


class CostStressExperiment:
    """
    Cost Stress Test Experiment

    Measures strategy degradation under realistic execution costs.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        df: pd.DataFrame,
        base_fee_bps: float = 4.0,
        base_slippage_bps: float = 3.0,
    ):
        self.config = config
        self.df = df
        self.base_fee_bps = base_fee_bps
        self.base_slippage_bps = base_slippage_bps

    def generate_scenarios(self) -> list[CostScenario]:
        """Generate cost stress scenarios"""
        type_spec = self.config.type_specific

        fee_multipliers = type_spec.get("fee_multipliers", [1.0, 2.0, 3.0])
        slippage_modes = type_spec.get("slippage_modes", ["fixed"])
        latency_bars = type_spec.get("latency_bars", [0, 1, 2])

        scenarios = []

        # Fixed slippage scenarios
        if "fixed" in slippage_modes or "both" in slippage_modes:
            for fee_mult, latency in product(fee_multipliers, latency_bars):
                for slip_bps in [1, 3, 5, 10]:
                    scenarios.append(
                        CostScenario(
                            fee_multiplier=fee_mult,
                            slippage_bps=slip_bps,
                            slippage_type="fixed",
                            latency_bars=latency,
                            order_type="market",
                        )
                    )

        # ATR-based slippage scenarios
        if "atr" in slippage_modes or "both" in slippage_modes:
            for fee_mult, latency in product(fee_multipliers, latency_bars):
                for atr_mult in [0.1, 0.2, 0.3]:
                    scenarios.append(
                        CostScenario(
                            fee_multiplier=fee_mult,
                            slippage_bps=atr_mult * 10000,  # Store as equivalent BPS
                            slippage_type="atr",
                            latency_bars=latency,
                            order_type="market",
                        )
                    )

        return scenarios

    def run(self) -> ExperimentResult:
        """Execute cost stress test"""
        logger.info(f"Starting Cost Stress Test: {self.config.experiment_id}")

        scenarios = self.generate_scenarios()
        logger.info(f"Generated {len(scenarios)} cost scenarios")

        results = []
        baseline_sharpe = None

        for idx, scenario in enumerate(scenarios):
            logger.info(
                f"Running scenario {idx+1}/{len(scenarios)}: "
                f"fee={scenario.fee_multiplier}x, "
                f"slip={scenario.slippage_bps:.1f}bps ({scenario.slippage_type}), "
                f"latency={scenario.latency_bars}bars"
            )

            result = self._run_scenario(scenario)
            results.append(result)

            # Track baseline (1x fees, min slippage, 0 latency)
            if (
                scenario.fee_multiplier == 1.0
                and scenario.slippage_bps <= 3
                and scenario.latency_bars == 0
            ):
                if baseline_sharpe is None or result.sharpe_ratio > baseline_sharpe:
                    baseline_sharpe = result.sharpe_ratio

        # Calculate robustness score
        robustness_score = self._calculate_robustness(results, baseline_sharpe or 0.0)

        # Summary statistics
        sharpe_ratios = [r.sharpe_ratio for r in results]
        pnls = [r.net_pnl for r in results]

        summary = {
            "baseline_sharpe": baseline_sharpe or 0.0,
            "worst_sharpe": min(sharpe_ratios) if sharpe_ratios else 0.0,
            "mean_sharpe": np.mean(sharpe_ratios) if sharpe_ratios else 0.0,
            "median_pnl": np.median(pnls) if pnls else 0.0,
            "positive_pnl_ratio": sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0.0,
            "scenario_count": len(results),
        }

        verdict = ExperimentResult.calculate_verdict(robustness_score)

        return ExperimentResult(
            config=self.config,
            summary=summary,
            scenarios=results,
            robustness_score=robustness_score,
            verdict=verdict,
        )

    def _run_scenario(self, scenario: CostScenario) -> ScenarioResult:
        """Run backtest for a single cost scenario"""
        # Create strategy
        strategy = self._create_strategy()

        # Prepare data
        df = self.df.copy()

        # Calculate ATR if needed
        if scenario.slippage_type == "atr":
            df["atr"] = self._calculate_atr(df)

        # Run simulation
        equity = self.config.initial_equity
        peak_equity = equity
        max_drawdown = 0.0

        position_side = "flat"
        position_qty = 0.0
        position_entry_price = 0.0

        trades = []
        pending_signal = None
        pending_bar_idx = None

        for i in range(len(df)):
            row = df.iloc[i]
            bar = Bar(
                timestamp=row.name if hasattr(row.name, "isoformat") else str(row.name),
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
            )

            # Get signal from strategy
            position_obj = type("Position", (), {"side": position_side, "entry_price": position_entry_price})()
            signal = strategy.on_bar(bar, position_obj)

            # Store signal for delayed execution
            if signal in ["long", "short", "exit"]:
                pending_signal = signal
                pending_bar_idx = i

            # Execute pending signal after latency
            if (
                pending_signal is not None
                and pending_bar_idx is not None
                and i >= pending_bar_idx + scenario.latency_bars
            ):
                exec_price = row["close"]

                # Apply slippage
                if scenario.slippage_type == "atr":
                    atr_val = row.get("atr", row["close"] * 0.02)
                    slippage_mult = scenario.slippage_bps / 10000
                    slippage_amt = atr_val * slippage_mult
                else:
                    slippage_amt = exec_price * (scenario.slippage_bps / 10000)

                # Apply fees
                fee_bps = self.base_fee_bps * scenario.fee_multiplier
                fee_pct = fee_bps / 10000

                # Execute signal
                if pending_signal == "long" and position_side != "long":
                    # Close short if any
                    if position_side == "short":
                        exit_price = exec_price + slippage_amt
                        pnl = position_qty * (position_entry_price - exit_price)
                        fee = position_qty * exit_price * fee_pct
                        equity += pnl - fee
                        trades.append(pnl - fee)

                    # Open long
                    entry_price = exec_price + slippage_amt
                    position_qty = (equity * 0.95) / entry_price
                    fee = position_qty * entry_price * fee_pct
                    equity -= fee
                    position_side = "long"
                    position_entry_price = entry_price

                elif pending_signal == "short" and position_side != "short":
                    # Close long if any
                    if position_side == "long":
                        exit_price = exec_price - slippage_amt
                        pnl = position_qty * (exit_price - position_entry_price)
                        fee = position_qty * exit_price * fee_pct
                        equity += pnl - fee
                        trades.append(pnl - fee)

                    # Open short
                    entry_price = exec_price - slippage_amt
                    position_qty = (equity * 0.95) / entry_price
                    fee = position_qty * entry_price * fee_pct
                    equity -= fee
                    position_side = "short"
                    position_entry_price = entry_price

                elif pending_signal == "exit" and position_side != "flat":
                    if position_side == "long":
                        exit_price = exec_price - slippage_amt
                        pnl = position_qty * (exit_price - position_entry_price)
                    else:
                        exit_price = exec_price + slippage_amt
                        pnl = position_qty * (position_entry_price - exit_price)

                    fee = position_qty * exit_price * fee_pct
                    equity += pnl - fee
                    trades.append(pnl - fee)
                    position_side = "flat"
                    position_qty = 0.0

                pending_signal = None
                pending_bar_idx = None

            # Track drawdown
            peak_equity = max(peak_equity, equity)
            drawdown = (equity - peak_equity) / peak_equity * 100
            max_drawdown = min(max_drawdown, drawdown)

        # Close final position
        if position_side != "flat":
            final_price = df.iloc[-1]["close"]
            if position_side == "long":
                pnl = position_qty * (final_price - position_entry_price)
            else:
                pnl = position_qty * (position_entry_price - final_price)
            equity += pnl
            trades.append(pnl)

        # Calculate metrics
        net_pnl = equity - self.config.initial_equity
        total_return = net_pnl / self.config.initial_equity

        # CAGR
        days = (pd.to_datetime(df.index[-1]) - pd.to_datetime(df.index[0])).days
        years = max(days / 365.25, 0.01)
        cagr = (1 + total_return) ** (1 / years) - 1

        # Sharpe (simplified)
        if trades:
            trade_returns = np.array(trades) / self.config.initial_equity
            sharpe_ratio = (
                np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(252)
                if np.std(trade_returns) > 0
                else 0.0
            )
        else:
            sharpe_ratio = 0.0

        # Profit factor
        wins = [t for t in trades if t > 0]
        losses = [abs(t) for t in trades if t < 0]
        profit_factor = sum(wins) / sum(losses) if losses else (100.0 if wins else 0.0)

        # Win rate
        win_rate = len(wins) / len(trades) * 100 if trades else 0.0

        scenario_id = (
            f"fee{scenario.fee_multiplier}x_"
            f"slip{scenario.slippage_bps:.1f}{scenario.slippage_type}_"
            f"lat{scenario.latency_bars}"
        )

        return ScenarioResult(
            scenario_id=scenario_id,
            scenario_params={
                "fee_multiplier": scenario.fee_multiplier,
                "slippage_bps": scenario.slippage_bps,
                "slippage_type": scenario.slippage_type,
                "latency_bars": scenario.latency_bars,
            },
            net_pnl=net_pnl,
            cagr=cagr * 100,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            win_rate=win_rate,
            trade_count=len(trades),
        )

    def _create_strategy(self) -> Strategy:
        """Create strategy instance"""
        # Handle original strategies
        if self.config.strategy_name == "ema_cross":
            from trader.strategy.ema_cross import EMACrossStrategy
            return EMACrossStrategy(**self.config.strategy_params)
        elif self.config.strategy_name == "rsi":
            from trader.strategy.rsi import RSIStrategy
            return RSIStrategy(**self.config.strategy_params)
        elif self.config.strategy_name == "macd":
            from trader.strategy.macd import MACDStrategy
            return MACDStrategy(**self.config.strategy_params)
        elif self.config.strategy_name == "bollinger":
            from trader.strategy.bollinger import BollingerBandStrategy
            return BollingerBandStrategy(**self.config.strategy_params)

        # Handle strategy families
        for family_name, factory in STRATEGY_FACTORIES.items():
            if self.config.strategy_name.startswith(family_name):
                strategy_type = self.config.strategy_name
                return factory(strategy_type, self.config.strategy_params)

        raise ValueError(f"Unknown strategy: {self.config.strategy_name}")

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr.fillna(close * 0.02)

    def _calculate_robustness(
        self, results: list[ScenarioResult], baseline_sharpe: float
    ) -> float:
        """
        Calculate robustness score based on performance degradation

        Score = 1 - avg(degradation across scenarios)
        """
        if baseline_sharpe <= 0:
            return 0.0

        degradations = []
        for result in results:
            if result.sharpe_ratio < baseline_sharpe:
                degradation = (baseline_sharpe - result.sharpe_ratio) / abs(baseline_sharpe)
                degradations.append(min(degradation, 1.0))
            else:
                degradations.append(0.0)

        avg_degradation = np.mean(degradations) if degradations else 1.0
        robustness = max(1.0 - avg_degradation, 0.0)

        return robustness

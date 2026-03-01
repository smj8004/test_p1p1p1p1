"""
Regime Gating Experiment

Tests strategy performance across market regimes:
- Trend regimes: UPTREND, DOWNTREND, SIDEWAYS
- Volatility regimes: HIGH_VOL, LOW_VOL
- Gating modes: on_off (trade filtering), sizing (position scaling)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
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
from trader.strategy.base import Bar, Strategy

logger = logging.getLogger(__name__)


class TrendRegime(str, Enum):
    """Trend regime classification"""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"


class VolRegime(str, Enum):
    """Volatility regime classification"""
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"


@dataclass
class RegimeLabel:
    """Combined regime label"""
    trend: TrendRegime
    volatility: VolRegime

    def __str__(self) -> str:
        return f"{self.trend.value}_{self.volatility.value}"


class RegimeDetector:
    """Detects market regimes using ADX and ATR"""

    def __init__(
        self,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        vol_lookback: int = 20,
        vol_threshold: float = 1.5,
    ):
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.vol_lookback = vol_lookback
        self.vol_threshold = vol_threshold

    def detect_regimes(self, df: pd.DataFrame) -> pd.Series:
        """Detect regime for each bar"""
        # Calculate ADX
        adx, plus_di, minus_di = self._calculate_adx(df)

        # Calculate volatility ratio
        vol_ratio = self._calculate_volatility_ratio(df)

        # Calculate trend direction
        trend_slope = self._calculate_trend_slope(df)

        # Classify regimes
        regimes = []
        for i in range(len(df)):
            # Trend regime
            if adx.iloc[i] > self.adx_threshold:
                if trend_slope.iloc[i] > 0.5:
                    trend = TrendRegime.UPTREND
                elif trend_slope.iloc[i] < -0.5:
                    trend = TrendRegime.DOWNTREND
                else:
                    trend = TrendRegime.SIDEWAYS
            else:
                trend = TrendRegime.SIDEWAYS

            # Volatility regime
            if vol_ratio.iloc[i] > self.vol_threshold:
                volatility = VolRegime.HIGH_VOL
            else:
                volatility = VolRegime.LOW_VOL

            regimes.append(RegimeLabel(trend, volatility))

        return pd.Series(regimes, index=df.index)

    def _calculate_adx(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate ADX indicator"""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smoothed averages
        atr = tr.rolling(self.adx_period).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(self.adx_period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(self.adx_period).mean() / atr

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(self.adx_period).mean()

        return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)

    def _calculate_volatility_ratio(self, df: pd.DataFrame) -> pd.Series:
        """Calculate current vol / average vol ratio"""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(14).mean()
        avg_atr = atr.rolling(self.vol_lookback * 5).mean()

        return (atr / (avg_atr + 1e-10)).fillna(1.0)

    def _calculate_trend_slope(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend slope using SMA"""
        close = df["close"]
        sma = close.rolling(50).mean()

        # Slope over 10 periods
        sma_slope = (sma - sma.shift(10)) / sma.shift(10) * 100

        return sma_slope.fillna(0)


class RegimeGateExperiment:
    """
    Regime Gating Experiment

    Tests strategy performance with regime-based filtering or sizing.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        df: pd.DataFrame,
    ):
        self.config = config
        self.df = df
        self.detector = RegimeDetector()

    def run(self) -> ExperimentResult:
        """Execute regime gating experiment"""
        logger.info(f"Starting Regime Gating: {self.config.experiment_id}")

        type_spec = self.config.type_specific
        regime_mode = type_spec.get("regime_mode", "both")  # "trend", "vol", or "both"
        gating_mode = type_spec.get("gating_mode", "on_off")  # "on_off" or "sizing"

        # Detect regimes
        logger.info("Detecting market regimes...")
        regimes = self.detector.detect_regimes(self.df)

        # Run baseline (no gating)
        logger.info("Running baseline (no gating)...")
        baseline = self._run_baseline(regimes)

        # Run regime-specific scenarios
        results = [baseline]

        if regime_mode in ["trend", "both"]:
            # Test trend-based gating
            for trend in TrendRegime:
                logger.info(f"Running scenario: {trend.value} only")
                result = self._run_regime_scenario(
                    regimes, trend_filter=trend, vol_filter=None, gating_mode=gating_mode
                )
                results.append(result)

        if regime_mode in ["vol", "both"]:
            # Test volatility-based gating
            for vol in VolRegime:
                logger.info(f"Running scenario: {vol.value} only")
                result = self._run_regime_scenario(
                    regimes, trend_filter=None, vol_filter=vol, gating_mode=gating_mode
                )
                results.append(result)

        if regime_mode == "both":
            # Test combined regimes
            for trend in TrendRegime:
                for vol in VolRegime:
                    logger.info(f"Running scenario: {trend.value} + {vol.value}")
                    result = self._run_regime_scenario(
                        regimes, trend_filter=trend, vol_filter=vol, gating_mode=gating_mode
                    )
                    results.append(result)

        # Calculate robustness score
        robustness_score = self._calculate_robustness(results, baseline)

        # Summary statistics
        summary = {
            "baseline_sharpe": baseline.sharpe_ratio,
            "baseline_pnl": baseline.net_pnl,
            "best_regime_sharpe": max(r.sharpe_ratio for r in results[1:]) if len(results) > 1 else 0.0,
            "regime_count": len(results),
            "gating_mode": gating_mode,
        }

        verdict = ExperimentResult.calculate_verdict(robustness_score)

        return ExperimentResult(
            config=self.config,
            summary=summary,
            scenarios=results,
            robustness_score=robustness_score,
            verdict=verdict,
        )

    def _run_baseline(self, regimes: pd.Series) -> ScenarioResult:
        """Run baseline without regime gating"""
        result = self._run_backtest(regimes, trend_filter=None, vol_filter=None, sizing_factor=1.0)

        return ScenarioResult(
            scenario_id="baseline",
            scenario_params={"regime_filter": "none", "sizing_factor": 1.0},
            net_pnl=result["net_pnl"],
            cagr=result["cagr"],
            max_drawdown=result["max_drawdown"],
            sharpe_ratio=result["sharpe_ratio"],
            profit_factor=result["profit_factor"],
            win_rate=result["win_rate"],
            trade_count=result["trade_count"],
        )

    def _run_regime_scenario(
        self,
        regimes: pd.Series,
        trend_filter: TrendRegime | None,
        vol_filter: VolRegime | None,
        gating_mode: str,
    ) -> ScenarioResult:
        """Run backtest with regime filtering or sizing"""
        if gating_mode == "on_off":
            sizing_factor = 1.0
        else:
            # Adaptive sizing based on regime favorability
            # This is a simple heuristic - could be optimized
            sizing_factor = 1.0

        result = self._run_backtest(
            regimes, trend_filter=trend_filter, vol_filter=vol_filter, sizing_factor=sizing_factor
        )

        scenario_id = "regime_"
        if trend_filter:
            scenario_id += trend_filter.value
        if vol_filter:
            scenario_id += f"_{vol_filter.value}" if trend_filter else vol_filter.value

        return ScenarioResult(
            scenario_id=scenario_id,
            scenario_params={
                "trend_filter": trend_filter.value if trend_filter else "all",
                "vol_filter": vol_filter.value if vol_filter else "all",
                "sizing_factor": sizing_factor,
            },
            net_pnl=result["net_pnl"],
            cagr=result["cagr"],
            max_drawdown=result["max_drawdown"],
            sharpe_ratio=result["sharpe_ratio"],
            profit_factor=result["profit_factor"],
            win_rate=result["win_rate"],
            trade_count=result["trade_count"],
        )

    def _run_backtest(
        self,
        regimes: pd.Series,
        trend_filter: TrendRegime | None,
        vol_filter: VolRegime | None,
        sizing_factor: float,
    ) -> dict[str, float]:
        """Run backtest with regime filtering"""
        # Create strategy
        strategy = self._create_strategy()

        # Run simulation
        equity = self.config.initial_equity
        peak_equity = equity
        max_drawdown = 0.0

        position_side = "flat"
        position_qty = 0.0
        position_entry_price = 0.0

        trades = []

        for i in range(len(self.df)):
            row = self.df.iloc[i]
            regime = regimes.iloc[i]

            # Check if we should trade in this regime
            regime_allowed = True
            if trend_filter and regime.trend != trend_filter:
                regime_allowed = False
            if vol_filter and regime.volatility != vol_filter:
                regime_allowed = False

            bar = Bar(
                timestamp=row.name if hasattr(row.name, "isoformat") else str(row.name),
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
            )

            # Get signal
            position_obj = type("Position", (), {"side": position_side, "entry_price": position_entry_price})()
            signal = strategy.on_bar(bar, position_obj)

            # Execute signal (only if regime allowed)
            if not regime_allowed:
                # Close position if we're in unfavorable regime
                if position_side != "flat":
                    signal = "exit"
                else:
                    signal = None

            if signal:
                exec_price = row["close"]
                fee_pct = 0.0004

                if signal == "long" and position_side != "long":
                    # Close short if any
                    if position_side == "short":
                        pnl = position_qty * (position_entry_price - exec_price)
                        fee = position_qty * exec_price * fee_pct
                        equity += pnl - fee
                        trades.append(pnl - fee)

                    # Open long
                    position_qty = (equity * 0.95 * sizing_factor) / exec_price
                    fee = position_qty * exec_price * fee_pct
                    equity -= fee
                    position_side = "long"
                    position_entry_price = exec_price

                elif signal == "short" and position_side != "short":
                    # Close long if any
                    if position_side == "long":
                        pnl = position_qty * (exec_price - position_entry_price)
                        fee = position_qty * exec_price * fee_pct
                        equity += pnl - fee
                        trades.append(pnl - fee)

                    # Open short
                    position_qty = (equity * 0.95 * sizing_factor) / exec_price
                    fee = position_qty * exec_price * fee_pct
                    equity -= fee
                    position_side = "short"
                    position_entry_price = exec_price

                elif signal == "exit" and position_side != "flat":
                    if position_side == "long":
                        pnl = position_qty * (exec_price - position_entry_price)
                    else:
                        pnl = position_qty * (position_entry_price - exec_price)

                    fee = position_qty * exec_price * fee_pct
                    equity += pnl - fee
                    trades.append(pnl - fee)
                    position_side = "flat"
                    position_qty = 0.0

            # Track drawdown
            peak_equity = max(peak_equity, equity)
            drawdown = (equity - peak_equity) / peak_equity * 100
            max_drawdown = min(max_drawdown, drawdown)

        # Close final position
        if position_side != "flat":
            final_price = self.df.iloc[-1]["close"]
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
        days = (pd.to_datetime(self.df.index[-1]) - pd.to_datetime(self.df.index[0])).days
        years = max(days / 365.25, 0.01)
        cagr = (1 + total_return) ** (1 / years) - 1

        # Sharpe
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

        return {
            "net_pnl": net_pnl,
            "cagr": cagr * 100,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "profit_factor": profit_factor,
            "win_rate": win_rate,
            "trade_count": len(trades),
        }

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
                return factory(self.config.strategy_name, self.config.strategy_params)

        raise ValueError(f"Unknown strategy: {self.config.strategy_name}")

    def _calculate_robustness(
        self, results: list[ScenarioResult], baseline: ScenarioResult
    ) -> float:
        """
        Calculate robustness score

        If any regime significantly outperforms baseline, there's potential edge.
        Score based on: best regime performance vs baseline + consistency across regimes.
        """
        if not results or baseline.sharpe_ratio <= 0:
            return 0.0

        # Best regime performance vs baseline
        regime_results = results[1:]  # Exclude baseline
        if not regime_results:
            return 0.0

        sharpe_ratios = [r.sharpe_ratio for r in regime_results]
        best_sharpe = max(sharpe_ratios)
        baseline_sharpe = baseline.sharpe_ratio

        # Improvement ratio
        if baseline_sharpe > 0:
            improvement = (best_sharpe - baseline_sharpe) / abs(baseline_sharpe)
        else:
            improvement = 0.0

        # Consistency: how many regimes are profitable
        positive_regimes = sum(1 for r in regime_results if r.net_pnl > 0)
        consistency = positive_regimes / len(regime_results)

        # Combined score
        robustness = 0.6 * min(improvement, 1.0) + 0.4 * consistency

        return max(robustness, 0.0)

"""
Experiment Report Generator

Generates comprehensive reports for experiments:
- JSON summary
- CSV scenario details
- Markdown summary
- Visualization plots
- Verdict text file
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from trader.experiments.core import ExperimentResult, ExperimentType

logger = logging.getLogger(__name__)
console = Console()


class ExperimentReporter:
    """Generate comprehensive experiment reports"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self, result: ExperimentResult) -> None:
        """Generate all report outputs"""
        logger.info(f"Generating report in {self.output_dir}")

        # JSON summary
        self._save_json(result)

        # CSV scenarios
        self._save_csv(result)

        # Markdown summary
        self._save_markdown(result)

        # Verdict
        self._save_verdict(result)

        # Visualizations
        if result.config.experiment_type == ExperimentType.COST_STRESS:
            self._plot_cost_degradation(result)
        elif result.config.experiment_type == ExperimentType.WALK_FORWARD:
            self._plot_wfo_performance(result)
        elif result.config.experiment_type == ExperimentType.REGIME_GATE:
            self._plot_regime_breakdown(result)

        # Console output
        self._print_summary(result)

        logger.info(f"Report generated successfully at {self.output_dir}")

    def _save_json(self, result: ExperimentResult) -> None:
        """Save full result as JSON"""
        output_file = self.output_dir / "report.json"

        with open(output_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"Saved JSON report: {output_file}")

    def _save_csv(self, result: ExperimentResult) -> None:
        """Save scenario details as CSV"""
        output_file = self.output_dir / "scenarios.csv"

        rows = []
        for scenario in result.scenarios:
            row = {
                "scenario_id": scenario.scenario_id,
                "net_pnl": scenario.net_pnl,
                "cagr": scenario.cagr,
                "max_drawdown": scenario.max_drawdown,
                "sharpe_ratio": scenario.sharpe_ratio,
                "profit_factor": scenario.profit_factor,
                "win_rate": scenario.win_rate,
                "trade_count": scenario.trade_count,
            }
            # Add scenario params as separate columns
            for key, val in scenario.scenario_params.items():
                row[f"param_{key}"] = val

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)

        logger.info(f"Saved CSV scenarios: {output_file}")

    def _save_markdown(self, result: ExperimentResult) -> None:
        """Save markdown summary"""
        output_file = self.output_dir / "summary.md"

        md = []
        md.append(f"# Experiment Report: {result.config.experiment_type.value}")
        md.append("")
        md.append(f"**Experiment ID:** {result.config.experiment_id}")
        md.append(f"**Strategy:** {result.config.strategy_name}")
        md.append(f"**Symbol:** {result.config.symbol}")
        md.append(f"**Timeframe:** {result.config.timeframe}")
        md.append(f"**Period:** {result.config.start_date} to {result.config.end_date}")
        md.append("")

        # Verdict
        md.append("## Verdict")
        md.append("")
        md.append(f"**{result.verdict}**")
        md.append("")
        md.append(f"- Robustness Score: **{result.robustness_score:.3f}**")
        md.append("")

        # Interpretation
        if result.robustness_score >= 0.7:
            md.append("✅ Strategy shows strong robustness across scenarios. Likely has real edge.")
        elif result.robustness_score >= 0.4:
            md.append("⚠️ Strategy shows moderate robustness. Further validation recommended.")
        else:
            md.append("❌ Strategy shows weak robustness. Likely overfit or no real edge.")
        md.append("")

        # Summary metrics
        md.append("## Summary Metrics")
        md.append("")
        md.append("| Metric | Value |")
        md.append("|--------|-------|")
        for key, val in result.summary.items():
            if isinstance(val, float):
                md.append(f"| {key} | {val:.4f} |")
            else:
                md.append(f"| {key} | {val} |")
        md.append("")

        # Top scenarios
        md.append("## Top 5 Scenarios")
        md.append("")
        sorted_scenarios = sorted(
            result.scenarios, key=lambda x: x.sharpe_ratio, reverse=True
        )[:5]

        md.append("| Rank | Scenario | Sharpe | PnL | CAGR | Max DD | Trades |")
        md.append("|------|----------|--------|-----|------|--------|--------|")
        for i, scenario in enumerate(sorted_scenarios, 1):
            md.append(
                f"| {i} | {scenario.scenario_id} | {scenario.sharpe_ratio:.2f} | "
                f"${scenario.net_pnl:.0f} | {scenario.cagr:.1f}% | "
                f"{scenario.max_drawdown:.1f}% | {scenario.trade_count} |"
            )
        md.append("")

        # Worst scenarios
        md.append("## Bottom 5 Scenarios")
        md.append("")
        worst_scenarios = sorted(result.scenarios, key=lambda x: x.sharpe_ratio)[:5]

        md.append("| Rank | Scenario | Sharpe | PnL | CAGR | Max DD | Trades |")
        md.append("|------|----------|--------|-----|------|--------|--------|")
        for i, scenario in enumerate(worst_scenarios, 1):
            md.append(
                f"| {i} | {scenario.scenario_id} | {scenario.sharpe_ratio:.2f} | "
                f"${scenario.net_pnl:.0f} | {scenario.cagr:.1f}% | "
                f"{scenario.max_drawdown:.1f}% | {scenario.trade_count} |"
            )
        md.append("")

        # Config
        md.append("## Configuration")
        md.append("")
        md.append("```json")
        md.append(json.dumps(result.config.to_dict(), indent=2))
        md.append("```")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(md))

        logger.info(f"Saved Markdown summary: {output_file}")

    def _save_verdict(self, result: ExperimentResult) -> None:
        """Save simple verdict text file"""
        output_file = self.output_dir / "verdict.txt"

        verdict_text = f"{result.verdict}\nRobustness Score: {result.robustness_score:.3f}\n"

        with open(output_file, "w") as f:
            f.write(verdict_text)

        logger.info(f"Saved verdict: {output_file}")

    def _plot_cost_degradation(self, result: ExperimentResult) -> None:
        """Plot performance degradation curve for cost stress test"""
        output_file = self.output_dir / "degradation_curve.png"

        # Extract fee multipliers and sharpe ratios
        data = []
        for scenario in result.scenarios:
            fee_mult = scenario.scenario_params.get("fee_multiplier", 1.0)
            data.append((fee_mult, scenario.sharpe_ratio))

        # Group by fee multiplier
        df = pd.DataFrame(data, columns=["fee_mult", "sharpe"])
        grouped = df.groupby("fee_mult")["sharpe"].agg(["mean", "std", "min", "max"])

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(grouped.index, grouped["mean"], marker="o", label="Mean Sharpe", linewidth=2)
        ax.fill_between(
            grouped.index,
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.3,
            label="±1 Std Dev",
        )
        ax.plot(grouped.index, grouped["min"], linestyle="--", alpha=0.5, label="Min")
        ax.plot(grouped.index, grouped["max"], linestyle="--", alpha=0.5, label="Max")

        ax.set_xlabel("Fee Multiplier", fontsize=12)
        ax.set_ylabel("Sharpe Ratio", fontsize=12)
        ax.set_title("Performance Degradation vs Fee Multiplier", fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.axhline(y=0, color="red", linestyle="-", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()

        logger.info(f"Saved degradation curve: {output_file}")

    def _plot_wfo_performance(self, result: ExperimentResult) -> None:
        """Plot walk-forward OOS performance"""
        output_file = self.output_dir / "oos_performance.png"

        # Extract split performance
        split_ids = []
        sharpe_ratios = []
        pnls = []

        for scenario in result.scenarios:
            split_id = scenario.scenario_params.get("split_id")
            if split_id is not None:
                split_ids.append(split_id)
                sharpe_ratios.append(scenario.sharpe_ratio)
                pnls.append(scenario.net_pnl)

        if not split_ids:
            logger.warning("No split data found for WFO plot")
            return

        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Sharpe ratios
        ax1.bar(split_ids, sharpe_ratios, alpha=0.7, color="steelblue")
        ax1.axhline(y=0, color="red", linestyle="-", alpha=0.3)
        ax1.set_xlabel("Split ID", fontsize=12)
        ax1.set_ylabel("Sharpe Ratio", fontsize=12)
        ax1.set_title("Out-of-Sample Sharpe Ratio by Split", fontsize=14)
        ax1.grid(alpha=0.3)

        # PnL
        colors = ["green" if p > 0 else "red" for p in pnls]
        ax2.bar(split_ids, pnls, alpha=0.7, color=colors)
        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax2.set_xlabel("Split ID", fontsize=12)
        ax2.set_ylabel("Net PnL ($)", fontsize=12)
        ax2.set_title("Out-of-Sample PnL by Split", fontsize=14)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()

        logger.info(f"Saved OOS performance plot: {output_file}")

    def _plot_regime_breakdown(self, result: ExperimentResult) -> None:
        """Plot regime-specific performance breakdown"""
        output_file = self.output_dir / "regime_breakdown.png"

        # Extract regime performance
        regime_labels = []
        sharpe_ratios = []
        pnls = []

        for scenario in result.scenarios:
            if scenario.scenario_id == "baseline":
                continue

            regime_labels.append(scenario.scenario_id.replace("regime_", ""))
            sharpe_ratios.append(scenario.sharpe_ratio)
            pnls.append(scenario.net_pnl)

        if not regime_labels:
            logger.warning("No regime data found for regime plot")
            return

        # Limit to top 10 regimes for readability
        if len(regime_labels) > 10:
            # Sort by Sharpe and take top 10
            sorted_data = sorted(
                zip(regime_labels, sharpe_ratios, pnls),
                key=lambda x: x[1],
                reverse=True,
            )[:10]
            regime_labels, sharpe_ratios, pnls = zip(*sorted_data)

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Sharpe ratios
        y_pos = np.arange(len(regime_labels))
        ax1.barh(y_pos, sharpe_ratios, alpha=0.7, color="steelblue")
        ax1.axvline(x=0, color="red", linestyle="-", alpha=0.3)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(regime_labels, fontsize=9)
        ax1.set_xlabel("Sharpe Ratio", fontsize=12)
        ax1.set_title("Sharpe Ratio by Regime", fontsize=14)
        ax1.grid(alpha=0.3, axis="x")

        # PnL
        colors = ["green" if p > 0 else "red" for p in pnls]
        ax2.barh(y_pos, pnls, alpha=0.7, color=colors)
        ax2.axvline(x=0, color="black", linestyle="-", alpha=0.3)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(regime_labels, fontsize=9)
        ax2.set_xlabel("Net PnL ($)", fontsize=12)
        ax2.set_title("PnL by Regime", fontsize=14)
        ax2.grid(alpha=0.3, axis="x")

        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()

        logger.info(f"Saved regime breakdown plot: {output_file}")

    def _print_summary(self, result: ExperimentResult) -> None:
        """Print summary to console"""
        console.print("\n[bold]Experiment Summary[/bold]")
        console.print(f"Type: {result.config.experiment_type.value}")
        console.print(f"Strategy: {result.config.strategy_name}")
        console.print(f"Period: {result.config.start_date} to {result.config.end_date}")
        console.print("")

        # Verdict
        verdict_color = "green" if result.robustness_score >= 0.7 else "yellow" if result.robustness_score >= 0.4 else "red"
        console.print(f"[bold {verdict_color}]Verdict: {result.verdict}[/bold {verdict_color}]")
        console.print(f"Robustness Score: {result.robustness_score:.3f}")
        console.print("")

        # Summary table
        table = Table(title="Summary Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        for key, val in result.summary.items():
            if isinstance(val, float):
                table.add_row(key, f"{val:.4f}")
            else:
                table.add_row(key, str(val))

        console.print(table)
        console.print("")

        # Scenario stats
        sharpe_ratios = [s.sharpe_ratio for s in result.scenarios]
        pnls = [s.net_pnl for s in result.scenarios]

        console.print(f"Scenarios: {len(result.scenarios)}")
        console.print(f"Sharpe Range: [{min(sharpe_ratios):.2f}, {max(sharpe_ratios):.2f}]")
        console.print(f"PnL Range: [${min(pnls):.0f}, ${max(pnls):.0f}]")
        console.print(f"Positive PnL: {sum(1 for p in pnls if p > 0)}/{len(pnls)}")
        console.print("")

        console.print(f"[dim]Full report saved to: {self.output_dir}[/dim]")

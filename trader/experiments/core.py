"""
Core data structures for experiment framework
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ExperimentType(str, Enum):
    """Type of experiment"""
    COST_STRESS = "cost_stress"
    WALK_FORWARD = "walk_forward"
    REGIME_GATE = "regime_gate"


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run"""
    experiment_type: ExperimentType
    experiment_id: str
    strategy_name: str
    strategy_params: dict[str, Any]
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    seed: int = 42
    initial_equity: float = 10_000.0

    # Type-specific config
    type_specific: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict"""
        return {
            "experiment_type": self.experiment_type.value,
            "experiment_id": self.experiment_id,
            "strategy_name": self.strategy_name,
            "strategy_params": self.strategy_params,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "seed": self.seed,
            "initial_equity": self.initial_equity,
            "type_specific": self.type_specific,
        }


@dataclass
class ScenarioResult:
    """Result from a single scenario within an experiment"""
    scenario_id: str
    scenario_params: dict[str, Any]

    # Core metrics
    net_pnl: float
    cagr: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    win_rate: float
    trade_count: int

    # Additional metrics
    extra_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict"""
        return {
            "scenario_id": self.scenario_id,
            "scenario_params": self.scenario_params,
            "net_pnl": self.net_pnl,
            "cagr": self.cagr,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "profit_factor": self.profit_factor,
            "win_rate": self.win_rate,
            "trade_count": self.trade_count,
            **self.extra_metrics,
        }


@dataclass
class ExperimentResult:
    """Complete result from an experiment"""
    config: ExperimentConfig
    summary: dict[str, float]
    scenarios: list[ScenarioResult]
    robustness_score: float
    verdict: str  # "HAS EDGE", "UNCERTAIN", "NO EDGE"

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict"""
        return {
            "config": self.config.to_dict(),
            "summary": self.summary,
            "scenarios": [s.to_dict() for s in self.scenarios],
            "robustness_score": self.robustness_score,
            "verdict": self.verdict,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
        }

    @staticmethod
    def calculate_verdict(robustness_score: float) -> str:
        """Calculate verdict from robustness score"""
        if robustness_score >= 0.7:
            return "HAS EDGE"
        elif robustness_score >= 0.4:
            return "UNCERTAIN"
        else:
            return "NO EDGE"

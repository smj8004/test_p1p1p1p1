from .base import Bar, Signal, Strategy, StrategyPosition

# Original strategies
from .bollinger import BollingerBandStrategy
from .ema_cross import EMACrossStrategy
from .macd import MACDStrategy
from .rsi import RSIStrategy

# Strategy families
from .trend_family import (
    TrendEMACrossStrategy,
    TrendSuperTrendStrategy,
    TrendDonchianBreakout,
    TrendKeltnerChannel,
    create_trend_strategy,
    TREND_STRATEGIES,
)
from .meanrev_family import (
    MeanRevBollingerStrategy,
    MeanRevZScoreStrategy,
    MeanRevRSIStrategy,
    MeanRevStochRSIStrategy,
    create_meanrev_strategy,
    MEANREV_STRATEGIES,
)
from .breakout_family import (
    BreakoutVolatilityStrategy,
    BreakoutRangeStrategy,
    BreakoutMomentumStrategy,
    BreakoutATRChannelStrategy,
    create_breakout_strategy,
    BREAKOUT_STRATEGIES,
)
from .vol_regime import (
    VolRegime,
    VolRegimeAdaptiveStrategy,
    VolRegimeVIXStrategy,
    VolTargetStrategy,
    VolClusterStrategy,
    create_volregime_strategy,
    VOLREGIME_STRATEGIES,
)
from .carry import (
    CarryFundingRateStrategy,
    CarryPremiumStrategy,
    CarryYieldStrategy,
    CarryMomentumStrategy,
    create_carry_strategy,
    CARRY_STRATEGIES,
)
from .microstructure import (
    MicroVWAPStrategy,
    MicroVolumeProfileStrategy,
    MicroOrderFlowStrategy,
    MicroVolumeMomentumStrategy,
    create_microstructure_strategy,
    MICROSTRUCTURE_STRATEGIES,
)

__all__ = [
    # Base
    "Bar",
    "Signal",
    "Strategy",
    "StrategyPosition",
    # Original strategies
    "EMACrossStrategy",
    "RSIStrategy",
    "MACDStrategy",
    "BollingerBandStrategy",
    # Trend family
    "TrendEMACrossStrategy",
    "TrendSuperTrendStrategy",
    "TrendDonchianBreakout",
    "TrendKeltnerChannel",
    "create_trend_strategy",
    "TREND_STRATEGIES",
    # Mean reversion family
    "MeanRevBollingerStrategy",
    "MeanRevZScoreStrategy",
    "MeanRevRSIStrategy",
    "MeanRevStochRSIStrategy",
    "create_meanrev_strategy",
    "MEANREV_STRATEGIES",
    # Breakout family
    "BreakoutVolatilityStrategy",
    "BreakoutRangeStrategy",
    "BreakoutMomentumStrategy",
    "BreakoutATRChannelStrategy",
    "create_breakout_strategy",
    "BREAKOUT_STRATEGIES",
    # Volatility regime family
    "VolRegime",
    "VolRegimeAdaptiveStrategy",
    "VolRegimeVIXStrategy",
    "VolTargetStrategy",
    "VolClusterStrategy",
    "create_volregime_strategy",
    "VOLREGIME_STRATEGIES",
    # Carry family
    "CarryFundingRateStrategy",
    "CarryPremiumStrategy",
    "CarryYieldStrategy",
    "CarryMomentumStrategy",
    "create_carry_strategy",
    "CARRY_STRATEGIES",
    # Microstructure family
    "MicroVWAPStrategy",
    "MicroVolumeProfileStrategy",
    "MicroOrderFlowStrategy",
    "MicroVolumeMomentumStrategy",
    "create_microstructure_strategy",
    "MICROSTRUCTURE_STRATEGIES",
]

# All strategy registries combined
ALL_STRATEGY_FAMILIES = {
    "trend": TREND_STRATEGIES,
    "meanrev": MEANREV_STRATEGIES,
    "breakout": BREAKOUT_STRATEGIES,
    "volregime": VOLREGIME_STRATEGIES,
    "carry": CARRY_STRATEGIES,
    "microstructure": MICROSTRUCTURE_STRATEGIES,
}

# Factory functions by family
STRATEGY_FACTORIES = {
    "trend": create_trend_strategy,
    "meanrev": create_meanrev_strategy,
    "breakout": create_breakout_strategy,
    "volregime": create_volregime_strategy,
    "carry": create_carry_strategy,
    "microstructure": create_microstructure_strategy,
}

"""Custom exceptions for the trading system."""

from __future__ import annotations


class TradingError(Exception):
    """Base exception for all trading-related errors."""

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class BrokerError(TradingError):
    """Exception for broker/exchange communication errors."""

    def __init__(
        self,
        message: str,
        *,
        broker: str | None = None,
        operation: str | None = None,
        details: dict | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.broker = broker
        self.operation = operation


class StrategyError(TradingError):
    """Exception for strategy execution errors."""

    def __init__(
        self,
        message: str,
        *,
        strategy_name: str | None = None,
        details: dict | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.strategy_name = strategy_name


class ConfigError(TradingError):
    """Exception for configuration/settings errors."""

    def __init__(
        self,
        message: str,
        *,
        config_key: str | None = None,
        details: dict | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.config_key = config_key


class DataError(TradingError):
    """Exception for data fetching/processing errors."""

    def __init__(
        self,
        message: str,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        details: dict | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.symbol = symbol
        self.timeframe = timeframe


class RiskLimitError(TradingError):
    """Exception for risk management limit violations."""

    def __init__(
        self,
        message: str,
        *,
        limit_type: str | None = None,
        limit_value: float | None = None,
        current_value: float | None = None,
        details: dict | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.limit_type = limit_type
        self.limit_value = limit_value
        self.current_value = current_value

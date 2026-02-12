from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Literal

from rich.logging import RichHandler

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_DEFAULT_LOG_DIR = Path("logs")
_DEFAULT_LOG_FILE = "trader.log"
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_CONSOLE_FORMAT = "%(message)s"

_configured = False


def setup_logging(
    level: LogLevel = "INFO",
    *,
    log_dir: Path | str | None = None,
    log_file: str | None = None,
    console_level: LogLevel | None = None,
    file_level: LogLevel | None = None,
    enable_file_logging: bool = True,
) -> None:
    """
    Configure the logging system with console and optional file handlers.

    Args:
        level: Default log level for both console and file (if specific levels not set)
        log_dir: Directory for log files (default: ./logs)
        log_file: Log file name (default: trader.log)
        console_level: Override log level for console output
        file_level: Override log level for file output
        enable_file_logging: Whether to enable file logging (default: True)
    """
    global _configured
    if _configured:
        return
    _configured = True

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    effective_console_level = console_level or level
    console_handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_path=False,
    )
    console_handler.setLevel(getattr(logging, effective_console_level.upper()))
    console_handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt="[%X]"))
    root_logger.addHandler(console_handler)

    if enable_file_logging:
        effective_log_dir = Path(log_dir) if log_dir else _DEFAULT_LOG_DIR
        effective_log_file = log_file or _DEFAULT_LOG_FILE
        effective_file_level = file_level or level

        try:
            effective_log_dir.mkdir(parents=True, exist_ok=True)
            log_path = effective_log_dir / effective_log_file

            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(getattr(logging, effective_file_level.upper()))
            file_handler.setFormatter(
                logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATE_FORMAT)
            )
            root_logger.addHandler(file_handler)
        except (OSError, PermissionError) as exc:
            console_handler.setLevel(logging.WARNING)
            root_logger.warning(f"Failed to create log file: {exc}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance by name.

    Args:
        name: The name for the logger (typically __name__)

    Returns:
        A logging.Logger instance
    """
    return logging.getLogger(name)


def set_level(logger_name: str, level: LogLevel) -> None:
    """
    Set the log level for a specific logger.

    Args:
        logger_name: The name of the logger to configure
        level: The log level to set
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level.upper()))


def reset_logging() -> None:
    """
    Reset the logging configuration. Useful for testing.
    """
    global _configured
    _configured = False
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)

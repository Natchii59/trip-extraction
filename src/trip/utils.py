"""
Utility functions for the trip package.

This module provides common utilities for logging, formatting,
and other helper functions.
"""

import logging
from pathlib import Path
from typing import Optional

from .config import get_config


def setup_logging(
    level: Optional[int | str] = None, log_file: Optional[str | Path] = None, force: bool = True
):
    """
    Configure logging for the application.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG, "INFO", "DEBUG").
              If None, uses level from config.
        log_file: Optional file path to write logs to.
                 If None, uses config setting.
        force: If True, override existing logging configuration.

    Examples:
        >>> setup_logging(level="DEBUG")
        >>> setup_logging(level=logging.INFO, log_file="app.log")
    """
    config = get_config()

    # Resolve level
    if level is None:
        level = getattr(logging, config.logging.level)
    elif isinstance(level, str):
        level = getattr(logging, level.upper())

    # Resolve log file
    if log_file is None and config.logging.log_to_file:
        log_file = config.logging.log_file

    handlers = [logging.StreamHandler()]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=level,
        format=config.logging.format,
        datefmt=config.logging.date_format,
        handlers=handlers,
        force=force,
    )


def format_trip_result(departure: Optional[str], arrival: Optional[str]) -> str:
    """
    Format trip extraction result for display.

    Args:
        departure: Departure city or None.
        arrival: Arrival city or None.

    Returns:
        Formatted string representation of the trip.
    """
    if departure and arrival:
        return f"{departure} → {arrival}"
    elif departure:
        return f"{departure} → ?"
    elif arrival:
        return f"? → {arrival}"
    else:
        return "No trip information found"

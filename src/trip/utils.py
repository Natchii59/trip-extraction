"""
Utility functions for the trip package.
"""

import logging
from typing import Optional


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None):
    """
    Configure logging for the application.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional file path to write logs to.
    """
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
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

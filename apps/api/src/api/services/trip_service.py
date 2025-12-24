"""
Trip Parser service for the API.

This module provides a singleton wrapper around the TripParser
to ensure the models are loaded once and reused across requests.
"""

import logging
from typing import Optional

from trip_parser import TripParser
from trip_parser.exceptions import InvalidInputError, ModelNotFoundError, TripExtractionError

from ..schemas import TripParseResponse

logger = logging.getLogger(__name__)


class TripParserService:
    """
    Service class for trip parsing.

    Manages a singleton TripParser instance to avoid loading models
    multiple times. The parser is loaded lazily on first use.
    """

    _instance: Optional["TripParserService"] = None
    _parser: TripParser | None = None
    _initialized: bool = False

    def __new__(cls) -> "TripParserService":
        """Singleton pattern: return the same instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the service (only once)."""
        if not self._initialized:
            logger.info("Initializing TripParserService")
            self._initialized = True

    def _get_parser(self) -> TripParser:
        """
        Get the TripParser instance, loading it if necessary.

        Returns:
            TripParser instance.

        Raises:
            ModelNotFoundError: If models are not found.
        """
        if self._parser is None:
            logger.info("Loading TripParser models...")
            try:
                self._parser = TripParser()
                logger.info("TripParser models loaded successfully")
            except ModelNotFoundError as e:
                logger.error(f"Failed to load models: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error loading models: {e}", exc_info=True)
                raise TripExtractionError(f"Failed to initialize parser: {str(e)}") from e

        return self._parser

    def parse_trip(self, text: str) -> TripParseResponse:
        """
        Parse trip information from text.

        Args:
            text: Input text describing a trip in French.

        Returns:
            TripParseResponse with departure and arrival cities.

        Raises:
            InvalidInputError: If input is invalid.
            ModelNotFoundError: If models are not found.
            TripExtractionError: For other extraction errors.
        """
        try:
            parser = self._get_parser()
            departure, arrival = parser.parse_trip(text)

            # Determine success based on whether we found both cities
            success = departure is not None and arrival is not None
            message = None

            if not success:
                if departure is None and arrival is None:
                    message = "Could not extract departure and arrival cities from the text"
                elif departure is None:
                    message = "Could not extract departure city from the text"
                else:
                    message = "Could not extract arrival city from the text"

            return TripParseResponse(
                departure=departure,
                arrival=arrival,
                success=success,
                message=message,
            )

        except InvalidInputError:
            # Re-raise input validation errors
            raise

        except (ModelNotFoundError, TripExtractionError):
            # Re-raise known extraction errors
            raise

        except Exception as e:
            # Log and wrap unexpected errors
            logger.error(f"Unexpected error in parse_trip: {e}", exc_info=True)
            raise TripExtractionError(f"Unexpected error: {str(e)}") from e

    def is_ready(self) -> bool:
        """
        Check if the service is ready (models loaded).

        Returns:
            True if models are loaded, False otherwise.
        """
        return self._parser is not None

    @classmethod
    def reset(cls):
        """
        Reset the singleton instance.

        Useful for testing or forcing a reload of models.
        """
        if cls._instance is not None:
            cls._instance._parser = None
            cls._instance._initialized = False
            cls._instance = None
            logger.info("TripParserService reset")

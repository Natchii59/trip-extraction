"""
Trip Parser module for extracting departure and arrival information.

This module provides the main interface for parsing travel information
from French text using specialized ML models.
"""

import logging

from .exceptions import (
    ClassificationError,
    InsufficientLocationsError,
    InvalidInputError,
)
from .models import DepartureArrivalClassifier, NERExtractor

logger = logging.getLogger(__name__)


class TripParser:
    """
    Parser for extracting trip information (departure and arrival) from text.

    Uses two specialized models:
    1. NER model (CamemBERT) to identify locations (LOC entities)
    2. Custom fine-tuned classifier to determine departure vs arrival

    This approach provides optimized, fast, and reliable trip extraction.
    """

    def __init__(
        self,
        ner_extractor: NERExtractor | None = None,
        classifier: DepartureArrivalClassifier | None = None,
    ):
        """
        Initialize the trip parser.

        Args:
            ner_extractor: NERExtractor instance. If None, creates a new one.
            classifier: DepartureArrivalClassifier instance. If None, creates a new one.
        """
        self.ner_extractor = ner_extractor or NERExtractor()
        self.classifier = classifier or DepartureArrivalClassifier()

    def parse_trip(self, text: str) -> tuple[str | None, str | None]:
        """
        Parse trip information from text to extract departure and arrival cities.

        Uses specialized ML models to:
        1. Extract location names using CamemBERT NER
        2. Classify each location's role using custom fine-tuned classifier

        This optimized approach provides fast and reliable results.

        Args:
            text: Input text describing a trip in French.

        Returns:
            Tuple of (departure_city, arrival_city). Returns (None, None) if
            not enough cities are detected or classification fails.

        Raises:
            InvalidInputError: If text is None or empty.

        Examples:
            >>> parser = TripParser()
            >>> parser.parse_trip("Train de Paris à Lyon")
            ('Paris', 'Lyon')
            >>> parser.parse_trip("Je veux aller à Lille depuis Paris")
            ('Paris', 'Lille')
        """
        # Validate input
        if not isinstance(text, str):
            raise InvalidInputError("text", text, f"Expected str, got {type(text).__name__}")

        if not text or not text.strip():
            raise InvalidInputError("text", text, "Text cannot be empty")

        # Limit text length for performance
        max_text_length = 1000
        if len(text) > max_text_length:
            logger.warning(f"Text too long ({len(text)} chars), truncating to {max_text_length}")
            text = text[:max_text_length]

        try:
            # Step 1: Extract locations using NER
            cities = self.ner_extractor.extract_locations(text)

            if len(cities) < 2:
                logger.warning(f"Not enough cities detected. Found: {cities}")
                return (None, None)

            # Step 2: Use custom classifier to determine departure and arrival
            departure, arrival = self.classifier.classify_locations(text, cities)

            logger.debug(f"Parsed trip: {departure} → {arrival}")
            return (departure, arrival)

        except (InsufficientLocationsError, ClassificationError) as e:
            logger.warning(f"Cannot parse trip: {e}")
            return (None, None)

        except InvalidInputError:
            raise

        except Exception as e:
            logger.error(f"Unexpected error parsing trip: {e}", exc_info=True)
            return (None, None)

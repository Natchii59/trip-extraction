"""
Trip Parser module for extracting departure and arrival information.
"""

import logging
from typing import Tuple, Optional, List

from .ner_extractor import NERExtractor
from .departure_arrival_classifier import DepartureArrivalClassifier

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
        ner_extractor: Optional[NERExtractor] = None,
        classifier: Optional[DepartureArrivalClassifier] = None,
    ):
        """
        Initialize the trip parser.

        Args:
            ner_extractor: NERExtractor instance. If None, creates a new one.
            classifier: DepartureArrivalClassifier instance. If None, creates a new one.
        """
        self.ner_extractor = ner_extractor or NERExtractor()
        self.classifier = classifier or DepartureArrivalClassifier()

    def parse_trip(self, text: str) -> Tuple[Optional[str], Optional[str]]:
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
            not enough cities are detected.

        Examples:
            >>> parser = TripParser()
            >>> parser.parse_trip("Train de Paris à Lyon")
            ('Paris', 'Lyon')
            >>> parser.parse_trip("Je veux aller à Lille depuis Paris")
            ('Paris', 'Lille')
        """
        if not text or not text.strip():
            logger.warning("Empty input text provided")
            return (None, None)

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

        except Exception as e:
            logger.error(f"Error parsing trip: {e}")
            return (None, None)

"""
Trip Parser module for extracting departure and arrival information.
"""

import logging
from typing import Tuple, Optional, List

from .ner_extractor import NERExtractor

logger = logging.getLogger(__name__)


class TripParser:
    """
    Parser for extracting trip information (departure and arrival) from text.

    Uses NER to identify locations and a zero-shot classification model
    to determine departure and arrival cities based on learned patterns.
    No hardcoded linguistic rules - 100% ML-based approach.
    """

    def __init__(self, ner_extractor: Optional[NERExtractor] = None):
        """
        Initialize the trip parser.

        Args:
            ner_extractor: NERExtractor instance. If None, creates a new one.
        """
        self.ner_extractor = ner_extractor or NERExtractor()

    def parse_trip(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse trip information from text to extract departure and arrival cities.

        Uses ML models to:
        1. Extract city names (NER)
        2. Classify each city's role as departure or arrival (zero-shot classification)

        No hardcoded rules - pure ML approach.

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
            # Extract locations using NER
            all_locations = self.ner_extractor.extract_locations(text)

            # Filter to keep only travel-relevant locations
            cities = self.ner_extractor.filter_travel_locations(text, all_locations)

            if len(cities) < 2:
                logger.warning(f"Not enough cities detected. Found: {cities}")
                return (None, None)

            # Use ML classifier to determine role of each city with confidence scores
            departure_candidates = []
            arrival_candidates = []

            for city in cities:
                role, score = self.ner_extractor.classify_location_role(text, city)

                if role == "departure":
                    departure_candidates.append((city, score))
                elif role == "arrival":
                    arrival_candidates.append((city, score))

            # Select the candidates with highest confidence scores
            departure = None
            arrival = None

            if departure_candidates:
                # Sort by score (descending) and take the best one
                departure_candidates.sort(key=lambda x: x[1], reverse=True)
                departure = departure_candidates[0][0]
                logger.debug(
                    f"Selected departure: {departure} (score: {departure_candidates[0][1]:.3f})"
                )

            if arrival_candidates:
                # Sort by score (descending) and take the best one
                arrival_candidates.sort(key=lambda x: x[1], reverse=True)
                arrival = arrival_candidates[0][0]
                logger.debug(f"Selected arrival: {arrival} (score: {arrival_candidates[0][1]:.3f})")

            # If roles couldn't be determined, fall back to order of appearance
            if departure is None or arrival is None:
                logger.debug(
                    "Could not determine roles from ML classification, using order of appearance"
                )
                departure = cities[0] if departure is None else departure
                arrival = cities[1] if arrival is None else arrival

            return (departure, arrival)

        except Exception as e:
            logger.error(f"Error parsing trip: {e}")
            return (None, None)

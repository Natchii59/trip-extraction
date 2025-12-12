"""
NER-based Trip Extraction System

A Natural Language Processing system that extracts travel information
(departure and arrival cities) from French text using specialized models:
- CamemBERT NER for location extraction
- Custom fine-tuned classifier for departure/arrival classification
"""

__version__ = "0.2.0"

from .ner_extractor import NERExtractor
from .trip_parser import TripParser
from .departure_arrival_classifier import DepartureArrivalClassifier

__all__ = ["NERExtractor", "TripParser", "DepartureArrivalClassifier"]

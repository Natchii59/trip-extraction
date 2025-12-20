"""
NER-based Trip Extraction System

A Natural Language Processing system that extracts travel information
(departure and arrival cities) from French text using specialized models:
- CamemBERT NER for location extraction
- Custom fine-tuned classifier for departure/arrival classification
"""

__version__ = "0.1.0"

from .config import Config, get_config
from .exceptions import (
    ClassificationError,
    InsufficientLocationsError,
    InvalidInputError,
    ModelNotFoundError,
    TripExtractionError,
)
from .models import DepartureArrivalClassifier, NERExtractor
from .trip_parser import TripParser

__all__ = [
    "NERExtractor",
    "TripParser",
    "DepartureArrivalClassifier",
    "get_config",
    "Config",
    "TripExtractionError",
    "ModelNotFoundError",
    "InsufficientLocationsError",
    "InvalidInputError",
    "ClassificationError",
]

"""
NER-based Trip Extraction System

A Natural Language Processing system that extracts travel information
(departure and arrival cities) from French text using specialized models:
- CamemBERT NER for location extraction
- Custom fine-tuned classifier for departure/arrival classification
"""

__version__ = "0.3.0"

from .models import NERExtractor, DepartureArrivalClassifier
from .trip_parser import TripParser
from .config import get_config, Config
from .exceptions import (
    TripExtractionError,
    ModelNotFoundError,
    InsufficientLocationsError,
    InvalidInputError,
    ClassificationError,
)

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

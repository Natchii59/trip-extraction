"""
NER-based Trip Extraction System

A Natural Language Processing system that extracts travel information
(departure and arrival cities) from French text using CamemBERT NER.
"""

__version__ = "0.1.0"

from .ner_extractor import NERExtractor
from .trip_parser import TripParser

__all__ = ["NERExtractor", "TripParser"]

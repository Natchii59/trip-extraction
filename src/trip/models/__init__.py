"""
Models subpackage for the Trip Extraction system.

This package contains all ML model implementations:
- NER extraction
- Departure-Arrival classification
"""

from .classifier import DepartureArrivalClassifier
from .ner import NERExtractor

__all__ = ["NERExtractor", "DepartureArrivalClassifier"]

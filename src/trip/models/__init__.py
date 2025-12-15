"""
Models subpackage for the Trip Extraction system.

This package contains all ML model implementations:
- NER extraction
- Departure-Arrival classification
"""

from .ner import NERExtractor
from .classifier import DepartureArrivalClassifier

__all__ = ["NERExtractor", "DepartureArrivalClassifier"]

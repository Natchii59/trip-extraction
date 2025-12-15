"""
Base classes and protocols for ML models.

This module defines common interfaces and base functionality
for all models in the system.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Any
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all ML models in the system.

    Provides common functionality like model loading and error handling.
    """

    def __init__(self):
        """Initialize the base model."""
        self._model = None
        self._tokenizer = None

    @abstractmethod
    def _load_model(self):
        """Load the model and tokenizer. Must be implemented by subclasses."""
        pass

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self._model is not None

    def __repr__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(loaded={self.is_loaded})"


class Extractor(Protocol):
    """Protocol for entity extraction models."""

    def extract_entities(self, text: str) -> list[dict[str, Any]]:
        """Extract entities from text."""
        ...

    def extract_locations(self, text: str) -> list[str]:
        """Extract location entities from text."""
        ...


class Classifier(Protocol):
    """Protocol for classification models."""

    def classify_location(self, text: str, location: str) -> tuple[str, float]:
        """Classify a single location."""
        ...

    def classify_locations(self, text: str, locations: list[str]) -> tuple[str | None, str | None]:
        """Classify multiple locations and determine departure/arrival."""
        ...

"""
NER Extractor module using CamemBERT for French entity recognition.

This module provides Named Entity Recognition capabilities specifically
optimized for extracting location entities from French text.
"""

import logging

from transformers import (
    AutoModelForTokenClassification,
    CamembertTokenizer,
    pipeline,
)

from ..config import get_config
from ..exceptions import InvalidInputError, ModelLoadError
from .base import BaseModel

logger = logging.getLogger(__name__)


class NERExtractor(BaseModel):
    """
    Named Entity Recognition extractor for French text using CamemBERT.

    This class wraps the CamemBERT NER model to extract named entities,
    particularly locations, from French text.

    Note: This class only handles location extraction (NER).
    Classification of departure/arrival is handled by DepartureArrivalClassifier.

    Attributes:
        model_name: Name of the Hugging Face model used for NER.

    Examples:
        >>> extractor = NERExtractor()
        >>> locations = extractor.extract_locations("Je vais de Paris à Lyon")
        >>> print(locations)
        ['Paris', 'Lyon']
    """

    def __init__(self, model_name: str | None = None):
        """
        Initialize the NER extractor with a pre-trained model.

        Args:
            model_name: Name of the Hugging Face model to use for NER.
                       Defaults to the value from config (Jean-Baptiste/camembert-ner).

        Raises:
            ModelLoadError: If the model cannot be loaded.
            InvalidInputError: If model_name is empty.
        """
        super().__init__()
        config = get_config()
        self.model_name = model_name or config.model.ner_model_name

        if not self.model_name:
            raise InvalidInputError("model_name", self.model_name, "Model name cannot be empty")

        self._pipeline = None
        self._load_model()

    def _load_model(self):
        """
        Load the tokenizer, model, and create the NER pipeline.

        Raises:
            ModelLoadError: If model loading fails.
        """
        try:
            logger.info(f"Loading NER model: {self.model_name}")

            # Load CamemBERT tokenizer directly to avoid fast tokenizer conversion issues
            tokenizer = CamembertTokenizer.from_pretrained(self.model_name)
            model = AutoModelForTokenClassification.from_pretrained(self.model_name)

            # Create the NER pipeline with explicitly loaded components
            self._pipeline = pipeline(
                "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple"
            )
            self._model = model
            self._tokenizer = tokenizer

            logger.info("NER model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load NER model: {e}")
            raise ModelLoadError(self.model_name, e) from e

    def extract_entities(self, text: str) -> list[dict[str, any]]:  # type: ignore[valid-type]
        """
        Extract all named entities from the given text.

        Args:
            text: Input text in French.

        Returns:
            List of entities with their type, text, and confidence score.
            Each entity is a dict with keys: entity_group, word, score, start, end.

        Raises:
            InvalidInputError: If text is empty or None.

        Examples:
            >>> extractor = NERExtractor()
            >>> entities = extractor.extract_entities("Paris est en France")
            >>> print(entities)
            [{'entity_group': 'LOC', 'word': 'Paris', 'score': 0.99, ...}, ...]
        """
        if not text or not text.strip():
            raise InvalidInputError("text", text, "Input text cannot be empty")

        try:
            entities = self._pipeline(text)  # type: ignore[misc]
            logger.debug(f"Extracted {len(entities)} entities from text")
            return entities

        except Exception as e:
            logger.error(f"Error during NER extraction: {e}")
            raise

    def _split_compound_locations(self, location: str) -> list[str]:
        """
        Split compound location strings into individual cities.

        Handles cases like "Paris Marseille" -> ["Paris", "Marseille"]

        Args:
            location: A location string that may contain multiple city names.

        Returns:
            List of individual city names.

        Examples:
            >>> extractor = NERExtractor()
            >>> extractor._split_compound_locations("Paris Marseille")
            ['Paris', 'Marseille']
            >>> extractor._split_compound_locations("New York")
            ['New York']
        """
        # Split on whitespace
        words = location.split()

        # If we have multiple words that start with capital letters,
        # treat them as separate cities
        if len(words) > 1:
            # Check if all words start with capital letters (likely city names)
            if all(word and word[0].isupper() for word in words):
                return words

        return [location]

    def extract_locations(self, text: str) -> list[str]:
        """
        Extract location entities (cities, places) from text.

        Args:
            text: Input text in French.

        Returns:
            List of location names found in the text, with compound
            locations split into individual cities.

        Raises:
            InvalidInputError: If text is empty or None.

        Examples:
            >>> extractor = NERExtractor()
            >>> locations = extractor.extract_locations("Train de Paris à Lyon")
            >>> print(locations)
            ['Paris', 'Lyon']
        """
        entities = self.extract_entities(text)
        locations = []

        for ent in entities:
            if ent["entity_group"] == "LOC":
                location = str(ent["word"]).strip()
                # Try to split compound locations (e.g., "Paris Marseille")
                split_locations = self._split_compound_locations(location)
                locations.extend(split_locations)

        logger.debug(f"Found {len(locations)} locations: {locations}")
        return locations

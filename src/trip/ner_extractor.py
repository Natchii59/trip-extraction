"""
NER Extractor module using CamemBERT for French entity recognition.
"""

import logging
from typing import List, Dict, Optional
from transformers import (
    AutoModelForTokenClassification,
    pipeline,
    CamembertTokenizer,
)

logger = logging.getLogger(__name__)


class NERExtractor:
    """
    Named Entity Recognition extractor for French text using CamemBERT.

    This class wraps the CamemBERT NER model to extract named entities,
    particularly locations, from French text.

    Note: This class only handles location extraction (NER).
    Classification of departure/arrival is handled by DepartureArrivalClassifier.
    """

    DEFAULT_MODEL = "Jean-Baptiste/camembert-ner"

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the NER extractor with a pre-trained model.

        Args:
            model_name: Name of the Hugging Face model to use for NER.
                       Defaults to Jean-Baptiste/camembert-ner.

        Raises:
            RuntimeError: If the model cannot be loaded.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self._pipeline = None
        self._load_model()

    def _load_model(self):
        """Load the tokenizer, model, and create the NER pipeline."""
        try:
            logger.info(f"Loading NER model: {self.model_name}")

            # Ensure model_name is not None
            if not self.model_name:
                raise ValueError("Model name cannot be None or empty")

            # Load CamemBERT tokenizer directly to avoid fast tokenizer conversion issues
            tokenizer = CamembertTokenizer.from_pretrained(self.model_name)
            model = AutoModelForTokenClassification.from_pretrained(self.model_name)

            # Create the NER pipeline with explicitly loaded components
            self._pipeline = pipeline(
                "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple"
            )
            logger.info("NER model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Could not load NER model: {e}")

    def extract_entities(self, text: str) -> List[Dict[str, any]]:
        """
        Extract all named entities from the given text.

        Args:
            text: Input text in French.

        Returns:
            List of entities with their type, text, and confidence score.
            Each entity is a dict with keys: entity_group, word, score, start, end.

        Raises:
            ValueError: If text is empty or None.
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        try:
            entities = self._pipeline(text)
            logger.debug(f"Extracted {len(entities)} entities from text")
            return entities
        except Exception as e:
            logger.error(f"Error during NER extraction: {e}")
            raise

    def _split_compound_locations(self, location: str) -> List[str]:
        """
        Split compound location strings (e.g., "Paris Marseille" -> ["Paris", "Marseille"]).

        Args:
            location: A location string that may contain multiple city names.

        Returns:
            List of individual city names.
        """
        # Split on whitespace
        words = location.split()

        # If we have multiple words that start with capital letters,
        # treat them as separate cities
        if len(words) > 1:
            # Check if all words start with capital letters (likely city names)
            if all(word[0].isupper() for word in words if word):
                return words

        return [location]

    def extract_locations(self, text: str) -> List[str]:
        """
        Extract location entities (cities, places) from text.

        Args:
            text: Input text in French.

        Returns:
            List of location names found in the text.
        """
        entities = self.extract_entities(text)
        locations = []

        for ent in entities:
            if ent["entity_group"] == "LOC":
                location = ent["word"].strip()
                # Try to split compound locations (e.g., "Paris Marseille")
                split_locations = self._split_compound_locations(location)
                locations.extend(split_locations)

        logger.debug(f"Found {len(locations)} locations: {locations}")
        return locations

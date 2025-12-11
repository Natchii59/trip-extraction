"""
NER Extractor module using CamemBERT for French entity recognition.
"""

import logging
from typing import List, Dict, Optional, Tuple
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
    """

    DEFAULT_MODEL = "Jean-Baptiste/camembert-ner"
    DEFAULT_CLASSIFIER_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

    def __init__(self, model_name: Optional[str] = None, classifier_model: Optional[str] = None):
        """
        Initialize the NER extractor with a pre-trained model.

        Args:
            model_name: Name of the Hugging Face model to use for NER.
                       Defaults to Jean-Baptiste/camembert-ner.
            classifier_model: Name of the model for zero-shot classification.
                            Defaults to MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7.

        Raises:
            RuntimeError: If the model cannot be loaded.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.classifier_model = classifier_model or self.DEFAULT_CLASSIFIER_MODEL
        self._pipeline = None
        self._classifier_pipeline = None
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

            # Load zero-shot classification model for relation extraction
            logger.info(f"Loading classifier model: {self.classifier_model}")
            self._classifier_pipeline = pipeline(
                "zero-shot-classification", model=self.classifier_model
            )
            logger.info("Classifier model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Could not load models: {e}")

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

    def filter_travel_locations(self, text: str, locations: List[str]) -> List[str]:
        """
        Filter locations to keep only those relevant for travel by checking
        the immediate context around each location.

        Args:
            text: The full sentence context.
            locations: List of detected locations.

        Returns:
            Filtered list of travel-relevant locations.
        """
        if len(locations) <= 2:
            # If we have 2 or fewer locations, keep them all
            return locations

        filtered = []
        text_lower = text.lower()

        for location in locations:
            try:
                # Find the location in the text to get context
                loc_lower = location.lower()
                loc_index = text_lower.find(loc_lower)

                if loc_index == -1:
                    # Can't find location, keep it to be safe
                    filtered.append(location)
                    continue

                # Get context window around the location (20 chars before and after)
                start = max(0, loc_index - 20)
                end = min(len(text), loc_index + len(location) + 20)
                context = text[start:end]

                # Check if this specific context indicates it's not a place
                result = self._classifier_pipeline(
                    f"Le contexte '{context}' indique que {location} est",
                    ["un lieu géographique", "une boisson ou nourriture"],
                    multi_label=False,
                )

                is_place = result["labels"][0] == "un lieu géographique"
                logger.debug(
                    f"Location '{location}' (context: '{context}'): {result['labels'][0]} (score: {result['scores'][0]:.3f})"
                )

                if is_place:
                    filtered.append(location)
                else:
                    logger.info(
                        f"Filtered out '{location}' - classified as non-place in context '{context}'"
                    )
            except Exception as e:
                logger.warning(f"Could not classify '{location}', keeping it: {e}")
                filtered.append(location)

        return filtered if filtered else locations

    def classify_location_role(self, text: str, location: str) -> Tuple[str, float]:
        """
        Use zero-shot classification to determine if a location is a departure or arrival.

        Args:
            text: The full sentence context.
            location: The location to classify.

        Returns:
            Tuple of (role, confidence_score) where role is 'departure', 'arrival', or 'unknown'.
        """
        try:
            # Create a focused query about this location
            hypothesis_template = f"Cette phrase indique que {location} est {{}}."
            candidate_labels = ["le point de départ", "la destination"]

            result = self._classifier_pipeline(
                text, candidate_labels, hypothesis_template=hypothesis_template
            )

            # Get the top prediction
            if result["labels"][0] == "le point de départ":
                logger.debug(
                    f"Classified '{location}' as DEPARTURE (score: {result['scores'][0]:.3f})"
                )
                return ("departure", result["scores"][0])
            else:
                logger.debug(
                    f"Classified '{location}' as ARRIVAL (score: {result['scores'][0]:.3f})"
                )
                return ("arrival", result["scores"][0])
        except Exception as e:
            logger.warning(f"Classification failed for '{location}': {e}")
            return ("unknown", 0.0)

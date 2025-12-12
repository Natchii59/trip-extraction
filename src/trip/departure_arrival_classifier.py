"""
Departure-Arrival Classifier module.

This module contains a custom classifier trained specifically for identifying
departure and arrival locations in French travel sentences.
"""

import logging
import torch
from typing import Tuple, Optional
from pathlib import Path
from transformers import (
    CamembertTokenizer,
    CamembertForSequenceClassification,
)

logger = logging.getLogger(__name__)


class DepartureArrivalClassifier:
    """
    Custom classifier for identifying departure and arrival locations.

    This classifier uses a fine-tuned CamemBERT model trained specifically
    on travel sentence patterns to determine whether a location is a
    departure point or an arrival destination.

    The model is trained to classify locations marked with [LOC] tags
    in the context of the full sentence.
    """

    DEFAULT_MODEL_PATH = "./models/departure_arrival_classifier"

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the classifier.

        Args:
            model_path: Path to the fine-tuned model directory.
                       If None, uses the default path.

        Raises:
            RuntimeError: If the model cannot be loaded.
        """
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the fine-tuned model and tokenizer."""
        try:
            model_path = Path(self.model_path)

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found at {self.model_path}. "
                    f"Please train the model first by running: python train_model.py"
                )

            logger.info(f"Loading departure-arrival classifier from {self.model_path}")

            self._tokenizer = CamembertTokenizer.from_pretrained(self.model_path)
            self._model = CamembertForSequenceClassification.from_pretrained(self.model_path)
            self._model.to(self.device)
            self._model.eval()

            logger.info("Departure-arrival classifier loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load classifier model: {e}")
            raise RuntimeError(f"Could not load classifier model: {e}")

    def classify_location(self, text: str, location: str) -> Tuple[str, float]:
        """
        Classify whether a location is a departure or arrival point.

        Args:
            text: The full sentence context.
            location: The location to classify.

        Returns:
            Tuple of (role, confidence) where:
                - role is 'departure' or 'arrival'
                - confidence is a float between 0 and 1

        Examples:
            >>> classifier = DepartureArrivalClassifier()
            >>> classifier.classify_location("Je vais de Paris à Lyon", "Paris")
            ('departure', 0.98)
            >>> classifier.classify_location("Je vais de Paris à Lyon", "Lyon")
            ('arrival', 0.97)
        """
        if not text or not location:
            logger.warning("Empty text or location provided")
            return ("unknown", 0.0)

        try:
            # Find location in text (case-insensitive)
            text_lower = text.lower()
            loc_lower = location.lower()
            loc_pos = text_lower.find(loc_lower)

            if loc_pos == -1:
                logger.warning(f"Location '{location}' not found in text: {text}")
                return ("unknown", 0.0)

            # Get actual location with correct casing from text
            loc_in_text = text[loc_pos : loc_pos + len(location)]

            # Create input text with location marker using special tokens
            marked_text = text.replace(loc_in_text, f"[LOC] {loc_in_text} [/LOC]", 1)

            # Tokenize with max_length matching training config
            inputs = self._tokenizer(
                marked_text,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Predict
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0, predicted_class].item()

            # Map class to role
            role = "departure" if predicted_class == 0 else "arrival"

            logger.debug(
                f"Classified '{location}' as {role.upper()} " f"(confidence: {confidence:.3f})"
            )

            return (role, confidence)

        except Exception as e:
            logger.error(f"Error classifying location '{location}': {e}")
            return ("unknown", 0.0)

    def classify_locations(self, text: str, locations: list) -> Tuple[Optional[str], Optional[str]]:
        """
        Classify multiple locations and determine departure and arrival.

        Args:
            text: The full sentence context.
            locations: List of locations to classify.

        Returns:
            Tuple of (departure, arrival). Returns (None, None) if
            locations cannot be classified.

        Examples:
            >>> classifier = DepartureArrivalClassifier()
            >>> classifier.classify_locations(
            ...     "Train de Paris à Lyon",
            ...     ["Paris", "Lyon"]
            ... )
            ('Paris', 'Lyon')
        """
        if not locations or len(locations) < 2:
            logger.warning(f"Need at least 2 locations, got {len(locations)}")
            return (None, None)

        # Classify each location
        departure_candidates = []
        arrival_candidates = []

        for location in locations:
            role, confidence = self.classify_location(text, location)

            if role == "departure":
                departure_candidates.append((location, confidence))
            elif role == "arrival":
                arrival_candidates.append((location, confidence))

        # Select best candidates - 100% ML
        departure = None
        arrival = None

        if departure_candidates:
            departure_candidates.sort(key=lambda x: x[1], reverse=True)
            departure = departure_candidates[0][0]
            logger.debug(
                f"Selected departure: {departure} "
                f"(confidence: {departure_candidates[0][1]:.3f})"
            )

        if arrival_candidates:
            arrival_candidates.sort(key=lambda x: x[1], reverse=True)
            arrival = arrival_candidates[0][0]
            logger.debug(
                f"Selected arrival: {arrival} " f"(confidence: {arrival_candidates[0][1]:.3f})"
            )

        if departure is None or arrival is None:
            logger.warning(f"Model failed. Add this pattern to training: '{text}'")
            return (None, None)

        return (departure, arrival)

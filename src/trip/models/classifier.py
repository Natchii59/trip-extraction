"""
Departure-Arrival Classifier module.

This module contains a custom classifier trained specifically for identifying
departure and arrival locations in French travel sentences.
"""

import logging
from pathlib import Path

import torch
from transformers import (
    CamembertForSequenceClassification,
    CamembertTokenizer,
)

from ..config import get_config
from ..exceptions import (
    ClassificationError,
    InsufficientLocationsError,
    InvalidInputError,
    ModelLoadError,
    ModelNotFoundError,
)
from .base import BaseModel

logger = logging.getLogger(__name__)


class DepartureArrivalClassifier(BaseModel):
    """
    Custom classifier for identifying departure and arrival locations.

    This classifier uses a fine-tuned CamemBERT model trained specifically
    on travel sentence patterns to determine whether a location is a
    departure point or an arrival destination.

    The model is trained to classify locations marked with [LOC] tags
    in the context of the full sentence.

    Attributes:
        model_path: Path to the fine-tuned model directory.
        device: PyTorch device (cuda/cpu) used for inference.

    Examples:
        >>> classifier = DepartureArrivalClassifier()
        >>> role, conf = classifier.classify_location("Je vais de Paris à Lyon", "Paris")
        >>> print(f"{role}: {conf:.2f}")
        departure: 0.98
    """

    def __init__(self, model_path: str | Path | None = None):
        """
        Initialize the classifier.

        Args:
            model_path: Path to the fine-tuned model directory.
                       If None, uses the default path from config.

        Raises:
            ModelNotFoundError: If the model directory doesn't exist.
            ModelLoadError: If the model cannot be loaded.
        """
        super().__init__()
        config = get_config()

        # Use provided path or default from config
        if model_path is None:
            self.model_path = config.paths.departure_arrival_model
        else:
            self.model_path = Path(model_path)

        # Auto-detect device or use config
        device_name = config.model.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_name)

        logger.info(f"Using device: {self.device}")
        self._load_model()

    def _load_model(self):
        """
        Load the fine-tuned model and tokenizer.

        Raises:
            ModelNotFoundError: If model directory doesn't exist.
            ModelLoadError: If model loading fails.
        """
        try:
            if not self.model_path.exists():
                raise ModelNotFoundError(str(self.model_path))

            logger.info(f"Loading departure-arrival classifier from {self.model_path}")

            self._tokenizer = CamembertTokenizer.from_pretrained(str(self.model_path))
            self._model = CamembertForSequenceClassification.from_pretrained(str(self.model_path))
            self._model.to(self.device)
            self._model.eval()

            logger.info("Departure-arrival classifier loaded successfully")

        except ModelNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to load classifier model: {e}")
            raise ModelLoadError(str(self.model_path), e) from e

    def classify_location(
        self, text: str, location: str, confidence_threshold: float | None = None
    ) -> tuple[str, float]:
        """
        Classify whether a location is a departure or arrival point.

        Args:
            text: The full sentence context.
            location: The location to classify.
            confidence_threshold: Minimum confidence threshold (0-1).
                                If None, uses value from config.

        Returns:
            Tuple of (role, confidence) where:
                - role is 'departure', 'arrival', or 'unknown'
                - confidence is a float between 0 and 1

        Raises:
            InvalidInputError: If text or location is empty.

        Examples:
            >>> classifier = DepartureArrivalClassifier()
            >>> role, conf = classifier.classify_location("Je vais de Paris à Lyon", "Paris")
            >>> print(f"{role}: {conf:.2f}")
            departure: 0.98
            >>> role, conf = classifier.classify_location("Je vais de Paris à Lyon", "Lyon")
            >>> print(f"{role}: {conf:.2f}")
            arrival: 0.97
        """
        if not text or not text.strip():
            raise InvalidInputError("text", text, "Text cannot be empty")

        if not location or not location.strip():
            raise InvalidInputError("location", location, "Location cannot be empty")

        # Use config threshold if not provided
        if confidence_threshold is None:
            config = get_config()
            confidence_threshold = config.model.confidence_threshold

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

            # Get max_length from config
            config = get_config()
            max_length = config.training.max_length

            # Tokenize with max_length matching training config
            inputs = self._tokenizer(
                marked_text,
                max_length=max_length,
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
                predicted_class = int(torch.argmax(probabilities, dim=-1).item())
                confidence = float(probabilities[0, predicted_class].item())

            # Map class to role
            role = "departure" if predicted_class == 0 else "arrival"

            # Check confidence threshold
            if confidence < confidence_threshold:
                logger.debug(
                    f"Low confidence ({confidence:.3f}) for '{location}', " f"marking as unknown"
                )
                return ("unknown", confidence)

            logger.debug(
                f"Classified '{location}' as {role.upper()} " f"(confidence: {confidence:.3f})"
            )

            return (role, confidence)

        except Exception as e:
            logger.error(f"Error classifying location '{location}': {e}")
            return ("unknown", 0.0)

    def classify_locations(self, text: str, locations: list[str]) -> tuple[str | None, str | None]:
        """
        Classify multiple locations and determine departure and arrival.

        Args:
            text: The full sentence context.
            locations: List of locations to classify.

        Returns:
            Tuple of (departure, arrival). Returns (None, None) if
            locations cannot be classified.

        Raises:
            InvalidInputError: If text is empty.
            InsufficientLocationsError: If fewer than 2 locations provided.
            ClassificationError: If classification fails and pattern should be added to training.

        Examples:
            >>> classifier = DepartureArrivalClassifier()
            >>> departure, arrival = classifier.classify_locations(
            ...     "Train de Paris à Lyon",
            ...     ["Paris", "Lyon"]
            ... )
            >>> print(f"{departure} → {arrival}")
            Paris → Lyon
        """
        if not text or not text.strip():
            raise InvalidInputError("text", text, "Text cannot be empty")

        if not locations or len(locations) < 2:
            raise InsufficientLocationsError(
                found_count=len(locations) if locations else 0, required_count=2
            )

        # Classify each location
        departure_candidates = []
        arrival_candidates = []

        for location in locations:
            try:
                role, confidence = self.classify_location(text, location)

                if role == "departure":
                    departure_candidates.append((location, confidence))
                elif role == "arrival":
                    arrival_candidates.append((location, confidence))

            except InvalidInputError as e:
                logger.warning(f"Skipping invalid location: {e}")
                continue

        # Select best candidates based on confidence
        departure = None
        arrival = None

        if departure_candidates:
            # Sort by confidence (highest first)
            departure_candidates.sort(key=lambda x: x[1], reverse=True)
            departure = departure_candidates[0][0]
            logger.debug(
                f"Selected departure: {departure} "
                f"(confidence: {departure_candidates[0][1]:.3f})"
            )

        if arrival_candidates:
            # Sort by confidence (highest first)
            arrival_candidates.sort(key=lambda x: x[1], reverse=True)
            arrival = arrival_candidates[0][0]
            logger.debug(
                f"Selected arrival: {arrival} " f"(confidence: {arrival_candidates[0][1]:.3f})"
            )

        # If we couldn't classify both departure and arrival, raise error
        if departure is None or arrival is None:
            logger.warning(
                f"Classification failed for text: '{text}'. "
                f"Found {len(departure_candidates)} departure(s) and "
                f"{len(arrival_candidates)} arrival(s)"
            )
            raise ClassificationError(text, locations)

        return (departure, arrival)

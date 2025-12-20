"""
Custom exceptions for the Trip Extraction system.

This module defines domain-specific exceptions for better error handling
and debugging.
"""


class TripExtractionError(Exception):
    """Base exception for all trip extraction errors."""

    pass


class ModelNotFoundError(TripExtractionError):
    """Raised when a required model file cannot be found."""

    def __init__(self, model_path: str, message: str = ""):
        self.model_path = model_path
        default_message = (
            f"Model not found at '{model_path}'. "
            f"Please train the model first by running: python scripts/train.py"
        )
        super().__init__(message or default_message)


class ModelLoadError(TripExtractionError):
    """Raised when a model fails to load."""

    def __init__(self, model_name: str, original_error: Exception):
        self.model_name = model_name
        self.original_error = original_error
        super().__init__(f"Failed to load model '{model_name}': {str(original_error)}")


class InsufficientLocationsError(TripExtractionError):
    """Raised when not enough locations are detected in the input text."""

    def __init__(self, found_count: int, required_count: int = 2):
        self.found_count = found_count
        self.required_count = required_count
        super().__init__(f"Need at least {required_count} locations, but only found {found_count}")


class InvalidInputError(TripExtractionError):
    """Raised when input validation fails."""

    def __init__(self, field: str, value, reason: str = ""):
        self.field = field
        self.value = value
        message = f"Invalid input for '{field}'"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class ClassificationError(TripExtractionError):
    """Raised when location classification fails."""

    def __init__(self, text: str, locations: list):
        self.text = text
        self.locations = locations
        super().__init__(
            f"Failed to classify locations in text. "
            f"Consider adding this pattern to the training dataset: '{text}'"
        )


class TokenizationError(TripExtractionError):
    """Raised when tokenization fails."""

    def __init__(self, text: str, original_error: Exception):
        self.text = text
        self.original_error = original_error
        super().__init__(f"Tokenization failed for text: {str(original_error)}")

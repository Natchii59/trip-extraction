"""
Configuration module for the Trip Extraction system.

This module provides centralized configuration management with absolute paths
and environment-based settings.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Paths:
    """Centralized path configuration using absolute paths."""

    # Project root is 3 levels up from this file: src/trip/config.py
    PROJECT_ROOT: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)

    @property
    def models_dir(self) -> Path:
        """Directory containing trained models."""
        return self.PROJECT_ROOT / "models"

    @property
    def data_dir(self) -> Path:
        """Directory containing datasets."""
        return self.PROJECT_ROOT / "data"

    @property
    def logs_dir(self) -> Path:
        """Directory for log files."""
        return self.PROJECT_ROOT / "logs"

    @property
    def departure_arrival_model(self) -> Path:
        """Path to the departure-arrival classifier model."""
        return self.models_dir / "departure_arrival_classifier"

    @property
    def training_dataset(self) -> Path:
        """Path to the training dataset."""
        return self.data_dir / "training_dataset.json"

    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Configuration for ML models."""

    # NER Model
    ner_model_name: str = "Jean-Baptiste/camembert-ner"

    # Classifier Model
    classifier_base_model: str = "camembert-base"
    max_sequence_length: int = 128

    # Inference settings
    confidence_threshold: float = 0.5
    device: Optional[str] = None  # None = auto-detect (cuda if available, else cpu)


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Dataset settings
    test_size: float = 0.2
    random_state: int = 42

    # Training hyperparameters
    num_epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 50
    gradient_accumulation_steps: int = 4

    # Model configuration
    max_length: int = 128
    num_labels: int = 2  # departure (0) or arrival (1)

    # Special tokens
    special_tokens: list[str] = field(default_factory=lambda: ["[LOC]", "[/LOC]"])


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    log_to_file: bool = False

    @property
    def log_file(self) -> Optional[Path]:
        """Path to log file if logging to file is enabled."""
        if self.log_to_file:
            paths = Paths()
            paths.ensure_directories()
            return paths.logs_dir / "trip_extraction.log"
        return None


@dataclass
class Config:
    """Main configuration class that aggregates all settings."""

    paths: Paths = field(default_factory=Paths)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self):
        """Ensure required directories exist."""
        self.paths.ensure_directories()


# Global configuration instance
config = Config()


def get_config() -> Config:
    """
    Get the global configuration instance.

    Returns:
        The global Config object.
    """
    return config


def update_config(**kwargs):
    """
    Update configuration values.

    Args:
        **kwargs: Configuration key-value pairs to update.

    Example:
        >>> update_config(logging={'level': 'DEBUG'})
    """
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

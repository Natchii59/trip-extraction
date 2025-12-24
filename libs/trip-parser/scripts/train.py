#!/usr/bin/env python3
"""
Training script for the departure-arrival classifier.

This script fine-tunes a CamemBERT model to classify location roles
in travel contexts (departure vs arrival). The model learns to identify
which location is the origin and which is the destination from French
travel sentences.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    CamembertForSequenceClassification,
    CamembertTokenizer,
    Trainer,
    TrainingArguments,
)

from trip_parser.config import get_config
from trip_parser.exceptions import InvalidInputError

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TripDataset(Dataset):
    """
    Dataset for training the departure-arrival classifier.

    Creates training examples by generating sentences with location markers
    and labels (0=departure, 1=arrival).
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: CamembertTokenizer,
        max_length: int = 128,
    ):
        """
        Initialize the dataset.

        Args:
            texts: List of input texts with [LOC] markers.
            labels: List of labels (0=departure, 1=arrival).
            tokenizer: Tokenizer for encoding texts.
            max_length: Maximum sequence length.
        """
        if len(texts) != len(labels):
            raise InvalidInputError(
                "texts/labels",
                f"lengths {len(texts)}/{len(labels)}",
                "Texts and labels must have the same length",
            )

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Get the number of examples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single training example.

        Args:
            idx: Index of the example.

        Returns:
            Dictionary with input_ids, attention_mask, and labels tensors.
        """
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def load_dataset(dataset_path: Path) -> list[dict]:
    """
    Load the training dataset from JSON file.

    Args:
        dataset_path: Path to the JSON dataset file.

    Returns:
        List of training examples.

    Raises:
        FileNotFoundError: If dataset file doesn't exist.
        json.JSONDecodeError: If dataset file is not valid JSON.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. " f"Please ensure the training data exists."
        )

    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} examples from {dataset_path}")
    return data


def create_training_examples(data: list[dict]) -> tuple[list[str], list[int]]:
    """
    Load training examples directly from the dataset.

    Dataset format: {"text": "phrase avec <LOC> ville </LOC>", "label": 0 ou 1}

    Args:
        data: List of dicts with keys: text (with <LOC> markers), label (0=departure, 1=arrival)

    Returns:
        Tuple of (texts, labels) where texts already contain <LOC> markers
    """
    texts = []
    labels = []

    for example in data:
        if "text" not in example or "label" not in example:
            logger.warning(f"Skipping invalid example: {example}")
            continue

        texts.append(example["text"])
        labels.append(example["label"])

    logger.info(f"Loaded {len(texts)} training examples")
    return texts, labels


def compute_metrics(eval_pred) -> dict[str, float]:
    """
    Compute accuracy for evaluation.

    Args:
        eval_pred: Tuple of (predictions, labels) from the trainer.

    Returns:
        Dictionary with computed metrics.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


def train_model():
    """
    Train the departure-arrival classifier.

    Returns:
        Tuple of (trainer, evaluation_results).
    """
    logger.info("Starting model training...")

    # Get configuration
    config = get_config()
    training_config = config.training

    logger.info("Configuration:")
    logger.info(f"  - Model: {config.model.classifier_base_model}")
    logger.info(f"  - Epochs: {training_config.num_epochs}")
    logger.info(f"  - Batch size: {training_config.batch_size}")
    logger.info(f"  - Learning rate: {training_config.learning_rate}")

    # Load dataset
    dataset_path = config.paths.training_dataset
    data = load_dataset(dataset_path)

    # Load training examples (already formatted with [LOC])
    texts, labels = create_training_examples(data)

    # Split into train and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=training_config.test_size,
        random_state=training_config.random_state,
        stratify=labels,
    )

    logger.info(f"Train set: {len(train_texts)} examples")
    logger.info(f"Validation set: {len(val_texts)} examples")

    # Load tokenizer and model
    logger.info(f"Loading model: {config.model.classifier_base_model}")
    tokenizer = CamembertTokenizer.from_pretrained(config.model.classifier_base_model)

    # Add special tokens for location markers
    special_tokens_dict = {"additional_special_tokens": training_config.special_tokens}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    logger.info(f"Added {num_added_tokens} special tokens: {training_config.special_tokens}")

    model = CamembertForSequenceClassification.from_pretrained(
        config.model.classifier_base_model,
        num_labels=training_config.num_labels,  # departure (0) or arrival (1)
    )

    # Resize model embeddings to accommodate new tokens
    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Resized model embeddings to {len(tokenizer)} tokens")

    # Create datasets
    train_dataset = TripDataset(train_texts, train_labels, tokenizer, training_config.max_length)
    val_dataset = TripDataset(val_texts, val_labels, tokenizer, training_config.max_length)

    # Ensure output directory exists
    output_dir = config.paths.departure_arrival_model
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_config.num_epochs,
        per_device_train_batch_size=training_config.batch_size,
        per_device_eval_batch_size=training_config.batch_size * 2,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        warmup_steps=training_config.warmup_steps,
        logging_dir=str(output_dir / "logs"),
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_only_model=True,
        report_to="none",
        fp16=False,
        dataloader_num_workers=0,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        disable_tqdm=False,
        use_cpu=False,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    logger.info("Starting training...")
    trainer.train()

    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    eval_results = trainer.evaluate()
    logger.info(f"Validation results: {eval_results}")

    # Save the final model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    logger.info("Training completed successfully!")

    return trainer, eval_results


def main():
    """Main training function."""
    try:
        config = get_config()

        # Ensure directories exist
        config.paths.ensure_directories()

        # Train the model
        trainer, results = train_model()

        # Print summary
        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)
        print(f"Model saved to: {config.paths.departure_arrival_model}")
        print(f"Final validation accuracy: {results['eval_accuracy']:.4f}")
        print("=" * 60)

        return 0

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

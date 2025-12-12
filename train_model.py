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
import torch
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from transformers import (
    CamembertTokenizer,
    CamembertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    model_name: str = "camembert-base"
    output_dir: str = "./models/departure_arrival_classifier"
    dataset_path: str = "./data/training_dataset.json"
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


class TripDataset(Dataset):
    """
    Dataset for training the departure-arrival classifier.

    Creates training examples by generating sentences with location markers
    and labels (0=departure, 1=arrival).
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
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
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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


def load_dataset(dataset_path: str) -> List[Dict]:
    """Load the training dataset from JSON file."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} examples from {dataset_path}")
    return data


def create_training_examples(data: List[Dict]) -> Tuple[List[str], List[int]]:
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
        texts.append(example["text"])
        labels.append(example["label"])

    logger.info(f"Loaded {len(texts)} training examples")
    return texts, labels


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute accuracy for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


def train_model(config: TrainingConfig):
    """
    Train the departure-arrival classifier.

    Args:
        config: Training configuration.
    """
    logger.info("Starting model training...")
    logger.info(f"Configuration: {config}")

    # Load dataset
    data = load_dataset(config.dataset_path)

    # Load training examples (déjà formatés avec <LOC>)
    texts, labels = create_training_examples(data)

    # Split into train and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=config.test_size, random_state=config.random_state, stratify=labels
    )

    logger.info(f"Train set: {len(train_texts)} examples")
    logger.info(f"Validation set: {len(val_texts)} examples")

    # Load tokenizer and model
    logger.info(f"Loading model: {config.model_name}")
    tokenizer = CamembertTokenizer.from_pretrained(config.model_name)

    # Add special tokens for location markers
    special_tokens_dict = {"additional_special_tokens": ["[LOC]", "[/LOC]"]}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    logger.info(f"Added {num_added_tokens} special tokens: [LOC], [/LOC]")

    model = CamembertForSequenceClassification.from_pretrained(
        config.model_name, num_labels=2  # departure (0) or arrival (1)
    )

    # Resize model embeddings to accommodate new tokens
    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Resized model embeddings to {len(tokenizer)} tokens")

    # Create datasets
    train_dataset = TripDataset(train_texts, train_labels, tokenizer, config.max_length)
    val_dataset = TripDataset(val_texts, val_labels, tokenizer, config.max_length)

    # Define training arguments - Ultra rapide
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        logging_dir=f"{config.output_dir}/logs",
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_only_model=True,
        report_to="none",
        fp16=False,
        dataloader_num_workers=0,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
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
    logger.info(f"Saving model to {config.output_dir}")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    logger.info("Training completed successfully!")

    return trainer, eval_results


def main():
    """Main training function."""
    config = TrainingConfig()

    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Train the model
    trainer, results = train_model(config)

    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Model saved to: {config.output_dir}")
    print(f"Final validation accuracy: {results['eval_accuracy']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

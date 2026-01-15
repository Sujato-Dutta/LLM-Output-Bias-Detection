"""
Configuration management for the bias detection project.

This module centralizes all configuration settings including
model paths, hyperparameters, and label mappings.
"""

import os
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv


# Load environment variables
load_dotenv()


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


# Model directories
MODEL_DIRS = {
    "logistic_regression": PROJECT_ROOT / "models" / "logistic_regression",
    "frozen_minilm": PROJECT_ROOT / "models" / "frozen_minilm",
    "lora_minilm": PROJECT_ROOT / "models" / "lora_minilm",
}


# Label mappings
LABEL_NAMES = {
    0: "anti-stereotype",
    1: "stereotype",
    2: "unrelated",
}

LABEL_TO_ID = {v: k for k, v in LABEL_NAMES.items()}


# Model configurations
MODEL_CONFIG = {
    "base_model": "sentence-transformers/all-MiniLM-L6-v2",
    "hidden_size": 384,
    "num_labels": 3,
    "max_length": 128,
}


# Training configurations (for reference - models are pre-trained)
TRAINING_CONFIG = {
    "lora": {
        "r": 8,
        "alpha": 32,
        "dropout": 0.1,
        "target_modules": ["query", "value"],
    },
    "split": {
        "test_size_first": 0.3,
        "test_size_second": 0.5,
        "random_state": 42,
    },
}


def get_hf_token() -> str:
    """
    Get the HuggingFace token from environment.

    Returns:
        HuggingFace token string.

    Raises:
        ValueError: If token is not set.
    """
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError(
            "HF_TOKEN not found. Please set it in your .env file. "
            "See .env.example for reference."
        )
    return token


def get_model_path(model_name: str) -> Path:
    """
    Get the path to a model directory.

    Args:
        model_name: Name of the model ('logistic_regression', 'frozen_minilm', 'lora_minilm').

    Returns:
        Path to the model directory.

    Raises:
        ValueError: If model name is not recognized.
    """
    if model_name not in MODEL_DIRS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(MODEL_DIRS.keys())}"
        )
    return MODEL_DIRS[model_name]


if __name__ == "__main__":
    print("Project Configuration")
    print("=" * 40)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"\nModel Directories:")
    for name, path in MODEL_DIRS.items():
        exists = "✓" if path.exists() else "✗"
        print(f"  {name}: {path} [{exists}]")

"""
Evaluation metrics for bias detection models.

This module provides functions to compute classification metrics
including accuracy, precision, recall, and F1 score.
"""

from typing import List, Dict, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from ..preprocess import LABEL_NAMES


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    average: str = "macro",
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: List of true labels.
        y_pred: List of predicted labels.
        average: Averaging strategy for multi-class metrics.
                 'macro' averages equally across classes (recommended for balanced datasets).

    Returns:
        Dictionary with accuracy, precision, recall, and F1 scores.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def get_classification_report(
    y_true: List[int],
    y_pred: List[int],
    output_dict: bool = False,
) -> str | Dict[str, Any]:
    """
    Generate a detailed classification report.

    Args:
        y_true: List of true labels.
        y_pred: List of predicted labels.
        output_dict: If True, return as dictionary instead of string.

    Returns:
        Classification report as string or dictionary.
    """
    target_names = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES.keys())]
    return classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=output_dict,
        zero_division=0,
    )


def get_confusion_matrix(y_true: List[int], y_pred: List[int]) -> np.ndarray:
    """
    Compute the confusion matrix.

    Args:
        y_true: List of true labels.
        y_pred: List of predicted labels.

    Returns:
        Confusion matrix as numpy array.
    """
    return confusion_matrix(y_true, y_pred)


def format_metrics_table(metrics_dict: Dict[str, Dict[str, float]]) -> str:
    """
    Format multiple models' metrics as a comparison table.

    Args:
        metrics_dict: Dictionary mapping model names to their metrics.
                      Example: {"LogReg": {"accuracy": 0.5, ...}, "LoRA": {...}}

    Returns:
        Formatted string table for display.
    """
    # Header
    lines = [
        "| Model | Accuracy | Precision | Recall | F1 Score |",
        "|-------|----------|-----------|--------|----------|",
    ]

    # Rows
    for model_name, metrics in metrics_dict.items():
        lines.append(
            f"| {model_name} | "
            f"{metrics.get('accuracy', 0):.4f} | "
            f"{metrics.get('precision', 0):.4f} | "
            f"{metrics.get('recall', 0):.4f} | "
            f"{metrics.get('f1', 0):.4f} |"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test
    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 0, 2, 1, 1, 1]

    metrics = compute_metrics(y_true, y_pred)
    print("Metrics:", metrics)

    print("\nClassification Report:")
    print(get_classification_report(y_true, y_pred))

    print("\nConfusion Matrix:")
    print(get_confusion_matrix(y_true, y_pred))

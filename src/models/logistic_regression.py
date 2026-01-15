"""
Logistic Regression model for bias detection baseline.

This module implements a classical TF-IDF + Logistic Regression baseline
for comparison with transformer-based models.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ..preprocess import get_train_val_test_split, LABEL_NAMES


# Default model save path
MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "logistic_regression"


class LogisticRegressionClassifier:
    """
    TF-IDF + Logistic Regression classifier for bias detection.
    
    This serves as a classical ML baseline for comparison with
    transformer-based approaches.
    """

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        max_iter: int = 1000,
        class_weight: str = "balanced",
    ):
        """
        Initialize the classifier.

        Args:
            max_features: Maximum number of TF-IDF features.
            ngram_range: Range of n-grams for TF-IDF.
            max_iter: Maximum iterations for logistic regression.
            class_weight: Class weighting strategy.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
        )
        self.model = LogisticRegression(
            max_iter=max_iter,
            class_weight=class_weight,
            solver="lbfgs",
            random_state=42,
        )
        self._is_fitted = False

    def fit(self, texts: List[str], labels: List[int]) -> "LogisticRegressionClassifier":
        """
        Train the classifier on the provided data.

        Args:
            texts: List of preprocessed text samples.
            labels: List of corresponding integer labels.

        Returns:
            Self for method chaining.
        """
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self._is_fitted = True
        return self

    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict labels for the given texts.

        Args:
            texts: List of preprocessed text samples.

        Returns:
            Array of predicted labels.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict class probabilities for the given texts.

        Args:
            texts: List of preprocessed text samples.

        Returns:
            Array of shape (n_samples, n_classes) with probabilities.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)

    def evaluate(self, texts: List[str], labels: List[int]) -> Dict[str, float]:
        """
        Evaluate the model on the given data.

        Args:
            texts: List of preprocessed text samples.
            labels: List of true labels.

        Returns:
            Dictionary with accuracy, precision, recall, and F1 scores.
        """
        predictions = self.predict(texts)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions, average="macro"),
            "recall": recall_score(labels, predictions, average="macro"),
            "f1": f1_score(labels, predictions, average="macro"),
        }

    def save(self, model_dir: Optional[Path] = None) -> None:
        """
        Save the model and vectorizer to disk.

        Args:
            model_dir: Directory to save to. Defaults to models/logistic_regression/
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before saving.")

        save_dir = model_dir or MODEL_DIR
        save_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, save_dir / "model.joblib")
        joblib.dump(self.vectorizer, save_dir / "vectorizer.joblib")

    @classmethod
    def load(cls, model_dir: Optional[Path] = None) -> "LogisticRegressionClassifier":
        """
        Load a saved model from disk.

        Args:
            model_dir: Directory to load from. Defaults to models/logistic_regression/

        Returns:
            Loaded classifier instance.
        """
        load_dir = model_dir or MODEL_DIR

        instance = cls()
        instance.model = joblib.load(load_dir / "model.joblib")
        instance.vectorizer = joblib.load(load_dir / "vectorizer.joblib")
        instance._is_fitted = True

        return instance


def train_and_save(model_dir: Optional[Path] = None) -> Dict[str, float]:
    """
    Train the Logistic Regression model and save it.

    Uses the same train/test split as the LoRA notebook to ensure
    fair comparison.

    Args:
        model_dir: Optional directory to save the model to.

    Returns:
        Dictionary with evaluation metrics on the test set.
    """
    print("Loading and preprocessing data...")
    splits = get_train_val_test_split()

    train_texts, train_labels = splits["train"]
    test_texts, test_labels = splits["test"]

    print(f"Training on {len(train_texts)} samples...")
    classifier = LogisticRegressionClassifier()
    classifier.fit(train_texts, train_labels)

    print("Evaluating on test set...")
    metrics = classifier.evaluate(test_texts, test_labels)

    print("\n--- Logistic Regression Test Results ---")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision']:.4f}")
    print(f"Recall (macro): {metrics['recall']:.4f}")
    print(f"Macro F1 Score: {metrics['f1']:.4f}")

    print("\nSaving model...")
    classifier.save(model_dir)
    print(f"Model saved to {model_dir or MODEL_DIR}")

    return metrics


if __name__ == "__main__":
    train_and_save()

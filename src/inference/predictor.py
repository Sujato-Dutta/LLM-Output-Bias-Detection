"""
Bias detection predictor for production inference.

This module provides a clean, unified interface for bias detection
using the LoRA fine-tuned MiniLM model. This is the primary interface
for the Streamlit demo application.
"""

from pathlib import Path
from typing import Dict, Optional

from ..models.lora_minilm import LoRAMiniLMPredictor
from ..preprocess import LABEL_NAMES


class BiasPredictor:
    """
    Production-ready bias predictor using LoRA MiniLM.

    This class provides a simple, clean interface for detecting
    stereotypical bias in context-sentence pairs.

    Usage:
        predictor = BiasPredictor()
        result = predictor.predict(
            context="The doctor examined the patient.",
            sentence="She was very thorough."
        )
        print(result)
        # {'label': 'anti-stereotype', 'confidence': 0.85, 'label_id': 0}
    """

    def __init__(self, model_dir: Optional[Path] = None):
        """
        Initialize the predictor.

        Args:
            model_dir: Optional path to the model directory.
                       Defaults to models/lora_minilm/
        """
        self._predictor = LoRAMiniLMPredictor(model_dir=model_dir)

    def predict(self, context: str, sentence: str) -> Dict[str, any]:
        """
        Predict the bias label for a context-sentence pair.

        Args:
            context: The context sentence establishing a scenario.
            sentence: The candidate sentence to classify.

        Returns:
            Dictionary containing:
                - label: Human-readable label ('stereotype', 'anti-stereotype', 'unrelated')
                - confidence: Prediction confidence as float (0.0 to 1.0)
                - label_id: Integer label ID (0, 1, or 2)

        Example:
            >>> predictor = BiasPredictor()
            >>> result = predictor.predict(
            ...     context="My neighbor is a software engineer.",
            ...     sentence="He spends all his time coding."
            ... )
            >>> print(result['label'])
            'stereotype'
        """
        return self._predictor.predict_single(context, sentence)

    def get_label_description(self, label: str) -> str:
        """
        Get a description for a label.

        Args:
            label: The label string.

        Returns:
            Human-readable description of what the label means.
        """
        descriptions = {
            "stereotype": "The sentence reinforces a common stereotype about the target group.",
            "anti-stereotype": "The sentence challenges or contradicts a common stereotype.",
            "unrelated": "The sentence is not meaningfully related to the context.",
        }
        return descriptions.get(label, "Unknown label.")

    @staticmethod
    def get_available_labels() -> Dict[int, str]:
        """
        Get all available label mappings.

        Returns:
            Dictionary mapping label IDs to label names.
        """
        return LABEL_NAMES.copy()


if __name__ == "__main__":
    # Demo usage
    predictor = BiasPredictor()

    print("=== Bias Detection Demo ===\n")

    examples = [
        ("The software engineer was debugging code.", "He stayed up all night fixing bugs."),
        ("The nurse came to check on the patient.", "She was very caring and attentive."),
        ("The CEO addressed the shareholders.", "The weather was nice that day."),
    ]

    for context, sentence in examples:
        result = predictor.predict(context, sentence)
        print(f"Context: {context}")
        print(f"Sentence: {sentence}")
        print(f"Prediction: {result['label']} (confidence: {result['confidence']:.2%})")
        print(f"Explanation: {predictor.get_label_description(result['label'])}")
        print()

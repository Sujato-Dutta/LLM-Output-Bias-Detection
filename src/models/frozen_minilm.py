"""
Frozen MiniLM model for bias detection.

This module provides inference functionality for the pre-trained Frozen MiniLM
model. The encoder is frozen and only a linear classification head was trained.

NOTE: This model is for evaluation comparison only. The Streamlit app uses
the LoRA model exclusively.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from ..preprocess import LABEL_NAMES


# Default model path
MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "frozen_minilm"


class FrozenMiniLMClassifier(nn.Module):
    """
    Frozen MiniLM encoder with trainable classification head.

    Architecture:
        - Encoder: sentence-transformers/all-MiniLM-L6-v2 (frozen)
        - Pooling: CLS token
        - Classifier: Linear(384, 3)
    """

    def __init__(self, base_model: nn.Module, hidden_size: int = 384, num_labels: int = 3):
        """
        Initialize the classifier.

        Args:
            base_model: Pre-trained transformer encoder.
            hidden_size: Size of the encoder hidden states.
            num_labels: Number of output classes.
        """
        super().__init__()
        self.encoder = base_model

        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs.
            attention_mask: Attention mask.
            token_type_ids: Token type IDs (optional, ignored).
            labels: Labels for loss computation (optional).

        Returns:
            Tuple of (loss, logits). Loss is None if labels not provided.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # CLS token pooling
        pooled = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return loss, logits


class FrozenMiniLMPredictor:
    """
    Predictor class for Frozen MiniLM inference.
    """

    def __init__(self, model_dir: Optional[Path] = None, device: Optional[str] = None):
        """
        Initialize the predictor.

        Args:
            model_dir: Directory containing the saved model. Defaults to models/frozen_minilm/
            device: Device to use for inference.
        """
        self.model_dir = model_dir or MODEL_DIR
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))

        # Load model
        self._load_model()

    def _load_model(self) -> None:
        """Load the model from disk."""
        # Load base encoder
        base_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

        # Create classifier
        self.model = FrozenMiniLMClassifier(base_model)

        # Load saved weights (includes both encoder and classifier)
        state_dict = torch.load(
            self.model_dir / "model.pt",
            map_location=self.device,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, texts: List[str]) -> List[Dict[str, any]]:
        """
        Predict labels for the given texts.

        Args:
            texts: List of preprocessed text samples.

        Returns:
            List of dictionaries with 'label', 'label_id', and 'confidence'.
        """
        # Tokenize
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )

        # Move to device
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        # Predict
        with torch.no_grad():
            _, logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            confidences = probs.max(dim=-1).values

        # Format results
        results = []
        for pred, conf in zip(predictions.cpu().numpy(), confidences.cpu().numpy()):
            results.append({
                "label": LABEL_NAMES[pred],
                "label_id": int(pred),
                "confidence": float(conf),
            })

        return results

    def predict_single(self, context: str, sentence: str) -> Dict[str, any]:
        """
        Predict for a single context-sentence pair.

        Args:
            context: The context sentence.
            sentence: The candidate sentence.

        Returns:
            Dictionary with 'label', 'label_id', and 'confidence'.
        """
        text = f"{context} <sep> {sentence}"
        return self.predict([text])[0]


if __name__ == "__main__":
    # Quick test
    predictor = FrozenMiniLMPredictor()
    result = predictor.predict_single(
        "The doctor examined the patient.",
        "She was very thorough.",
    )
    print(f"Prediction: {result}")

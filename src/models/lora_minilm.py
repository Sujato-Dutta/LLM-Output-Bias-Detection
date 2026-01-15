"""
LoRA MiniLM model for bias detection.

This module provides inference functionality for the LoRA fine-tuned MiniLM model.
This is the main model used in the Streamlit demo application.

Architecture:
    - Base: sentence-transformers/all-MiniLM-L6-v2
    - Adapter: LoRA (r=8, alpha=32, dropout=0.1, target=query,value)
    - Pooling: Mean pooling
    - Classifier: Linear(384, 3)
"""

from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

from ..preprocess import LABEL_NAMES


# Default model paths
MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "lora_minilm"


class LoRAMiniLMClassifier(nn.Module):
    """
    LoRA-adapted MiniLM encoder with classification head.

    Architecture:
        - Encoder: sentence-transformers/all-MiniLM-L6-v2 + LoRA adapters
        - Pooling: Mean pooling over last hidden states
        - Classifier: Linear(384, 3)
    """

    def __init__(self, lora_encoder: nn.Module, hidden_size: int = 384, num_labels: int = 3):
        """
        Initialize the classifier.

        Args:
            lora_encoder: Pre-trained encoder with LoRA adapters applied.
            hidden_size: Size of the encoder hidden states.
            num_labels: Number of output classes.
        """
        super().__init__()
        self.encoder = lora_encoder
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs.
            attention_mask: Attention mask.

        Returns:
            Logits tensor of shape (batch_size, num_labels).
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Mean pooling over last hidden states
        embeddings = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(embeddings)

        return logits


class LoRAMiniLMPredictor:
    """
    Predictor class for LoRA MiniLM inference.

    This is the main predictor used in the Streamlit demo.
    """

    def __init__(self, model_dir: Optional[Path] = None, device: Optional[str] = None):
        """
        Initialize the predictor.

        Args:
            model_dir: Directory containing the saved model. Defaults to models/lora_minilm/
            device: Device to use for inference.
        """
        self.model_dir = model_dir or MODEL_DIR
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Base model name
        self.base_model_name = "sentence-transformers/all-MiniLM-L6-v2"

        # Load tokenizer from the base model
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

        # Load model
        self._load_model()

    def _load_model(self) -> None:
        """Load the model from disk."""
        # Load base encoder
        base_model = AutoModel.from_pretrained(self.base_model_name)

        # Apply LoRA adapter
        lora_model = PeftModel.from_pretrained(
            base_model,
            str(self.model_dir / "adapter"),
        )

        # Create classifier
        self.model = LoRAMiniLMClassifier(lora_model)

        # Load classifier head weights
        classifier_state = torch.load(
            self.model_dir / "classifier.pt",
            map_location=self.device,
        )
        self.model.classifier.load_state_dict(classifier_state)

        self.model.to(self.device)
        self.model.eval()

    def predict(self, texts: List[str]) -> List[Dict[str, any]]:
        """
        Predict labels for the given texts.

        Args:
            texts: List of preprocessed text samples (format: "context <sep> sentence").

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
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
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
    predictor = LoRAMiniLMPredictor()
    result = predictor.predict_single(
        "The doctor examined the patient.",
        "She was very thorough.",
    )
    print(f"Prediction: {result}")

"""
Data preprocessing module for StereoSet bias detection.

This module handles preprocessing the raw StereoSet data into the format
required for model training and inference. It flattens the nested structure
and creates (text, label) pairs.

CRITICAL: The train/val/test split uses the EXACT same parameters as the
LoRA MiniLM training notebook to ensure the test set is identical, preventing
inconsistencies during evaluation comparison.

Split parameters (matching LoRA notebook):
    - random_state=42
    - First split: test_size=0.3 (70% train, 30% temp)
    - Second split: test_size=0.5 (15% val, 15% test from temp)
"""

from typing import List, Tuple, Dict, Optional

from sklearn.model_selection import train_test_split

from .load_data import load_stereoset


# Label mapping matching the notebook's gold_label encoding
LABEL_NAMES = {
    0: "anti-stereotype",
    1: "stereotype",
    2: "unrelated",
}

# Reverse mapping for inference
LABEL_TO_ID = {v: k for k, v in LABEL_NAMES.items()}


def prepare_dataset(
    dataset: Optional[dict] = None,
) -> Tuple[List[str], List[int], List[str]]:
    """
    Prepare the StereoSet dataset into flattened (text, label) pairs.

    Each example in StereoSet contains a context and multiple candidate
    sentences. This function flattens the structure so each (context, sentence)
    pair becomes an independent sample.

    Format: "{context} <sep> {sentence}"

    Args:
        dataset: Optional pre-loaded dataset. If None, will load from HuggingFace.

    Returns:
        Tuple of (texts, labels, bias_types) where:
            - texts: List of formatted "context <sep> sentence" strings
            - labels: List of integer labels (0=anti-stereotype, 1=stereotype, 2=unrelated)
            - bias_types: List of bias type strings (race, gender, profession, religion)
    """
    if dataset is None:
        dataset = load_stereoset()

    texts = []
    labels = []
    bias_types = []

    for example in dataset:
        context = example["context"]
        bias_type = example["bias_type"]

        sentences = example["sentences"]["sentence"]
        gold_labels = example["sentences"]["gold_label"]

        for sentence, label in zip(sentences, gold_labels):
            # Format matching the training notebooks
            combined = f"{context} <sep> {sentence}"
            texts.append(combined)
            labels.append(label)
            bias_types.append(bias_type)

    return texts, labels, bias_types


def get_train_val_test_split(
    texts: Optional[List[str]] = None,
    labels: Optional[List[int]] = None,
    random_state: int = 42,
) -> Dict[str, Tuple[List[str], List[int]]]:
    """
    Split data into train/val/test sets matching the LoRA notebook exactly.

    CRITICAL: Uses the exact same random_state and split ratios as the LoRA
    MiniLM training notebook to ensure identical test set. This prevents
    data leakage when comparing models.

    Split ratios: 70% train, 15% val, 15% test

    Args:
        texts: List of preprocessed text samples. If None, will prepare from dataset.
        labels: List of corresponding labels. If None, will prepare from dataset.
        random_state: Random seed for reproducibility. MUST be 42 to match notebooks.

    Returns:
        Dictionary with keys 'train', 'val', 'test', each containing (texts, labels) tuple.
    """
    if texts is None or labels is None:
        texts, labels, _ = prepare_dataset()

    # First split: 70% train, 30% temp (matching LoRA notebook)
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts,
        labels,
        test_size=0.3,
        random_state=random_state,
        stratify=labels,
    )

    # Second split: 50% of temp = 15% val, 15% test (matching LoRA notebook)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=0.5,
        random_state=random_state,
        stratify=temp_labels,
    )

    return {
        "train": (train_texts, train_labels),
        "val": (val_texts, val_labels),
        "test": (test_texts, test_labels),
    }


def format_input(context: str, sentence: str) -> str:
    """
    Format a context-sentence pair for model input.

    Args:
        context: The context sentence.
        sentence: The candidate sentence to classify.

    Returns:
        Formatted string: "{context} <sep> {sentence}"
    """
    return f"{context} <sep> {sentence}"


if __name__ == "__main__":
    # Quick test
    texts, labels, bias_types = prepare_dataset()
    print(f"Total samples: {len(texts)}")
    print(f"Label distribution: {dict(zip(*[[l for l in set(labels)], [labels.count(l) for l in set(labels)]]))}")

    splits = get_train_val_test_split(texts, labels)
    print(f"\nSplit sizes (matching LoRA notebook):")
    print(f"  Train: {len(splits['train'][0])}")
    print(f"  Val: {len(splits['val'][0])}")
    print(f"  Test: {len(splits['test'][0])}")

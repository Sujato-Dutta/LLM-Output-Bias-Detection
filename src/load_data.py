"""
Data loading module for StereoSet dataset.

This module handles loading the StereoSet dataset from HuggingFace,
using an authentication token from the environment for access.
"""

import os
from typing import Optional

from datasets import load_dataset
from dotenv import load_dotenv


def load_stereoset(
    subset: str = "intersentence",
    split: str = "validation",
    hf_token: Optional[str] = None,
) -> dict:
    """
    Load the StereoSet dataset from HuggingFace.

    Args:
        subset: Dataset subset to load ("intersentence" or "intrasentence").
                Default is "intersentence" as used in training.
        split: Dataset split to load. StereoSet only has "validation" split.
        hf_token: Optional HuggingFace token. If not provided, will attempt
                  to load from HF_TOKEN environment variable.

    Returns:
        HuggingFace dataset object containing the StereoSet data.

    Raises:
        ValueError: If HF_TOKEN is not found and not provided.
    """
    # Load environment variables
    load_dotenv()

    # Get token from parameter or environment
    token = hf_token or os.getenv("HF_TOKEN")

    if token is None:
        raise ValueError(
            "HuggingFace token not found. Please set HF_TOKEN in your .env file "
            "or pass it directly to this function. See .env.example for reference."
        )

    # Load the dataset
    dataset = load_dataset(
        "McGill-NLP/stereoset",
        subset,
        split=split,
        token=token,
    )

    return dataset


if __name__ == "__main__":
    # Quick test
    ds = load_stereoset()
    print(f"Loaded {len(ds)} examples from StereoSet")
    print(f"Features: {ds.features}")

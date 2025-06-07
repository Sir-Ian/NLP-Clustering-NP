"""Utility functions for estimating token usage."""

import logging

from utils import load_data, setup_logging


def estimate_tokens(path: str, column: str = "Processed_Concern", sample_size: int = 100, avg_token_length: int = 4) -> float:
    """Estimate token count for a sample of text values."""
    setup_logging("token_est.log")
    df = load_data(path)
    if df is None:
        return 0.0

    df_sample = df.sample(n=sample_size)
    total = sum(len(text) / avg_token_length for text in df_sample[column])
    logging.info("Estimated total tokens for %d samples: %.0f", sample_size, total)
    return total


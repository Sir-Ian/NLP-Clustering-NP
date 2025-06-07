"""Utilities for matching FAQ questions to chat responses."""

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils import load_data, save_data, setup_logging


def answer_faq(preprocessed_path: str, faq_path: str, output_path: str, model_name: str = "all-mpnet-base-v2", threshold: float = 0.5) -> None:
    """Match FAQ questions to existing chat responses using embeddings."""
    setup_logging("faq_processing.log")
    model = SentenceTransformer(model_name)

    pre_df = load_data(preprocessed_path)
    faq_df = load_data(faq_path)
    if pre_df is None or faq_df is None:
        logging.error("Failed to load input data")
        return

    concerns = pre_df["Concern"].fillna("").tolist()
    questions = faq_df["Question"].fillna("").tolist()

    concerns_emb = model.encode(concerns, show_progress_bar=True)
    questions_emb = model.encode(questions, show_progress_bar=True)
    similarities = cosine_similarity(questions_emb, concerns_emb)

    for i, _ in enumerate(questions):
        best_idx = int(np.argmax(similarities[i]))
        if similarities[i][best_idx] > threshold:
            faq_df.loc[i, "Answer"] = pre_df.iloc[best_idx]["Response"]
        else:
            faq_df.loc[i, "Answer"] = "REVIEW: No satisfying answer found"

    save_data(faq_df, output_path)
    logging.info("FAQ answers saved to %s", output_path)


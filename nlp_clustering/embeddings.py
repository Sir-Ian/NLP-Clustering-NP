"""Embedding generation utilities."""

import json
import logging

import numpy as np
import openai
from utils import load_data, save_data, setup_logging, get_env_variable


OPENAI_API_KEY = get_env_variable("OPENAI_API_KEY")


def generate_openai_embeddings(df, text_column: str, model: str = "text-similarity-babbage-001") -> None:
    """Populate the DataFrame with OpenAI embeddings for the given text column."""
    if not OPENAI_API_KEY:
        logging.error("OPENAI_API_KEY not set")
        return
    openai.api_key = OPENAI_API_KEY

    embeddings = []
    for text in df[text_column]:
        try:
            resp = openai.Embedding.create(input=[text], engine=model)
            embeddings.append(resp["data"][0]["embedding"])
        except Exception as exc:
            logging.error("Embedding generation failed: %s", exc)
            embeddings.append(np.nan)
    df["embeddings"] = [json.dumps(e) for e in embeddings]


def add_embeddings_to_csv(input_path: str, output_path: str, text_column: str = "Processed_Concern", model: str = "text-similarity-babbage-001") -> None:
    """Load a CSV, generate embeddings, and save the result."""
    setup_logging("embedding.log")
    df = load_data(input_path)
    if df is None:
        return
    generate_openai_embeddings(df, text_column, model=model)
    save_data(df, output_path)
    logging.info("Embeddings saved to %s", output_path)


"""Clustering utilities for embeddings."""

import json
import logging
import os

import numpy as np
from hdbscan import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from utils import load_data, save_data, setup_logging


def cosine_matrix(embeddings: np.ndarray, output_file: str | None = None) -> np.ndarray:
    """Return a cosine similarity matrix and optionally save it."""
    mat = cosine_similarity(embeddings)
    if output_file:
        np.save(output_file, mat)
    return mat


def hdbscan_clustering(
    embeddings: np.ndarray | None = None,
    *,
    cosine_sim: np.ndarray | None = None,
    min_cluster_size: int = 5,
    ) -> np.ndarray:
    """Run HDBSCAN on embeddings or a precomputed cosine matrix."""
    if cosine_sim is None:
        if embeddings is None:
            raise ValueError("Either embeddings or cosine_sim must be provided")
        cosine_sim = cosine_matrix(embeddings)
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric="precomputed")
    labels = clusterer.fit_predict(1 - cosine_sim)
    return labels


def cluster_csv(
    input_path: str,
    output_path: str,
    embedding_column: str = "embeddings",
    cosine_path: str | None = None,
    ) -> None:
    """Cluster a CSV of embeddings, optionally using a precomputed matrix."""
    setup_logging("clustering.log")
    df = load_data(input_path)
    if df is None or embedding_column not in df.columns:
        logging.error("Embeddings column '%s' missing", embedding_column)
        return

    embeddings = df[embedding_column].dropna().apply(
        lambda x: np.asarray(json.loads(x)) if isinstance(x, str) else np.asarray(x)
    )
    if embeddings.empty:
        logging.error("No embeddings available for clustering")
        return

    if cosine_path and os.path.exists(cosine_path):
        cosine_sim = np.load(cosine_path)
        labels = hdbscan_clustering(cosine_sim=cosine_sim)
    else:
        labels = hdbscan_clustering(np.stack(embeddings))

    df.loc[embeddings.index, "cluster"] = labels
    save_data(df, output_path)
    logging.info("Cluster assignments saved to %s", output_path)


def precompute_csv_cosine(input_path: str, output_file: str, embedding_column: str = "embeddings") -> None:
    """Precompute a cosine similarity matrix from embeddings in a CSV."""
    setup_logging("clustering.log")
    df = load_data(input_path)
    if df is None or embedding_column not in df.columns:
        logging.error("Embeddings column '%s' missing", embedding_column)
        return

    embeddings = df[embedding_column].dropna().apply(
        lambda x: np.asarray(json.loads(x)) if isinstance(x, str) else np.asarray(x)
    )
    if embeddings.empty:
        logging.error("No embeddings to compute cosine matrix")
        return

    cosine_matrix(np.stack(embeddings), output_file)
    logging.info("Cosine matrix saved to %s", output_file)


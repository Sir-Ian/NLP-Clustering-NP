"""Clustering utilities for embeddings."""

import json
import logging

import numpy as np
from hdbscan import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from utils import load_data, save_data, setup_logging


def hdbscan_clustering(embeddings: np.ndarray, min_cluster_size: int = 5) -> np.ndarray:
    """Run HDBSCAN on a cosine distance matrix."""
    cosine_sim = cosine_similarity(embeddings)
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric="precomputed")
    labels = clusterer.fit_predict(1 - cosine_sim)
    return labels


def cluster_csv(input_path: str, output_path: str, embedding_column: str = "embeddings") -> None:
    """Load embeddings from CSV, cluster them, and save the labeled data."""
    setup_logging("clustering.log")
    df = load_data(input_path)
    if df is None or embedding_column not in df.columns:
        logging.error("Embeddings column '%s' missing", embedding_column)
        return

    embeddings = df[embedding_column].dropna().apply(lambda x: np.asarray(json.loads(x)) if isinstance(x, str) else np.asarray(x))
    if embeddings.empty:
        logging.error("No embeddings available for clustering")
        return

    labels = hdbscan_clustering(np.stack(embeddings))
    df.loc[embeddings.index, "cluster"] = labels
    save_data(df, output_path)
    logging.info("Cluster assignments saved to %s", output_path)


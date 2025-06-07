# NLP-Clustering-NP

This project contains utilities for structuring and clustering customer chat logs using OpenAI embeddings and HDBSCAN. The original scripts were consolidated into a small library so the workflow is easier to follow.

## Modules

- **`nlp_clustering.preprocessing`** – text cleaning and lemmatization utilities
- **`nlp_clustering.embeddings`** – functions to generate embeddings via OpenAI
- **`nlp_clustering.clustering`** – HDBSCAN clustering on cosine distances
- **`nlp_clustering.faq`** – match FAQ questions to existing responses
- **`nlp_clustering.tokens`** – estimate token counts

Example usage is provided in `scripts/run_pipeline.py` which performs preprocessing, embedding generation and clustering for a CSV file.

For very large datasets you may prefer to pre-compute the cosine similarity matrix once and reuse it for clustering. The function `precompute_csv_cosine` in `nlp_clustering.clustering` does this and stores the matrix to a `.npy` file. `cluster_csv` can then load the matrix using the `cosine_path` argument.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set the `OPENAI_API_KEY` environment variable before running scripts that access the OpenAI API.

## Running the example pipeline

```bash
python scripts/run_pipeline.py input.csv output.csv
```

This will create intermediate files with preprocessed text and embeddings in the working directory and output a CSV containing cluster labels.

To pre-compute a cosine similarity matrix and cluster using it:

```bash
python scripts/precompute_cosine.py embedded.csv cosine.npy
python scripts/run_pipeline.py input.csv output.csv --cosine cosine.npy
```


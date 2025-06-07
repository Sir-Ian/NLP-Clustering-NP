"""Example pipeline script for preprocessing, embedding, and clustering."""

import argparse
from nlp_clustering import preprocessing, embeddings, clustering


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NLP clustering pipeline")
    parser.add_argument("input", help="Path to raw CSV file")
    parser.add_argument("output", help="Path to output clustered CSV file")
    parser.add_argument("--model", default="text-similarity-babbage-001", help="OpenAI embedding model")
    parser.add_argument("--cosine", help="Optional precomputed cosine .npy file")
    args = parser.parse_args()

    tmp_preprocessed = "preprocessed_tmp.csv"
    tmp_embedded = "embedded_tmp.csv"

    preprocessing.preprocess_dataframe(args.input, tmp_preprocessed)
    embeddings.add_embeddings_to_csv(tmp_preprocessed, tmp_embedded, model=args.model)
    clustering.cluster_csv(tmp_embedded, args.output, cosine_path=args.cosine)


if __name__ == "__main__":
    main()



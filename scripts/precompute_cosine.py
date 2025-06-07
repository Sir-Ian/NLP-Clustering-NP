"""Precompute cosine similarity matrix from an embeddings CSV."""

import argparse
from nlp_clustering import clustering


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute cosine similarity matrix")
    parser.add_argument("input", help="CSV with embeddings")
    parser.add_argument("output", help="Output .npy file for cosine matrix")
    parser.add_argument("--column", default="embeddings", help="Embeddings column name")
    args = parser.parse_args()

    clustering.precompute_csv_cosine(args.input, args.output, embedding_column=args.column)


if __name__ == "__main__":
    main()


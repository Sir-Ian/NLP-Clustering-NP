"""Text preprocessing utilities."""

from typing import Iterable
import logging
import string

import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from utils import load_data, save_data, setup_logging


nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


def preprocess_text(text: str, nlp_model, lemmatizer, stop_words: Iterable[str]) -> str:
    """Tokenize, lemmatize, and remove stop words from text."""
    doc = nlp_model(text)
    lemmas = [lemmatizer.lemmatize(token.text.lower()) for token in doc]
    cleaned = [w for w in lemmas if w not in stop_words and w not in string.punctuation]
    return " ".join(cleaned)


def preprocess_dataframe(file_path: str, output_path: str) -> None:
    """Load CSV, clean text columns, and save the result."""
    setup_logging("preprocess_data.log")
    df = load_data(file_path)
    if df is None:
        logging.error("Could not load file: %s", file_path)
        return

    # Drop unnamed or empty columns
    df.drop(columns=[c for c in df.columns if "Unnamed" in c], inplace=True)
    df.fillna("", inplace=True)

    nlp_model = spacy.load("en_core_web_sm")
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    text_columns = df.select_dtypes(include=["object"]).columns
    for col in text_columns:
        df[col + "_clean"] = df[col].apply(lambda t: preprocess_text(t, nlp_model, lemmatizer, stop_words))

    save_data(df, output_path)
    logging.info("Preprocessed data saved to %s", output_path)


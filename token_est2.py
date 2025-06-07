from utils import setup_logging, load_data
import logging
import pandas as pd

def estimate_tokens(file_path, column_name='Processed_Concern', sample_size=100, avg_token_length=4):
    """
    Estimate the number of tokens for a subset of the dataset based on average token length.
    """
    setup_logging('token_est.log')
    df = load_data(file_path)
    if df is None:
        logging.error(f"Could not load file: {file_path}")
        return 0

    # Select a random sample of the data
    df_sample = df.sample(n=sample_size)

    total_tokens = 0
    for text in df_sample[column_name]:
        # Estimate the number of tokens as the total character count divided by the average token length
        token_count = len(text) / avg_token_length
        total_tokens += token_count

    print(f"Estimated total tokens for {sample_size} samples: {total_tokens:.0f}")
    logging.info(f"Estimated total tokens for {sample_size} samples: {total_tokens:.0f}")
    return total_tokens

# Usage
file_path = '/Users/ian/Desktop/TokenEst.csv'
estimate_tokens(file_path, sample_size=380)

import pandas as pd
import logging
import os

def setup_logging(log_file='processing.log', level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        filename=log_file,
        level=level,
        format='%(asctime)s %(levelname)s: %(message)s',
        filemode='a'
    )


def load_data(file_path):
    """Load a CSV file into a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        return None


def save_data(df, file_path):
    """Save a DataFrame to a CSV file."""
    try:
        df.to_csv(file_path, index=False)
        logging.info(f"File saved successfully at {file_path}")
    except Exception as e:
        logging.error(f"Error saving file at {file_path}: {e}")


def get_env_variable(var_name, default=None):
    """Get an environment variable or return default."""
    return os.environ.get(var_name, default)

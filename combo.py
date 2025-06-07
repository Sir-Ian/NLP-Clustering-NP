import pandas as pd
import numpy as np
import openai
import logging
from hdbscan import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import json

# Setup logging
logging.basicConfig(filename='/Users/ian/Desktop/ChatData/Logs/processing.log',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    filemode='a')  # 'w' to overwrite the log file on each run

logging.info("Script started")

def load_data(file_path):
    logging.info(f"Attempting to load data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise e  # Re-raise the exception after logging

def generate_openai_embeddings(df, openai_api_key):
    logging.info("Starting to generate embeddings")
    openai.api_key = openai_api_key

    df['embeddings'] = None  # Initialize the column for embeddings
    logging.info("Starting to generate OpenAI embeddings")

    for i, row in df.iterrows():
        try:
            response = openai.Embedding.create(input=[row['Processed_Concern']], engine="text-similarity-babbage-001")
            df.at[i, 'embeddings'] = json.dumps(response['data'][0]['embedding'])
            logging.info(f"Embeddings generated for row {i}")
        except Exception as e:
            logging.error(f"Error generating embeddings for row {i}: {e}")
            df.at[i, 'embeddings'] = np.nan  # Set as NaN to handle missing embeddings later

    logging.info("OpenAI embeddings generated for all rows successfully")
    return df

def perform_clustering(df):
    logging.info("Starting clustering process") 
    try:
        # Convert string embeddings back to lists
        if 'embeddings' not in df.columns or df['embeddings'].isnull().all():
            logging.error("No embeddings available for clustering.")
            return

        embeddings = df['embeddings'].dropna().apply(json.loads)
        if embeddings.empty:
            logging.error("All embeddings are NaN or empty.")
            return

        embeddings_matrix = np.array(embeddings.tolist())
        if embeddings_matrix.ndim != 2 or embeddings_matrix.shape[0] == 0:
            logging.error(f"Embeddings matrix is not in the correct shape: {embeddings_matrix.shape}")
            return

        cosine_sim = cosine_similarity(embeddings_matrix)
        logging.info("Cosine similarity matrix computed")

        clusterer = HDBSCAN(min_cluster_size=5, metric='precomputed')
        cluster_labels = clusterer.fit_predict(1 - cosine_sim)

        df.loc[df.index[embeddings.index], 'cluster'] = cluster_labels
        logging.info("Clustering completed successfully")

    except Exception as e:
        logging.error(f"Error in clustering: {e}")
        raise e

def save_data(df, file_path):
    logging.info(f"Attempting to save data to {file_path}")
    try:
        df.to_csv(file_path, index=False)
        logging.info(f"File saved successfully at {file_path}")
    except Exception as e:
        logging.error(f"Error saving file at {file_path}: {e}")
        raise e

if __name__ == "__main__":
    file_path = '/Users/ian/Downloads/cleancsvmaybe.csv'
    output_file_path = '/Users/ian/Desktop/ChatData/Output/test_final_output_with_embeddings.csv'
    openai_api_key = ''  # Replace with your actual OpenAI API key

    df = load_data(file_path)
    if df is not None:
        df = generate_openai_embeddings(df, openai_api_key)
        if df is not None:
            perform_clustering(df)
            save_data(df, output_file_path)
logging.info("Script finished")
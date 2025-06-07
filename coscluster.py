import pandas as pd
import numpy as np
import logging
from hdbscan import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(filename='/Users/ian/Desktop/ChatData/Logs/processing.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info("Data loaded successfully")
        return df
    except Exception as e:
        logging.error("Error loading data: " + str(e))
        return None

def convert_embeddings(df):
    try:
        # Convert embeddings from string to list of floats
        df['embeddings'] = df['embeddings'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))
        logging.info("Embeddings converted successfully")
    except Exception as e:
        logging.error("Error converting embeddings: " + str(e))

def perform_clustering(df):
    try:
        # Convert embeddings to cosine similarity
        cosine_sim = cosine_similarity(np.stack(df['embeddings']))
        logging.info("Cosine similarity matrix computed")

        # Clustering with HDBSCAN
        clusterer = HDBSCAN(min_cluster_size=5, metric='precomputed')
        df['cluster'] = clusterer.fit_predict(1 - cosine_sim) # 1 - cosine_sim converts similarity to distance
        logging.info("Clustering completed successfully")
    except Exception as e:
        logging.error("Error in clustering: " + str(e))

def save_data(df, file_path):
    try:
        df.to_csv(file_path, index=False)
        logging.info("File saved successfully")
    except Exception as e:
        logging.error("Error saving file: " + str(e))

# Main execution
if __name__ == "__main__":
    file_path = '/Users/ian/Desktop/ChatData/Input/clustered_data_with_clusters.csv'
    output_file_path = '/Users/ian/Desktop/ChatData/Output/cos_cluster_output.csv'

    df = load_data(file_path)

    if df is not None:
        convert_embeddings(df)
        perform_clustering(df)
        save_data(df, output_file_path)

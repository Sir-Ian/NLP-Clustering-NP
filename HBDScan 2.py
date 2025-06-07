from utils import setup_logging, load_data, save_data
import pandas as pd
import numpy as np
from hdbscan import HDBSCAN
from sklearn.metrics.pairwise import cosine_distances
import logging

# Set up logging
setup_logging('/Users/ian/Desktop/ChatData/Logs/processing.log')
logging.info('Starting script')

try:
    # Load the data with embeddings
    file_path = '/Users/ian/Desktop/ChatData/Input/processed_data_with_embeddings.csv'
    df = load_data(file_path)
    if df is None:
        raise Exception('Data could not be loaded')
    logging.info('Data loaded successfully')

    # Convert embeddings from string to list of floats
    embeddings = df['embeddings'].apply(lambda x: np.fromstring(x[1:-1], sep=' ')).tolist()
    logging.info('Converted embeddings from string to list of floats')

    # Chunking setup
    batch_size = 1000  # Adjust based on your system's memory capacity
    num_embeddings = len(embeddings)
    cosine_dist = np.zeros((num_embeddings, num_embeddings))

    # Compute pairwise cosine distances in chunks
    logging.info('Starting to compute pairwise cosine distances in chunks')
    for start_idx in range(0, num_embeddings, batch_size):
        end_idx = min(start_idx + batch_size, num_embeddings)
        batch = embeddings[start_idx:end_idx]
        cosine_dist[start_idx:end_idx, :] = cosine_distances(batch, embeddings)
    logging.info('Pairwise cosine distances computed in chunks')

    # Clustering with HDBSCAN using precomputed distances
    logging.info('Starting HDBSCAN clustering')
    clusterer = HDBSCAN(min_cluster_size=3, metric='precomputed')
    clusters = clusterer.fit_predict(cosine_dist)
    logging.info('HDBSCAN clustering completed')

    # Create a new DataFrame for clusters
    cluster_df = pd.DataFrame({'cluster': clusters})

    # Concatenate the new DataFrame with the original DataFrame
    final_df = pd.concat([df, cluster_df], axis=1)

    # Save the final DataFrame with original data and new cluster labels
    output_path = '/Users/ian/Desktop/ChatData/Output/clustered_data_with_clusters.csv'
    save_data(final_df, output_path)
    logging.info('File has been processed and saved successfully')

except Exception as e:
    logging.error('Error occurred: ' + str(e))

logging.info('Script completed')

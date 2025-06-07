from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from utils import setup_logging, load_data, save_data
import logging

def main():
    setup_logging('faq_processing.log')
    # Pre-trained model
    model_name = 'all-mpnet-base-v2' # Model provided by Hugging Face's transformers library

    # File paths to your CSV files
    preprocessed_data_path = '/Users/ian/Downloads/PreprocessData.csv'
    remaining_faq_path = '/Users/ian/Downloads/remaining_structured_faq_items.csv'

    # Load the pre-trained model
    model = SentenceTransformer(model_name)

    # Read the CSV files
    preprocessed_data_df = load_data(preprocessed_data_path)
    remaining_faq_df = load_data(remaining_faq_path)
    if preprocessed_data_df is None or remaining_faq_df is None:
        logging.error('Could not load input data.')
        return

    # Generate embeddings for the concerns and questions
    # Note: Ensure you have sufficient RAM, as this may require a lot of memory
    concerns = preprocessed_data_df['Concern'].dropna().tolist()
    questions = remaining_faq_df['Question'].tolist()
    
    preprocessed_embeddings = model.encode(concerns, show_progress_bar=True)
    faq_embeddings = model.encode(questions, show_progress_bar=True)

    # Compute cosine similarities
    cos_sim = cosine_similarity(faq_embeddings, preprocessed_embeddings)

    # Iterate through the similarities and find the best match
    for idx, question in enumerate(questions):
        sim_scores = cos_sim[idx]
        best_match_idx = np.argmax(sim_scores)
        
        # You can adjust the similarity threshold as needed
        if sim_scores[best_match_idx] > 0.5:
            remaining_faq_df.loc[idx, 'Answer'] = preprocessed_data_df.iloc[best_match_idx]['Response']
        else:
            remaining_faq_df.loc[idx, 'Answer'] = 'REVIEW: No satisfying answer found.'

    # Save the results to a CSV
    output_path = '/Users/ian/Library/CloudStorage/OneDrive-JeffreyCik/Code/updated_remaining_structured_faq_items.csv'
    save_data(remaining_faq_df, output_path)
    print(f"FAQ processing complete! Updated file saved as '{output_path}'.")
    logging.info(f"FAQ processing complete! Updated file saved as '{output_path}'.")

if __name__ == "__main__":
    main()
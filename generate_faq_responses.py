from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

def main():
    # Pre-trained model
    model_name = 'all-mpnet-base-v2' # Model provided by Hugging Face's transformers library

    # File paths to your CSV files
    preprocessed_data_path = '/Users/ian/Downloads/PreprocessData.csv'
    remaining_faq_path = '/Users/ian/Downloads/remaining_structured_faq_items.csv'

    # Load the pre-trained model
    model = SentenceTransformer(model_name)

    # Read the CSV files
    preprocessed_data_df = pd.read_csv(preprocessed_data_path)
    remaining_faq_df = pd.read_csv(remaining_faq_path)

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
    remaining_faq_df.to_csv(output_path, index=False)
    print(f"FAQ processing complete! Updated file saved as '{output_path}'.")

if __name__ == "__main__":
    main()
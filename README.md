# NLP-Clustering-NP
NLP-based clustering solution using OpenAI embeddings, Python, and HDBSCAN to analyze 40,000 customer service interactions, accurately identifying top questions enabling AI-driven customer service automation

# NLP Chat Log Clustering & Automation

This project tackles the complex challenge of analyzing ~5 years of unstructured customer service chat logs (about 40,000 interactions). The original data consisted of continuous text strings with no clear separation between customer questions and agent responses.

## Approach

- **Data Structuring:**  
  Used OpenAI GPT models to extract and structure chat logs into clear question-answer pairs, transforming raw transcripts into analyzable data.

- **Semantic Embedding:**  
  Leveraged OpenAI's Text Embedding API to convert text into high-dimensional semantic embeddings, enabling robust understanding of meaning beyond simple keyword matching.

- **Clustering:**  
  Applied advanced clustering techniques—cosine similarity and HDBSCAN—to group semantically similar queries, even when phrased differently (e.g., "Does the product contain metal?" vs. "Does the product contain cloth?").

## Workflow

1. **Cleaning & Structuring:**  
   Scripts like [`Modified_Clean_and_Structure_Chat_Data.py`](Modified_Clean_and_Structure_Chat_Data.py) and [`Parse_GPT_Response.py`](Parse_GPT_Response.py) process raw logs and extract structured Q&A pairs.

2. **Preprocessing:**  
   [`Preprocess Data.py`](Preprocess%20Data.py) cleans and lemmatizes text, preparing it for embedding.

3. **Embedding Generation:**  
   [`combo.py`](combo.py) and [`generate_faq_responses.py`](generate_faq_responses.py) generate semantic embeddings using OpenAI or Sentence Transformers.

4. **Clustering:**  
   [`coscluster.py`](coscluster.py) and [`HBDScan 2.py`](HBDScan%202.py) perform clustering using cosine similarity and HDBSCAN to identify common question types.

5. **Token Estimation & Utilities:**  
   [`token_est2.py`](token_est2.py) estimates token usage for cost management. [`utils.py`](utils.py) provides shared functions for logging and data I/O.

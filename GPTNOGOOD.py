from utils import setup_logging, load_data, save_data
import pandas as pd
import openai
import logging

def chunk_dataframe(df, chunk_size):
    """ Yield successive chunk_size chunks from DataFrame """
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i:i + chunk_size]

def extract_unique_questions(chunk):
    """ Send a chunk of data to the OpenAI API to extract unique questions """
    prompt = f"Please extract unique, broadly applicable customer questions from the following data:\n\n{chunk.to_string()}"
    response = openai.Completion.create(
        model="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=2048  # Adjust as necessary
    )
    return response.choices[0].text

def main():
    setup_logging('gptnogood.log')
    input_file = 'path_to_your_input_file.csv'
    output_file = 'path_to_your_output_file.csv'
    chunk_size = 20  # Adjust based on your data and GPT's limits

    # Set up API key
    openai.api_key = 'your-api-key'

    # Process the data
    df = load_data(input_file)
    if df is None:
        logging.error('Could not load input data.')
        return
    unique_questions = []
    for chunk in chunk_dataframe(df, chunk_size):
        extracted_questions = extract_unique_questions(chunk)
        unique_questions.extend(extracted_questions.split('\n'))  # Assuming each question is on a new line

    # Save the results
    save_data(pd.DataFrame({'Unique Questions': unique_questions}), output_file)
    print(f"Unique questions extracted and saved to {output_file}")
    logging.info(f"Unique questions extracted and saved to {output_file}")

if __name__ == '__main__':
    main()

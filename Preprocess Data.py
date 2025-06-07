from utils import setup_logging, load_data, save_data
import logging
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# Ensure you have these NLTK datasets downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

setup_logging('preprocess_data.log')
file_path = '/Users/ian/Downloads/PreprocessData.csv'  # Replace with your file path
data = load_data(file_path)
if data is None:
    logging.error(f"Could not load file: {file_path}")
    exit(1)

# Remove irrelevant or empty columns
columns_to_drop = [col for col in data.columns if 'Unnamed' in col]
data.drop(columns=columns_to_drop, inplace=True)

# Handling missing values
data.fillna('', inplace=True)

# Initialize NLP tools
nlp = spacy.load('en_core_web_sm')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    # Tokenization and lemmatization
    doc = nlp(text)
    lemmatized = [lemmatizer.lemmatize(token.text.lower()) for token in doc]
    
    # Remove punctuation and stop words
    cleaned_text = [word for word in lemmatized if word not in stop_words and word not in string.punctuation]
    
    return ' '.join(cleaned_text)

# Apply preprocessing to all text columns
text_columns = data.select_dtypes(include=['object']).columns
for col in text_columns:
    data[col + '_clean'] = data[col].apply(preprocess_text)

# Save the preprocessed data to a new CSV file
output_file_path = '/Users/ian/Downloads/postprocessed_data.csv'  # You can change this file name and path as needed
save_data(data, output_file_path)
logging.info(f"Preprocessed data saved to {output_file_path}")

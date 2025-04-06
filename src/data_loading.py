import pandas as pd
import os
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK resources at import time
import nltk
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def load_dataset(file_path):
    """
    Load a dataset from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        # First try standard CSV loading
        return pd.read_csv(file_path)
    except Exception as e:
        try:
            # If that fails, try with error handling options for newer pandas versions
            return pd.read_csv(file_path, 
                               on_bad_lines='skip',    # Skip bad lines (new parameter)
                               quoting=3,              # No special quoting
                               sep=None,               # Auto-detect separator
                               engine='python')        # Use python engine for more flexibility
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")

def get_available_datasets(directory="datasets/"):
    """
    Get a list of available CSV datasets in the specified directory.
    
    Args:
        directory (str): Directory to search for CSV files.
        
    Returns:
        list: List of CSV file paths.
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        csv_files = [os.path.join(directory, f) for f in os.listdir(directory) 
                    if f.endswith('.csv')]
        return csv_files
    except Exception as e:
        raise Exception(f"Error listing datasets: {str(e)}")

def preprocess_text(text, remove_stopwords=True, perform_stemming=False, perform_lemmatization=False, handle_negations=False):
    """
    Preprocess text for sentiment analysis.
    
    Args:
        text (str): Text to preprocess.
        remove_stopwords (bool): Whether to remove stopwords.
        perform_stemming (bool): Whether to perform stemming.
        perform_lemmatization (bool): Whether to perform lemmatization.
        handle_negations (bool): Whether to handle negations (not good -> not_good).
        
    Returns:
        str: Preprocessed text.
    """
    # Handle missing values
    if pd.isna(text):
        return ""
    
    # Convert to string if input is not a string
    try:
        if not isinstance(text, str):
            text = str(text)
    except:
        return ""  # Return empty string if conversion fails
    
    # Handle empty strings
    if not text.strip():
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Handle negations before removing punctuation if requested
    if handle_negations:
        # Replace "not" followed by a word with "not_word"
        text = re.sub(r'not\s+(\w+)', r'not_\1', text)
        # Replace "n't" contractions
        text = re.sub(r"n't\s+(\w+)", r'not_\1', text)
        text = text.replace("n't", " not")
    
    # Remove punctuation, but keep the underscores used in negation handling
    if handle_negations:
        # Remove all punctuation except underscores
        punct = string.punctuation.replace('_', '')
        text = re.sub(f'[{re.escape(punct)}]', ' ', text)
    else:
        # Remove all punctuation
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    
    # Use proper tokenization with NLTK
    tokens = word_tokenize(text)
    
    # Remove stopwords (but keep negation words if we're handling negations)
    if remove_stopwords:
        try:
            stop_words = set(stopwords.words('english'))
            # If handling negations, keep "not" in the text
            if handle_negations:
                stop_words.discard('not')
            tokens = [token for token in tokens if token not in stop_words]
        except:
            nltk.download('stopwords')
            stop_words = set(stopwords.words('english'))
            if handle_negations:
                stop_words.discard('not')
            tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization takes precedence over stemming if both are selected
    if perform_lemmatization:
        try:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
        except:
            nltk.download('wordnet')
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Only perform stemming if lemmatization is not selected
    elif perform_stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    
    # Join tokens back into text
    return ' '.join(tokens)

def normalize_sentiment_labels(df, sentiment_column):
    """
    Normalize sentiment labels to 'positive', 'negative', and 'neutral'.
    Maps various formats to standard format.
    
    Args:
        df (pd.DataFrame): Dataset with sentiment labels.
        sentiment_column (str): Name of the column containing sentiment labels.
        
    Returns:
        pd.DataFrame: Dataset with normalized sentiment labels.
    """
    # Create a copy of the dataframe
    df_normalized = df.copy()
    
    # Define mappings for various label formats
    positive_labels = ['positive', 'pos', '1', 1, 'yes', 'good', 'true', True, '4', '5', 4, 5]
    negative_labels = ['negative', 'neg', '0', 0, 'no', 'bad', 'false', False, '1', '2', 1, 2]
    neutral_labels = ['neutral', 'neu', '2', 2, 'maybe', 'ok', 'okay', '3', 3]
    
    # Apply mapping
    def map_sentiment(label):
        if str(label).lower() in [str(l).lower() for l in positive_labels]:
            return 'positive'
        elif str(label).lower() in [str(l).lower() for l in negative_labels]:
            return 'negative'
        elif str(label).lower() in [str(l).lower() for l in neutral_labels]:
            return 'neutral'  # Default for any other label
        else:
            return 'neutral'  # Default for any other label
    
    df_normalized[sentiment_column] = df_normalized[sentiment_column].apply(map_sentiment)
    
    return df_normalized 
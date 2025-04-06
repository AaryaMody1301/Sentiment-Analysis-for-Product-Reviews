import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer

# Download NLTK resources at import time
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def preprocess_text(text, remove_stopwords=True, perform_stemming=False, 
                   perform_lemmatization=True, handle_negations=True):
    """
    Preprocess text for sentiment analysis.
    
    Args:
        text (str): Text to preprocess.
        remove_stopwords (bool): Whether to remove stopwords.
        perform_stemming (bool): Whether to perform stemming (not used if lemmatization is True).
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
        stop_words = set(stopwords.words('english'))
        # If handling negations, keep "not" in the text
        if handle_negations:
            stop_words.discard('not')
        tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization takes precedence over stemming if both are selected
    if perform_lemmatization:
        lemmatizer = WordNetLemmatizer()
        # Better lemmatization with POS tagging
        lemmatized_tokens = []
        for token in tokens:
            # Try lemmatizing as verb first, then as noun
            lemmatized_verb = lemmatizer.lemmatize(token, pos='v')
            # If lemmatizing as verb changes the word, use that
            if lemmatized_verb != token:
                lemmatized_tokens.append(lemmatized_verb)
            else:
                # Otherwise use noun lemmatization (default)
                lemmatized_tokens.append(lemmatizer.lemmatize(token))
        tokens = lemmatized_tokens
    
    # Join tokens back into text
    return ' '.join(tokens)

def create_tfidf_vectorizer(max_features=5000, ngram_range=(1, 2), use_idf=True):
    """
    Create a TF-IDF vectorizer with specified parameters.
    
    Args:
        max_features (int): Maximum number of features to extract.
        ngram_range (tuple): The lower and upper boundary of the range of n-values for n-grams.
        use_idf (bool): Whether to use inverse document frequency weighting.
        
    Returns:
        TfidfVectorizer: Configured TF-IDF vectorizer.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english',  # Remove English stop words
        use_idf=use_idf,
        min_df=5,  # Ignore terms that appear in less than 5 documents
        max_df=0.8,  # Ignore terms that appear in more than 80% of documents
        sublinear_tf=True  # Apply sublinear tf scaling (1 + log(tf))
    )

def create_hashing_vectorizer(n_features=2**18, ngram_range=(1, 2)):
    """
    Create a hashing vectorizer for memory-efficient feature extraction.
    
    Args:
        n_features (int): Number of features to extract.
        ngram_range (tuple): The lower and upper boundary of the range of n-values for n-grams.
        
    Returns:
        HashingVectorizer: Configured hashing vectorizer.
    """
    return HashingVectorizer(
        n_features=n_features,
        ngram_range=ngram_range,
        alternate_sign=False,  # No negative values, useful for Naive Bayes
        norm='l2'  # Normalize feature vectors
    )

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
            return 'neutral'
        else:
            return 'neutral'  # Default for any other label
    
    df_normalized[sentiment_column] = df_normalized[sentiment_column].apply(map_sentiment)
    
    return df_normalized

def detect_columns(df):
    """
    Automatically detect text and sentiment columns based on column names and content.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
        
    Returns:
        tuple: (text_column, sentiment_column)
    """
    # Common text column names
    text_patterns = ['review', 'text', 'comment', 'feedback', 'description', 'content', 'message']
    
    # Common sentiment column names
    sentiment_patterns = ['sentiment', 'label', 'rating', 'score', 'class', 'polarity', 'emotion', 'star']
    
    # Initialize best matches
    text_col = None
    sentiment_col = None
    
    # Get all column names in lowercase for case-insensitive matching
    cols_lower = {col.lower(): col for col in df.columns}
    
    # First, look for exact matches in column names
    for pattern in text_patterns:
        for col_lower, col in cols_lower.items():
            if pattern in col_lower:
                text_col = col
                break
        if text_col:
            break
    
    for pattern in sentiment_patterns:
        for col_lower, col in cols_lower.items():
            if pattern in col_lower:
                sentiment_col = col
                break
        if sentiment_col:
            break
    
    # If no match found yet, try heuristics based on content
    if not text_col or not sentiment_col:
        # Examine each column
        for col in df.columns:
            # Skip if we've already found this column type
            if col == sentiment_col:
                continue
                
            # Check if it might be a text column
            if not text_col and df[col].dtype == 'object':
                # Look at the average length of strings
                avg_len = df[col].astype(str).str.len().mean()
                # Text columns typically have longer strings
                if avg_len > 20:  # Arbitrary threshold
                    text_col = col
            
            # Check if it might be a sentiment column
            if not sentiment_col and df[col].dtype != 'object':
                # Sentiment columns often have a small number of unique values
                if df[col].nunique() <= 5:  # Typical for ratings (1-5)
                    sentiment_col = col
    
    # If still no match, take best guess
    if not text_col and len(df.columns) > 0:
        # Take the first string column with the longest average length
        str_cols = df.select_dtypes(include=['object']).columns
        if len(str_cols) > 0:
            avg_lengths = {col: df[col].astype(str).str.len().mean() for col in str_cols}
            text_col = max(avg_lengths.items(), key=lambda x: x[1])[0]
        else:
            text_col = df.columns[0]
    
    if not sentiment_col and len(df.columns) > 1:
        # Take a numeric column with few unique values, excluding any ID-like columns
        num_cols = df.select_dtypes(include=['number']).columns
        if len(num_cols) > 0:
            # Filter out columns that are likely IDs (many unique values)
            potential_cols = {}
            for col in num_cols:
                if col != text_col and df[col].nunique() < max(10, len(df) * 0.1):
                    potential_cols[col] = df[col].nunique()
            
            if potential_cols:
                # Choose column with fewest unique values
                sentiment_col = min(potential_cols.items(), key=lambda x: x[1])[0]
            else:
                # If no good candidate, take the first numeric column that's not the text column
                for col in num_cols:
                    if col != text_col:
                        sentiment_col = col
                        break
        
        if not sentiment_col:
            # Take the column with the fewest unique values that's not the text column
            unique_counts = {col: df[col].nunique() for col in df.columns if col != text_col}
            if unique_counts:
                sentiment_col = min(unique_counts.items(), key=lambda x: x[1])[0]
            else:
                sentiment_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    
    return text_col, sentiment_col

def compute_class_weights(y):
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        y (array-like): Target labels
        
    Returns:
        dict: Class weights dictionary
    """
    classes = np.unique(y)
    class_counts = np.bincount([np.where(classes == c)[0][0] for c in y])
    total_samples = len(y)
    
    # Calculate weights as inverse of class frequency
    weights = total_samples / (len(classes) * class_counts)
    
    # Create dictionary mapping class labels to weights
    class_weights = {c: w for c, w in zip(classes, weights)}
    
    return class_weights

def generate_feature_names(vectorizer, top_n=20):
    """
    Get feature names from a vectorizer.
    
    Args:
        vectorizer: TF-IDF vectorizer (must not be a hashing vectorizer)
        top_n (int): Number of top features to return
        
    Returns:
        list: List of feature names, or None if using HashingVectorizer
    """
    if hasattr(vectorizer, 'get_feature_names_out'):
        feature_names = vectorizer.get_feature_names_out()
        return feature_names
    else:
        return None 
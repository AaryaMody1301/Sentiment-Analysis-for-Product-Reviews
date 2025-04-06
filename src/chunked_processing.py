import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import os
import re
import time
import joblib
from tqdm import tqdm

from src.data_loading import preprocess_text, normalize_sentiment_labels

def detect_columns(df):
    """
    Automatically detect text and sentiment columns based on column names and content.
    
    Args:
        df (pd.DataFrame): Sample dataframe to analyze
        
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
    if not text_col:
        # Take the first string column with the longest average length
        str_cols = df.select_dtypes(include=['object']).columns
        if len(str_cols) > 0:
            avg_lengths = {col: df[col].astype(str).str.len().mean() for col in str_cols}
            text_col = max(avg_lengths.items(), key=lambda x: x[1])[0]
    
    if not sentiment_col:
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
    
    # Final fallback - use first and second columns if still not found
    if not text_col and len(df.columns) > 0:
        text_col = df.columns[0]
    
    if not sentiment_col and len(df.columns) > 1:
        sentiment_col = df.columns[1]
    elif not sentiment_col and len(df.columns) > 0 and df.columns[0] != text_col:
        sentiment_col = df.columns[0]
    
    return text_col, sentiment_col

def process_large_file(file_path, text_column=None, sentiment_column=None, 
                       chunksize=10000, test_size=0.2, 
                       remove_stopwords=True, perform_stemming=False, 
                       perform_lemmatization=True, handle_negations=True, 
                       n_features=2**18, callback=None):
    """
    Process a large CSV file in chunks and train a model incrementally.
    
    Args:
        file_path (str): Path to the CSV file
        text_column (str): Name of the column containing the text data
        sentiment_column (str): Name of the column containing sentiment labels
        chunksize (int): Number of rows to process at once
        test_size (float): Proportion of data to use for testing
        remove_stopwords (bool): Whether to remove stopwords
        perform_stemming (bool): Whether to perform stemming
        perform_lemmatization (bool): Whether to perform lemmatization
        handle_negations (bool): Whether to handle negations
        n_features (int): Number of features for HashingVectorizer
        callback (function): Callback function for progress updates
        
    Returns:
        tuple: (trained_model, vectorizer, metrics)
    """
    start_time = time.time()
    
    # Initialize the vectorizer
    vectorizer = HashingVectorizer(n_features=n_features, alternate_sign=False)
    
    # Initialize model
    model = MultinomialNB(alpha=1.0)
    
    # Calculate total rows for progress reporting
    total_rows = sum(1 for _ in pd.read_csv(file_path, chunksize=10000))
    
    # Initialize metrics
    all_y_true = []
    all_y_pred = []
    
    # First pass: detect columns if not provided
    first_chunk = next(pd.read_csv(file_path, chunksize=min(1000, chunksize)))
    
    if text_column is None or sentiment_column is None:
        detected_text_col, detected_sentiment_col = detect_columns(first_chunk)
        text_column = text_column or detected_text_col
        sentiment_column = sentiment_column or detected_sentiment_col
    
    if callback:
        callback(0, f"Processing file: detected columns - text: {text_column}, sentiment: {sentiment_column}")
    
    # Count chunks for progress
    total_chunks = total_rows // chunksize + (1 if total_rows % chunksize > 0 else 0)
    processed_chunks = 0
    
    # Process chunks
    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunksize)):
        processed_chunks += 1
        
        if callback:
            progress = processed_chunks / total_chunks
            callback(progress, f"Processing chunk {processed_chunks}/{total_chunks}")
        
        # Ensure the required columns exist
        if text_column not in chunk.columns or sentiment_column not in chunk.columns:
            if callback:
                callback(progress, f"Error: Required columns not found in chunk {i+1}")
            raise ValueError(f"Required columns not found: {text_column}, {sentiment_column}")
        
        # Preprocess text
        chunk['processed_text'] = chunk[text_column].apply(
            lambda x: preprocess_text(
                x, 
                remove_stopwords=remove_stopwords,
                perform_stemming=perform_stemming,
                perform_lemmatization=perform_lemmatization,
                handle_negations=handle_negations
            )
        )
        
        # Normalize sentiment labels
        chunk = normalize_sentiment_labels(chunk, sentiment_column)
        
        # Split into train and test
        X = chunk['processed_text'].values
        y = chunk[sentiment_column].values
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
        except ValueError:
            # If stratify fails (e.g., only one class), try without stratification
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        
        # Transform text to feature vectors - for training data
        X_train_vec = vectorizer.transform(X_train)
        
        # Partially fit the model
        classes = np.unique(y_train)
        model.partial_fit(X_train_vec, y_train, classes=classes)
        
        # Transform and predict test data
        X_test_vec = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_vec)
        
        # Collect true and predicted labels for overall metrics
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        # Update progress
        if callback:
            callback(progress, f"Processed chunk {processed_chunks}/{total_chunks}")
    
    # Calculate overall metrics
    metrics = {
        'accuracy': accuracy_score(all_y_true, all_y_pred),
        'precision': precision_score(all_y_true, all_y_pred, average='weighted'),
        'recall': recall_score(all_y_true, all_y_pred, average='weighted'),
        'f1_score': f1_score(all_y_true, all_y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(all_y_true, all_y_pred).tolist(),
        'classes': list(np.unique(all_y_true))
    }
    
    processing_time = time.time() - start_time
    
    if callback:
        callback(1.0, f"Processing complete in {processing_time:.2f} seconds")
    
    return model, vectorizer, metrics

def predict_batch(model, vectorizer, data, text_column, 
                  remove_stopwords=True, perform_stemming=False, 
                  perform_lemmatization=True, handle_negations=True,
                  callback=None):
    """
    Make predictions on a batch of data.
    
    Args:
        model: Trained model
        vectorizer: Feature vectorizer
        data (pd.DataFrame or str): DataFrame or path to CSV
        text_column (str): Column name containing text
        remove_stopwords, perform_stemming, perform_lemmatization, handle_negations: Preprocessing options
        callback: Progress callback function
        
    Returns:
        pd.DataFrame: Original data with predictions and probabilities added
    """
    start_time = time.time()
    
    # Check if data is a file path
    if isinstance(data, str):
        # Calculate total rows for progress
        total_rows = sum(1 for _ in pd.read_csv(data, chunksize=10000))
        total_chunks = total_rows // 10000 + (1 if total_rows % 10000 > 0 else 0)
        
        # Initialize results
        all_results = []
        
        # Process in chunks
        for i, chunk in enumerate(pd.read_csv(data, chunksize=10000)):
            if callback:
                callback((i+1)/total_chunks, f"Processing prediction chunk {i+1}/{total_chunks}")
            
            # Ensure text column exists
            if text_column not in chunk.columns:
                raise ValueError(f"Text column '{text_column}' not found in data")
            
            # Process the chunk
            chunk_results = process_prediction_chunk(
                model, vectorizer, chunk, text_column,
                remove_stopwords, perform_stemming, 
                perform_lemmatization, handle_negations
            )
            
            all_results.append(chunk_results)
        
        # Combine all chunks
        results = pd.concat(all_results, ignore_index=True)
    else:
        # Process as a single chunk
        if callback:
            callback(0.5, "Processing predictions...")
        
        results = process_prediction_chunk(
            model, vectorizer, data, text_column,
            remove_stopwords, perform_stemming, 
            perform_lemmatization, handle_negations
        )
        
        if callback:
            callback(1.0, "Prediction complete")
    
    processing_time = time.time() - start_time
    
    if callback:
        callback(1.0, f"Prediction complete in {processing_time:.2f} seconds")
    
    return results

def process_prediction_chunk(model, vectorizer, chunk, text_column,
                            remove_stopwords, perform_stemming, 
                            perform_lemmatization, handle_negations):
    """Helper function to process a single chunk of prediction data"""
    # Preprocess text
    chunk['processed_text'] = chunk[text_column].apply(
        lambda x: preprocess_text(
            x, 
            remove_stopwords=remove_stopwords,
            perform_stemming=perform_stemming,
            perform_lemmatization=perform_lemmatization,
            handle_negations=handle_negations
        )
    )
    
    # Transform text to feature vectors
    X = vectorizer.transform(chunk['processed_text'])
    
    # Make predictions
    predictions = model.predict(X)
    chunk['prediction'] = predictions
    
    # Get probabilities if model supports it
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)
        
        # For each class, add probability column
        for i, class_name in enumerate(model.classes_):
            chunk[f'confidence_{class_name}'] = probabilities[:, i]
    
    return chunk

def save_chunked_model(model, vectorizer, metrics, model_name, directory="models"):
    """
    Save a model trained on chunked data.
    
    Args:
        model: The trained model
        vectorizer: The feature vectorizer
        metrics (dict): Model metrics
        model_name (str): Name to save the model as
        directory (str): Directory to save in
        
    Returns:
        str: Path to the saved model directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Create model directory
    model_dir = os.path.join(directory, model_name.lower().replace(" ", "_"))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Save model
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)
    
    # Save vectorizer
    vectorizer_path = os.path.join(model_dir, "vectorizer.joblib")
    joblib.dump(vectorizer, vectorizer_path)
    
    # Save metrics
    metrics_path = os.path.join(model_dir, "metrics.joblib")
    joblib.dump(metrics, metrics_path)
    
    # Create info file
    info = {
        'name': model_name,
        'type': type(model).__name__,
        'n_features': getattr(vectorizer, 'n_features', None),
        'accuracy': metrics.get('accuracy'),
        'created_at': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    info_path = os.path.join(model_dir, "info.joblib")
    joblib.dump(info, info_path)
    
    return model_dir

def load_chunked_model(model_dir):
    """
    Load a model trained on chunked data.
    
    Args:
        model_dir (str): Directory containing the model
        
    Returns:
        tuple: (model, vectorizer, metrics, info)
    """
    model_path = os.path.join(model_dir, "model.joblib")
    vectorizer_path = os.path.join(model_dir, "vectorizer.joblib")
    metrics_path = os.path.join(model_dir, "metrics.joblib")
    info_path = os.path.join(model_dir, "info.joblib")
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    metrics = None
    if os.path.exists(metrics_path):
        metrics = joblib.load(metrics_path)
    
    info = None
    if os.path.exists(info_path):
        info = joblib.load(info_path)
    
    return model, vectorizer, metrics, info

def get_chunked_models(directory="models"):
    """
    Get a list of available chunked models.
    
    Args:
        directory (str): Directory to look in
        
    Returns:
        list: List of model directories
    """
    if not os.path.exists(directory):
        return []
    
    # Look for directories that contain model files
    model_dirs = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            model_file = os.path.join(item_path, "model.joblib")
            if os.path.exists(model_file):
                model_dirs.append(item_path)
    
    return model_dirs 
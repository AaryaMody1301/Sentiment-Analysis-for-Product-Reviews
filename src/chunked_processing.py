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
                       chunksize=20000, test_size=0.2, 
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
    
    # Initialize the vectorizer - use a larger n_features for better accuracy with large datasets
    vectorizer = HashingVectorizer(n_features=n_features, alternate_sign=False)
    
    # Get file size for logging
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    # Monitor memory usage
    try:
        import psutil
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        initial_memory = 0
    
    if callback:
        callback(0, f"Starting processing of {file_size_mb:.1f}MB file. Initial memory: {initial_memory:.1f}MB")
    
    # First, detect columns if not provided using a sample
    sample_chunk = pd.read_csv(file_path, nrows=min(10000, chunksize))
    first_chunk_size = len(sample_chunk)
    
    if text_column is None or sentiment_column is None:
        detected_text_col, detected_sentiment_col = detect_columns(sample_chunk)
        text_column = text_column or detected_text_col
        sentiment_column = sentiment_column or detected_sentiment_col
    
    if callback:
        callback(0.02, f"Processing file: detected columns - text: {text_column}, sentiment: {sentiment_column}")
        
    # Estimate total rows based on file size and first chunk
    avg_row_size = os.path.getsize(file_path) / first_chunk_size if first_chunk_size > 0 else 0
    estimated_total_rows = int(os.path.getsize(file_path) / avg_row_size) if avg_row_size > 0 else 0
    
    # Scan the entire file to find all possible sentiment classes (important to prevent errors)
    all_sentiment_classes = set()
    scan_chunks = 0
    
    if callback:
        callback(0.05, "Scanning file to identify all sentiment classes...")
    
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        scan_chunks += 1
        if callback and scan_chunks % 5 == 0:
            callback(0.05, f"Scanning for classes... (chunk {scan_chunks})")
        
        # Skip if sentiment column doesn't exist
        if sentiment_column not in chunk.columns:
            continue
            
        # Handle missing values
        chunk = chunk.dropna(subset=[sentiment_column])
        
        # Normalize sentiment labels
        chunk = normalize_sentiment_labels(chunk, sentiment_column)
        
        # Add sentiment values to our set
        all_sentiment_classes.update(chunk[sentiment_column].unique())
    
    # Convert to sorted list for consistent order
    all_sentiment_classes = sorted(list(all_sentiment_classes))
    
    if callback:
        callback(0.1, f"Found {len(all_sentiment_classes)} sentiment classes: {all_sentiment_classes}")
    
    # Check if we have enough classes for classification
    if len(all_sentiment_classes) < 2:
        if callback:
            callback(1.0, f"Error: Only found {len(all_sentiment_classes)} sentiment classes. Need at least 2 for classification.")
        raise ValueError(f"Not enough sentiment classes found. Need at least 2, found: {all_sentiment_classes}")
    
    # Now initialize the model with knowledge of all possible classes
    model = MultinomialNB(alpha=1.5)
    
    # Count chunks for progress
    total_chunks = max(1, estimated_total_rows // chunksize)
    processed_chunks = 0
    processed_rows = 0
    
    # These will store all test data for final evaluation
    test_texts = []
    test_labels = []
    
    # First model fit flag
    is_first_fit = True
    
    # Process chunks
    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunksize)):
        chunk_start_time = time.time()
        processed_chunks += 1
        processed_rows += len(chunk)
        
        # Update progress more frequently for large files
        if callback:
            progress = min(0.95, 0.1 + (processed_rows / max(processed_rows, estimated_total_rows)) * 0.85)
            callback(progress, f"Processing chunk {processed_chunks}/{total_chunks} ({len(chunk):,} rows)")
        
        # Ensure the required columns exist
        if text_column not in chunk.columns or sentiment_column not in chunk.columns:
            if callback:
                callback(progress, f"Error: Required columns not found in chunk {i+1}")
            raise ValueError(f"Required columns not found: {text_column}, {sentiment_column}")
        
        # Check for missing values
        text_missing = chunk[text_column].isnull().sum()
        sentiment_missing = chunk[sentiment_column].isnull().sum()
        
        if text_missing > 0 or sentiment_missing > 0:
            if callback:
                callback(progress, f"Warning: Found missing values - text: {text_missing}, sentiment: {sentiment_missing}")
            # Drop rows with missing values to avoid errors
            chunk = chunk.dropna(subset=[text_column, sentiment_column])
        
        # Handle non-string text columns
        if chunk[text_column].dtype != 'object':
            chunk[text_column] = chunk[text_column].astype(str)
        
        # Preprocess text
        try:
            # Create a copy to avoid SettingWithCopyWarning
            chunk = chunk.copy()
            
            # Use loc to avoid SettingWithCopyWarning
            chunk.loc[:, 'processed_text'] = chunk[text_column].apply(
                lambda x: preprocess_text(
                    x, 
                    remove_stopwords=remove_stopwords,
                    perform_stemming=perform_stemming,
                    perform_lemmatization=perform_lemmatization,
                    handle_negations=handle_negations
                )
            )
        except Exception as e:
            if callback:
                callback(progress, f"Error preprocessing text: {str(e)}")
            raise
        
        # Normalize sentiment labels
        chunk = normalize_sentiment_labels(chunk, sentiment_column)
        
        # Split into train and test
        X = chunk['processed_text'].values
        y = chunk[sentiment_column].values
        
        # Skip empty chunks
        if len(X) == 0:
            continue
        
        # Filter out any rows with sentiment values not in our predefined set
        # This ensures consistent classes across all chunks
        valid_indices = np.isin(y, all_sentiment_classes)
        if not all(valid_indices):
            if callback:
                callback(progress, f"Warning: Filtered out {sum(~valid_indices)} rows with unknown sentiment values")
            X = X[valid_indices]
            y = y[valid_indices]
            
            # Skip if nothing left
            if len(X) == 0:
                continue
        
        # For the first chunk, save some data for testing
        if i == 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            test_texts.extend(X_test)
            test_labels.extend(y_test)
        else:
            # For subsequent chunks, use a smaller test set
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=min(0.1, test_size), random_state=42)
            test_texts.extend(X_test)
            test_labels.extend(y_test)
        
        # Transform text to feature vectors
        X_train_vec = vectorizer.transform(X_train)
        
        # Partial fit the model with ALL possible classes (important!)
        try:
            model.partial_fit(X_train_vec, y_train, classes=all_sentiment_classes)
            is_first_fit = False
        except Exception as e:
            if callback:
                callback(progress, f"Error during model fitting: {str(e)}")
            raise
        
        # Update progress with timing information
        chunk_time = time.time() - chunk_start_time
        if callback:
            try:
                current_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                memory_diff = current_memory - initial_memory
            except:
                current_memory = 0
                memory_diff = 0
                
            callback(progress, f"Chunk {processed_chunks} processed in {chunk_time:.1f}s. "
                            f"Memory: {current_memory:.1f}MB (+{memory_diff:.1f}MB)")
    
    # Final evaluation on all test data
    if callback:
        callback(0.97, "Evaluating model on test data...")
    
    # Check if we have test data
    if len(test_texts) == 0 or len(test_labels) == 0:
        if callback:
            callback(0.98, "Warning: No test data available for evaluation.")
        return model, vectorizer, {"accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0, "confusion_matrix": [[0, 0], [0, 0]]}
    
    # Ensure test labels are also in our class list
    valid_test_indices = np.isin(test_labels, all_sentiment_classes)
    if not all(valid_test_indices):
        if callback:
            callback(0.98, f"Warning: Filtered out {sum(~valid_test_indices)} test samples with unknown sentiment values")
        test_texts = [test_texts[i] for i in range(len(test_texts)) if valid_test_indices[i]]
        test_labels = [test_labels[i] for i in range(len(test_labels)) if valid_test_indices[i]]
    
    # Check if we have enough test data left
    if len(test_texts) < 2:
        if callback:
            callback(0.98, "Warning: Not enough test data available after filtering. Using dummy metrics.")
        return model, vectorizer, {
            "accuracy": 0, 
            "precision": 0, 
            "recall": 0, 
            "f1_score": 0, 
            "confusion_matrix": [[0, 0], [0, 0]], 
            "classes": all_sentiment_classes
        }
    
    # Transform test data
    X_test_vec = vectorizer.transform(test_texts)
    
    # Predict
    y_pred = model.predict(X_test_vec)
    
    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(test_labels, y_pred),
        "precision": precision_score(test_labels, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(test_labels, y_pred, average='weighted', zero_division=0),
        "f1_score": f1_score(test_labels, y_pred, average='weighted', zero_division=0),
        "confusion_matrix": confusion_matrix(test_labels, y_pred).tolist(),
        "classes": all_sentiment_classes
    }
    
    total_time = time.time() - start_time
    if callback:
        callback(1.0, f"Processing complete in {total_time:.1f}s. Processed {processed_rows:,} rows.")
    
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
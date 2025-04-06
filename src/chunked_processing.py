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
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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
        is_memory_monitoring = True
    except ImportError:
        initial_memory = 0
        is_memory_monitoring = False
    
    # EXTREME OPTIMIZATION: For very large files (>200MB), use more aggressive settings
    extreme_optimization = file_size_mb > 200
    if extreme_optimization and callback:
        callback(0, f"Large file detected ({file_size_mb:.1f}MB). Using extreme optimization mode.")
        # Force reduced settings for very large files
        if perform_lemmatization:
            callback(0, "Disabling lemmatization for better performance with large file.")
            perform_lemmatization = False
        # Increase chunk size for faster processing but watch memory
        max_memory_percent = 0.6  # Use at most 60% of available RAM
    
    if callback:
        callback(0, f"Starting processing of {file_size_mb:.1f}MB file. Initial memory: {initial_memory:.1f}MB")
    
    # First, detect columns if not provided using a sample
    # Use smaller sample for very large files
    sample_size = min(1000, chunksize) if extreme_optimization else min(5000, chunksize)
    sample_chunk = pd.read_csv(file_path, nrows=sample_size)
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
    
    # PERFORMANCE OPTIMIZATION: Scan a subset of the file rather than entire file to identify sentiment classes
    # For large files, scan even fewer chunks and focus on beginning, middle and end
    all_sentiment_classes = set()
    scan_chunks = 0
    max_scan_chunks = 5 if extreme_optimization else 10  # Limit scanning to fewer chunks for large files
    
    if callback:
        callback(0.05, "Scanning sample of file to identify sentiment classes...")
    
    # For very large files, we'll scan the beginning, middle, and end instead of sequential chunks
    if extreme_optimization:
        # Get file size and estimate total chunks
        total_chunks = estimated_total_rows // chunksize
        
        # Positions to sample: beginning, 25%, 50%, 75%, end
        sample_positions = [
            0,  # Beginning 
            max(1, total_chunks // 4),  # ~25%
            max(1, total_chunks // 2),  # ~50%
            max(1, (total_chunks * 3) // 4),  # ~75%
            max(1, total_chunks - 1)  # End
        ]
        
        # Remove duplicates (in case total_chunks is small)
        sample_positions = sorted(list(set(sample_positions)))
        
        for pos in sample_positions:
            if callback:
                callback(0.05, f"Scanning for classes at position {pos+1}/{total_chunks}...")
            
            # Skip to the position
            try:
                # Calculate rows to skip
                skip_rows = pos * chunksize
                
                # Read the chunk
                scan_chunk = pd.read_csv(file_path, 
                                       skiprows=range(1, skip_rows + 1) if skip_rows > 0 else None,
                                       nrows=chunksize)
                
                # Check if sentiment column exists
                if sentiment_column not in scan_chunk.columns:
                    continue
                
                # Process the chunk
                scan_chunk = scan_chunk.dropna(subset=[sentiment_column])
                scan_chunk = normalize_sentiment_labels(scan_chunk, sentiment_column)
                all_sentiment_classes.update(scan_chunk[sentiment_column].unique())
                
                scan_chunks += 1
                
                # If we have at least 2 classes, we can potentially stop scanning
                if len(all_sentiment_classes) >= 2 and scan_chunks >= 3:
                    break
            except Exception as e:
                if callback:
                    callback(0.05, f"Error scanning at position {pos}: {str(e)}")
    else:
        # Original sequential scanning logic for smaller files
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            scan_chunks += 1
            if callback and scan_chunks % 2 == 0:
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
            
            # PERFORMANCE OPTIMIZATION: Stop scanning after max_scan_chunks
            if scan_chunks >= max_scan_chunks:
                break
    
    # If we have at least 2 classes, we can proceed, otherwise scan additional chunks
    if len(all_sentiment_classes) < 2 and estimated_total_rows > max_scan_chunks * chunksize:
        if callback:
            callback(0.05, "Not enough classes found in sample, scanning additional chunks...")
        
        # Scan a few more chunks at different positions
        additional_positions = [
            chunksize * max_scan_chunks,  # Right after our previous scan
            estimated_total_rows // 2,    # Middle of file
            max(0, estimated_total_rows - chunksize * 2)  # Near the end
        ]
        
        for skip_rows in additional_positions:
            if callback:
                callback(0.05, f"Extended scanning at row {skip_rows}...")
            
            try:
                # Skip to this position in the file
                additional_chunk = pd.read_csv(
                    file_path, 
                    skiprows=range(1, skip_rows) if skip_rows > 0 else None,
                    nrows=chunksize
                )
                
                # Skip if sentiment column doesn't exist
                if sentiment_column not in additional_chunk.columns:
                    continue
                    
                # Handle missing values
                additional_chunk = additional_chunk.dropna(subset=[sentiment_column])
                
                # Normalize sentiment labels
                additional_chunk = normalize_sentiment_labels(additional_chunk, sentiment_column)
                
                # Add sentiment values to our set
                all_sentiment_classes.update(additional_chunk[sentiment_column].unique())
                
                # If we have at least 2 classes, we can stop scanning
                if len(all_sentiment_classes) >= 2:
                    break
            except Exception as e:
                if callback:
                    callback(0.05, f"Error during extended scanning: {str(e)}")
    
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
    
    # Count chunks for progress tracking
    total_chunks = max(1, estimated_total_rows // chunksize)
    processed_chunks = 0
    processed_rows = 0
    
    # These will store a sample of test data for final evaluation
    # PERFORMANCE OPTIMIZATION: Limit test data size to reduce memory usage
    max_test_samples = 5000 if extreme_optimization else 10000
    test_texts = []
    test_labels = []
    
    # First model fit flag
    is_first_fit = True
    
    # PERFORMANCE OPTIMIZATION: Cached lemmatizer to avoid recreating for each text
    lemmatizer = None
    if perform_lemmatization:
        lemmatizer = WordNetLemmatizer()
    
    # Define a batch preprocess function that uses vectorized operations where possible
    def batch_preprocess(texts, lemmatizer_obj=None):
        """Preprocess a batch of texts more efficiently"""
        # Handle missing values and convert to strings
        processed_texts = []
        
        for text in texts:
            # Skip empty or missing values
            if pd.isna(text) or not isinstance(text, str):
                try:
                    if not isinstance(text, str):
                        text = str(text)
                except:
                    processed_texts.append("")
                    continue
            
            # Handle empty strings
            if not text.strip():
                processed_texts.append("")
                continue
            
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
            if perform_lemmatization and lemmatizer_obj:
                # Better lemmatization with POS tagging
                lemmatized_tokens = []
                for token in tokens:
                    # Try lemmatizing as verb first, then as noun
                    lemmatized_verb = lemmatizer_obj.lemmatize(token, pos='v')
                    # If lemmatizing as verb changes the word, use that
                    if lemmatized_verb != token:
                        lemmatized_tokens.append(lemmatized_verb)
                    else:
                        # Otherwise use noun lemmatization (default)
                        lemmatized_tokens.append(lemmatizer_obj.lemmatize(token))
                tokens = lemmatized_tokens
            
            # Join tokens back into text
            processed_texts.append(' '.join(tokens))
        
        return processed_texts
    
    # EXTREME OPTIMIZATION: Process a limited number of chunks for very large files
    max_chunks_to_process = None
    if extreme_optimization:
        # For extremely large files, limit to processing around 20-30% of the file
        # This significantly reduces processing time while still getting a good model
        max_chunks_to_process = max(10, total_chunks // 4)
        if callback:
            callback(0.1, f"Extreme optimization: Will process {max_chunks_to_process} chunks out of {total_chunks} total")
    
    # Process chunks
    chunk_reader = pd.read_csv(file_path, chunksize=chunksize)
    for i, chunk in enumerate(chunk_reader):
        # For extreme optimization, process a subset of chunks distributed throughout the file
        if extreme_optimization and max_chunks_to_process:
            # Process first few chunks, middle chunks, and last few chunks
            first_chunks = max_chunks_to_process // 3
            last_chunks = max_chunks_to_process // 3
            
            if i > first_chunks and i < (total_chunks - last_chunks):
                # Only process every Nth chunk in the middle of the file
                stride = (total_chunks - first_chunks - last_chunks) // (max_chunks_to_process - first_chunks - last_chunks)
                if stride > 1 and (i - first_chunks) % stride != 0:
                    continue
        
        chunk_start_time = time.time()
        processed_chunks += 1
        processed_rows += len(chunk)
        
        # Update progress more frequently for large files
        if callback:
            progress = min(0.95, 0.1 + (processed_chunks / max(max_chunks_to_process or total_chunks, total_chunks)) * 0.85)
            callback(progress, f"Processing chunk {processed_chunks}/{total_chunks} ({len(chunk):,} rows)")
        
        # Check available memory before processing this chunk
        if is_memory_monitoring and extreme_optimization:
            try:
                current_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                available_memory = psutil.virtual_memory().available / (1024 * 1024)
                total_memory = psutil.virtual_memory().total / (1024 * 1024)
                
                # If we're using too much memory, skip some chunks
                memory_usage_percent = 1 - (available_memory / total_memory)
                if memory_usage_percent > max_memory_percent:
                    if callback:
                        callback(progress, f"Memory usage high ({memory_usage_percent:.1%}), skipping some chunks...")
                    
                    # Skip the next few chunks to reduce memory pressure
                    for _ in range(2):  # Skip 2 chunks
                        try:
                            next(chunk_reader)
                            processed_chunks += 1
                        except StopIteration:
                            break
            except Exception as e:
                # If memory monitoring fails, continue anyway
                pass
        
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
        
        # PERFORMANCE OPTIMIZATION: Use batch preprocessing instead of row-by-row
        try:
            # EXTREME OPTIMIZATION: For very large files, sample rows from the chunk
            if extreme_optimization and len(chunk) > 1000:
                # Take a random sample of 50% of the rows
                sample_indices = np.random.choice(len(chunk), size=len(chunk)//2, replace=False)
                text_sample = chunk[text_column].values[sample_indices]
                sentiment_sample = chunk[sentiment_column].values[sample_indices]
                
                # Preprocess the sampled texts
                processed_texts = batch_preprocess(text_sample, lemmatizer)
                chunk_y = sentiment_sample
            else:
                # Preprocess all texts in the chunk
                processed_texts = batch_preprocess(chunk[text_column].values, lemmatizer)
                chunk_y = chunk[sentiment_column].values
            
        except Exception as e:
            if callback:
                callback(progress, f"Error preprocessing text: {str(e)}")
            raise
        
        # Normalize sentiment labels
        chunk = normalize_sentiment_labels(chunk, sentiment_column)
        
        # Get the sentiment labels if we didn't sample
        if not extreme_optimization or len(chunk) <= 1000:
            chunk_y = chunk[sentiment_column].values
        
        # Skip empty chunks
        if len(processed_texts) == 0:
            continue
        
        # Filter out any rows with sentiment values not in our predefined set
        # This ensures consistent classes across all chunks
        valid_indices = np.isin(chunk_y, all_sentiment_classes)
        if not all(valid_indices):
            if callback:
                callback(progress, f"Warning: Filtered out {sum(~valid_indices)} rows with unknown sentiment values")
            processed_texts = [processed_texts[i] for i in range(len(processed_texts)) if valid_indices[i]]
            chunk_y = chunk_y[valid_indices]
            
            # Skip if nothing left
            if len(processed_texts) == 0:
                continue
        
        # For the first chunk, save some data for testing
        if i == 0:
            # Take a smaller test set for very large files
            local_test_size = min(0.1, test_size) if extreme_optimization else test_size
            X_train, X_test, y_train, y_test = train_test_split(processed_texts, chunk_y, test_size=local_test_size, random_state=42)
            
            # PERFORMANCE OPTIMIZATION: Limit test set size
            test_idx = np.random.choice(len(X_test), min(len(X_test), max_test_samples // total_chunks), replace=False)
            test_texts.extend([X_test[j] for j in test_idx])
            test_labels.extend([y_test[j] for j in test_idx])
        else:
            # For subsequent chunks, use a smaller test set or none if we have enough
            if len(test_texts) < max_test_samples:
                local_test_size = min(0.05, test_size) if extreme_optimization else min(0.05, test_size)
                X_train, X_test, y_train, y_test = train_test_split(processed_texts, chunk_y, test_size=local_test_size, random_state=42)
                samples_needed = max_test_samples - len(test_texts)
                test_idx = np.random.choice(len(X_test), min(len(X_test), samples_needed), replace=False)
                test_texts.extend([X_test[j] for j in test_idx])
                test_labels.extend([y_test[j] for j in test_idx])
            else:
                X_train, y_train = processed_texts, chunk_y
        
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
                # PERFORMANCE OPTIMIZATION: Explicitly request garbage collection after each chunk
                import gc
                gc.collect()
                
                current_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                memory_diff = current_memory - initial_memory
            except:
                current_memory = 0
                memory_diff = 0
                
            callback(progress, f"Chunk {processed_chunks} processed in {chunk_time:.1f}s. "
                        f"Memory: {current_memory:.1f}MB (+{memory_diff:.1f}MB)")
        
        # EXTREME OPTIMIZATION: early stopping if we've processed enough chunks
        if extreme_optimization and max_chunks_to_process and processed_chunks >= max_chunks_to_process:
            if callback:
                callback(0.9, f"Processed {processed_chunks} chunks (early stopping to reduce memory usage)")
            break
    
    # Final evaluation on all test data
    if callback:
        callback(0.97, f"Evaluating model on {len(test_texts)} test samples...")
    
    # Check if we have test data
    if len(test_texts) == 0 or len(test_labels) == 0:
        if callback:
            callback(0.98, "Warning: No test data available for evaluation.")
        return model, vectorizer, {"accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0, "confusion_matrix": [[0, 0], [0, 0]]}
    
    # Transform test texts to feature vectors
    X_test_vec = vectorizer.transform(test_texts)
    
    # Make predictions
    y_pred = model.predict(X_test_vec)
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(test_labels, y_pred),
        "precision": precision_score(test_labels, y_pred, average='weighted'),
        "recall": recall_score(test_labels, y_pred, average='weighted'),
        "f1_score": f1_score(test_labels, y_pred, average='weighted'),
        "confusion_matrix": confusion_matrix(test_labels, y_pred).tolist()
    }
    
    # Log completion time
    total_time = time.time() - start_time
    if callback:
        callback(1.0, f"Processing complete in {total_time:.1f}s. Accuracy: {metrics['accuracy']:.4f}")
    
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
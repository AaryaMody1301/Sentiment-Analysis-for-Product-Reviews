import pandas as pd
import numpy as np
import os
import joblib
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from src.nlp_processing import create_tfidf_vectorizer, create_hashing_vectorizer, compute_class_weights

def get_available_models():
    """
    Return a dictionary of available models for sentiment analysis.
    
    Returns:
        dict: Dictionary with model names as keys and model instances as values.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, solver='liblinear'),
        "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
        "Linear SVM": LinearSVC(max_iter=2000, C=1.0, dual=False),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }

def get_hyperparameter_grid(model_name):
    """
    Return a dictionary of hyperparameters for grid search.
    
    Args:
        model_name (str): Name of the model.
        
    Returns:
        dict: Dictionary with parameter names and values for grid search.
    """
    # Common parameters for TfidfVectorizer
    tfidf_params = {
        'vectorizer__max_features': [3000, 5000],
        'vectorizer__ngram_range': [(1, 1), (1, 2)],
        'vectorizer__min_df': [2, 5],
        'vectorizer__max_df': [0.8, 0.9]
    }
    
    # Model-specific parameters
    if model_name == "Multinomial Naive Bayes":
        model_params = {
            'classifier__alpha': [0.1, 0.5, 1.0, 2.0]
        }
    elif model_name == "Logistic Regression":
        model_params = {
            'classifier__C': [0.1, 0.5, 1.0, 5.0, 10.0],
            'classifier__solver': ['liblinear', 'saga'],
            'classifier__penalty': ['l1', 'l2']
        }
    elif model_name == "Linear SVM":
        model_params = {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__loss': ['hinge', 'squared_hinge'],
            'classifier__dual': [False]
        }
    elif model_name == "Random Forest":
        model_params = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20, 30]
        }
    else:
        return {}
    
    # Combine TfidfVectorizer and model parameters
    return {**tfidf_params, **model_params}

def train_model(df, text_column, sentiment_column, model_name="Logistic Regression", 
                test_size=0.2, random_state=42, tune_hyperparameters=False, 
                use_hashing_vectorizer=False, max_features=5000, ngram_range=(1, 2),
                handle_class_imbalance=True, n_jobs=-1, cv=3, verbose=1):
    """
    Train a sentiment analysis model using TfidfVectorizer and the selected classifier.
    
    Args:
        df (pd.DataFrame): DataFrame containing text and sentiment columns.
        text_column (str): Name of the column containing review text.
        sentiment_column (str): Name of the column containing sentiment labels.
        model_name (str): Name of the model to use. Default is Logistic Regression.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.
        tune_hyperparameters (bool): Whether to perform hyperparameter tuning using GridSearchCV.
        use_hashing_vectorizer (bool): Whether to use HashingVectorizer instead of TfidfVectorizer.
        max_features (int): Maximum number of features for TfidfVectorizer.
        ngram_range (tuple): Range of n-grams for vectorizer.
        handle_class_imbalance (bool): Whether to handle class imbalance using class weights.
        n_jobs (int): Number of jobs for parallel processing (-1 for all cores).
        cv (int): Number of folds for cross-validation.
        verbose (int): Verbosity level for grid search.
        
    Returns:
        tuple: (model, X_train, X_test, y_train, y_test, best_params)
    """
    # Start timing
    start_time = time.time()
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_column], 
        df[sentiment_column], 
        test_size=test_size, 
        random_state=random_state,
        stratify=df[sentiment_column]
    )
    
    # Get the classifier based on model_name
    models = get_available_models()
    classifier = models.get(model_name, LogisticRegression(max_iter=1000, C=1.0, solver='liblinear'))
    
    # Handle class imbalance if specified
    if handle_class_imbalance and hasattr(classifier, 'class_weight'):
        class_weights = compute_class_weights(y_train)
        if hasattr(classifier, 'set_params'):
            classifier.set_params(class_weight='balanced')
    
    # Choose the appropriate vectorizer
    if use_hashing_vectorizer:
        vectorizer = create_hashing_vectorizer(n_features=2**18, ngram_range=ngram_range)
    else:
        vectorizer = create_tfidf_vectorizer(max_features=max_features, ngram_range=ngram_range)
    
    # Create a pipeline with vectorizer and the selected classifier
    model = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])
    
    # Perform hyperparameter tuning if requested
    if tune_hyperparameters:
        print(f"Starting hyperparameter tuning for {model_name}...")
        param_grid = get_hyperparameter_grid(model_name)
        
        if param_grid:
            # If dataset is very large, use a subset for grid search
            subsample_for_tuning = len(X_train) > 20000
            
            if subsample_for_tuning:
                # Use a random subset of 10,000 samples for tuning
                print("Using a subset of 10,000 samples for hyperparameter tuning...")
                X_tune, _, y_tune, _ = train_test_split(
                    X_train, y_train, train_size=10000, 
                    random_state=random_state, stratify=y_train
                )
            else:
                X_tune, y_tune = X_train, y_train
            
            # Create grid search
            grid_search = GridSearchCV(
                model,
                param_grid,
                cv=cv,
                n_jobs=n_jobs,
                scoring='f1_weighted',
                verbose=verbose,
                return_train_score=True
            )
            
            # Perform grid search
            print(f"Grid search with {len(param_grid)} parameters...")
            tuning_start = time.time()
            grid_search.fit(X_tune, y_tune)
            tuning_time = time.time() - tuning_start
            print(f"Hyperparameter tuning completed in {tuning_time:.2f}s")
            
            # Get the best model
            model = grid_search.best_estimator_
            
            # Return the best parameters for display
            best_params = grid_search.best_params_
            
            # If we used a subset for tuning, retrain on the full dataset
            if subsample_for_tuning:
                print("Retraining final model on full training set...")
                model.fit(X_train, y_train)
            
            # Print grid search results
            print(f"Best parameters: {best_params}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            
            # Return the model with best parameters
            return model, X_train, X_test, y_train, y_test, best_params
    
    # Train the model (if not using grid search)
    print(f"Training {model_name} model...")
    training_start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - training_start
    print(f"Model training completed in {training_time:.2f}s")
    
    # Return None for best_params if not using grid search
    return model, X_train, X_test, y_train, y_test, None

def evaluate_model(model, X_test, y_test, target_accuracy=0.92):
    """
    Evaluate the trained model on the test set.
    
    Args:
        model: Trained model pipeline.
        X_test: Test features.
        y_test: True labels for test data.
        target_accuracy: Target accuracy threshold (default 0.92)
        
    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Get per-class metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Check if target accuracy is achieved
    target_achieved = accuracy >= target_accuracy
    
    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': report,
        'target_achieved': target_achieved
    }
    
    # Print results
    print(f"\nModel Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}" + (" ✓" if target_achieved else " ✗"))
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    if target_achieved:
        print(f"\n✅ Success! Target accuracy of {target_accuracy:.2f} achieved.")
    else:
        print(f"\n❌ Target accuracy of {target_accuracy:.2f} not achieved. Consider:")
        print("   - Additional hyperparameter tuning")
        print("   - Using bigrams (ngram_range=(1,2))")
        print("   - Adding more preprocessing steps")
        print("   - Balancing the dataset")
    
    return metrics

def predict_sentiment(model, text):
    """
    Predict the sentiment of a given text.
    
    Args:
        model: Trained model pipeline.
        text (str): Text to predict sentiment for.
        
    Returns:
        str: Predicted sentiment label.
    """
    # Make prediction
    prediction = model.predict([text])[0]
    
    return prediction

def predict_sentiment_with_probability(model, text):
    """
    Predict the sentiment of a given text with probability.
    
    Args:
        model: Trained model pipeline.
        text (str): Text to predict sentiment for.
        
    Returns:
        tuple: (predicted_sentiment, probability_dict) where probability_dict
               maps each class to its probability
    """
    # Check if model supports predict_proba (not all models do)
    if hasattr(model, 'predict_proba'):
        # Get prediction and probabilities
        prediction = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]
        
        # Create dictionary mapping classes to probabilities
        classes = model.classes_
        probability_dict = {cls: prob for cls, prob in zip(classes, probabilities)}
        
        return prediction, probability_dict
    else:
        # For models without predict_proba (like LinearSVC)
        prediction = model.predict([text])[0]
        
        try:
            # Some SVC models have decision_function
            if hasattr(model, 'decision_function'):
                decision_scores = model.decision_function([text])[0]
                if isinstance(decision_scores, np.ndarray) and len(decision_scores) > 1:
                    # Multi-class case
                    classes = model.classes_
                    total = sum(np.exp(decision_scores))
                    pseudo_probs = {c: np.exp(s)/total for c, s in zip(classes, decision_scores)}
                    return prediction, pseudo_probs
                else:
                    # Binary case
                    score = decision_scores
                    prob = 1 / (1 + np.exp(-score))
                    classes = model.classes_
                    return prediction, {classes[0]: 1-prob, classes[1]: prob}
            else:
                # No probability/score method available
                return prediction, {prediction: 1.0}
        except:
            # Fallback
            return prediction, {prediction: 1.0}

def get_feature_importance(model, top_n=20):
    """
    Get the most important features for sentiment classification.
    
    Args:
        model: Trained model pipeline.
        top_n (int): Number of top features to return.
        
    Returns:
        dict: Dictionary containing feature importances per sentiment.
    """
    # Check if model is a pipeline
    if not hasattr(model, 'named_steps'):
        return None
    
    # Get vectorizer and classifier from pipeline
    try:
        if 'vectorizer' in model.named_steps:
            vectorizer = model.named_steps['vectorizer']
        elif 'tfidf' in model.named_steps:
            vectorizer = model.named_steps['tfidf']
        else:
            # Unknown vectorizer name
            return None
        
        if 'classifier' in model.named_steps:
            classifier = model.named_steps['classifier']
        else:
            # Unknown classifier name
            return None
    except:
        return None
    
    # Check if vectorizer is a HashingVectorizer
    if isinstance(vectorizer, HashingVectorizer):
        # Cannot get feature names from a HashingVectorizer
        return None
    
    # Get feature names
    try:
        feature_names = vectorizer.get_feature_names_out()
    except:
        try:
            feature_names = vectorizer.get_feature_names()
        except:
            return None
    
    # Get feature importances based on classifier type
    feature_importances = {}
    
    # For Logistic Regression (has coef_)
    if hasattr(classifier, 'coef_'):
        coefficients = classifier.coef_
        
        # If binary classification, convert to 2D
        if len(coefficients.shape) == 1:
            coefficients = coefficients.reshape(1, -1)
        
        # Get classes
        if hasattr(classifier, 'classes_'):
            classes = classifier.classes_
        else:
            classes = [f"Class {i}" for i in range(coefficients.shape[0])]
        
        # Get top features for each class
        for i, cls in enumerate(classes):
            # Get coefficients for this class
            coefs = coefficients[i]
            
            # Get top positive and negative features
            top_positive_idx = np.argsort(coefs)[-top_n:][::-1]
            top_negative_idx = np.argsort(coefs)[:top_n]
            
            # Create feature-value pairs
            top_positive = [(feature_names[idx], coefs[idx]) for idx in top_positive_idx]
            top_negative = [(feature_names[idx], coefs[idx]) for idx in top_negative_idx]
            
            # Store
            feature_importances[cls] = {
                'positive': top_positive,
                'negative': top_negative
            }
        
        # Also add overall importance (absolute value)
        if len(classes) > 1:
            # For multiclass, take the mean of absolute values across classes
            mean_importance = np.mean(np.abs(coefficients), axis=0)
            top_idx = np.argsort(mean_importance)[-top_n:][::-1]
            feature_importances['all'] = [(feature_names[idx], mean_importance[idx]) for idx in top_idx]
    
    # For ensemble methods (has feature_importances_)
    elif hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        indices = np.argsort(importances)[-top_n:][::-1]
        feature_importances['all'] = [(feature_names[i], importances[i]) for i in indices]
    
    # Fallback - no feature importance available
    else:
        return None
    
    return feature_importances

def save_model(model, model_name, models_dir="models"):
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model to save.
        model_name (str): Name for the saved model.
        models_dir (str): Directory to save the model.
        
    Returns:
        str: Path to the saved model.
    """
    # Create models directory if it doesn't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Define file path
    model_path = os.path.join(models_dir, f"{model_name}.joblib")
    
    # Save the model
    joblib.dump(model, model_path)
    
    print(f"Model saved to {model_path}")
    
    return model_path

def load_model(model_path):
    """
    Load a trained model from disk.
    
    Args:
        model_path (str): Path to the model file.
        
    Returns:
        object: Loaded model.
    """
    # Check if file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the model
    model = joblib.load(model_path)
    
    return model

def get_saved_models(models_dir="models"):
    """
    Get a list of saved models.
    
    Args:
        models_dir (str): Directory containing saved models.
        
    Returns:
        list: List of model file paths.
    """
    # Check if directory exists
    if not os.path.exists(models_dir):
        return []
    
    # Get model files
    model_files = [os.path.join(models_dir, f) for f in os.listdir(models_dir) 
                   if f.endswith('.joblib')]
    
    return model_files

def batch_predict(model, df, text_column, prediction_column='predicted_sentiment', 
                 confidence_column='confidence', batch_size=1000):
    """
    Make predictions on a DataFrame in batches.
    
    Args:
        model: Trained model pipeline.
        df (pd.DataFrame): DataFrame containing texts to predict.
        text_column (str): Name of the column containing review text.
        prediction_column (str): Name of the column to store predictions.
        confidence_column (str): Name of the column to store confidence scores.
        batch_size (int): Number of texts to predict at once.
        
    Returns:
        pd.DataFrame: DataFrame with predictions and confidence scores.
    """
    # Create a copy of the DataFrame
    result_df = df.copy()
    result_df[prediction_column] = None
    result_df[confidence_column] = None
    
    # Get total number of rows
    n_rows = len(df)
    n_batches = (n_rows + batch_size - 1) // batch_size
    
    print(f"Processing {n_rows} texts in {n_batches} batches...")
    
    # Process each batch
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_rows)
        
        print(f"Processing batch {i+1}/{n_batches} (rows {start_idx}-{end_idx})...")
        
        # Get texts for this batch
        batch_texts = df.iloc[start_idx:end_idx][text_column].values
        
        # Check if model supports predict_proba
        if hasattr(model, 'predict_proba'):
            # Get predictions and probabilities
            batch_predictions = model.predict(batch_texts)
            batch_probabilities = model.predict_proba(batch_texts)
            
            # Get the probability of the predicted class
            batch_confidences = [probs[np.where(model.classes_ == pred)[0][0]] 
                                for pred, probs in zip(batch_predictions, batch_probabilities)]
        else:
            # For models without predict_proba
            batch_predictions = model.predict(batch_texts)
            batch_confidences = [1.0] * len(batch_predictions)
        
        # Store predictions and confidences
        result_df.iloc[start_idx:end_idx, result_df.columns.get_loc(prediction_column)] = batch_predictions
        result_df.iloc[start_idx:end_idx, result_df.columns.get_loc(confidence_column)] = batch_confidences
    
    print("Batch prediction completed.")
    return result_df 
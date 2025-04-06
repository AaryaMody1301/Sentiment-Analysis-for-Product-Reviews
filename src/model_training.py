import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def get_available_models():
    """
    Return a dictionary of available models for sentiment analysis.
    
    Returns:
        dict: Dictionary with model names as keys and model instances as values.
    """
    return {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(max_iter=1000),
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
        'tfidf__max_features': [3000, 5000],
        'tfidf__ngram_range': [(1, 1), (1, 2)]
    }
    
    # Model-specific parameters
    if model_name == "Multinomial Naive Bayes":
        model_params = {
            'classifier__alpha': [0.1, 0.5, 1.0]
        }
    elif model_name == "Logistic Regression":
        model_params = {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__solver': ['liblinear', 'saga']
        }
    elif model_name == "Linear SVM":
        model_params = {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__loss': ['hinge', 'squared_hinge']
        }
    elif model_name == "Random Forest":
        model_params = {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [None, 10, 20]
        }
    else:
        return {}
    
    # Combine TfidfVectorizer and model parameters
    return {**tfidf_params, **model_params}

def train_model(df, text_column, sentiment_column, model_name=None, test_size=0.2, random_state=42, tune_hyperparameters=False):
    """
    Train a sentiment analysis model using TfidfVectorizer and the selected classifier.
    
    Args:
        df (pd.DataFrame): DataFrame containing text and sentiment columns.
        text_column (str): Name of the column containing review text.
        sentiment_column (str): Name of the column containing sentiment labels.
        model_name (str): Name of the model to use. If None, uses MultinomialNB.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.
        tune_hyperparameters (bool): Whether to perform hyperparameter tuning using GridSearchCV.
        
    Returns:
        tuple: (model, X_train, X_test, y_train, y_test)
    """
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
    classifier = models.get(model_name, MultinomialNB())
    
    # Create a pipeline with TfidfVectorizer and the selected classifier
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('classifier', classifier)
    ])
    
    # Perform hyperparameter tuning if requested
    if tune_hyperparameters:
        param_grid = get_hyperparameter_grid(model_name)
        if param_grid:
            grid_search = GridSearchCV(
                model,
                param_grid,
                cv=3,
                n_jobs=-1,
                scoring='f1_weighted',
                verbose=0
            )
            grid_search.fit(X_train, y_train)
            
            # Get the best model
            model = grid_search.best_estimator_
            
            # Return the best parameters for display
            best_params = grid_search.best_params_
            
            # Train the model with the best parameters
            return model, X_train, X_test, y_train, y_test, best_params
    
    # Train the model (if not using grid search)
    model.fit(X_train, y_train)
    
    # Return None for best_params if not using grid search
    return model, X_train, X_test, y_train, y_test, None

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.
    
    Args:
        model: Trained model pipeline.
        X_test: Test features.
        y_test: True labels for test data.
        
    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
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
        # Return dummy probabilities
        probability_dict = {prediction: 1.0}
        
        return prediction, probability_dict

def get_feature_importance(model, top_n=20):
    """
    Get the most important features (words) for each class.
    
    Args:
        model: Trained model pipeline.
        top_n (int): Number of top features to return.
        
    Returns:
        dict: Dictionary with classes as keys and lists of (feature, importance) tuples as values.
    """
    try:
        # Extract the vectorizer and classifier from the pipeline
        vectorizer = model.named_steps['tfidf']
        classifier = model.named_steps['classifier']
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # For Naive Bayes, coefficients are stored differently
        if isinstance(classifier, MultinomialNB):
            # Get log probabilities
            feature_importance = {}
            for i, class_label in enumerate(classifier.classes_):
                # For each class, get the log probability of each feature
                log_probs = classifier.feature_log_prob_[i]
                # Sort and get the highest ones
                sorted_indices = log_probs.argsort()[::-1][:top_n]
                feature_importance[class_label] = [(feature_names[j], log_probs[j]) for j in sorted_indices]
            
            return feature_importance
        
        # For linear models (LogisticRegression, LinearSVC)
        elif hasattr(classifier, 'coef_'):
            feature_importance = {}
            for i, class_label in enumerate(classifier.classes_):
                if len(classifier.classes_) == 2 and i == 0:
                    # For binary classification, only one set of coefficients
                    coefs = classifier.coef_[0]
                else:
                    coefs = classifier.coef_[i]
                
                # Sort and get the highest ones
                sorted_indices = coefs.argsort()[::-1][:top_n]
                feature_importance[class_label] = [(feature_names[j], coefs[j]) for j in sorted_indices]
            
            return feature_importance
        
        # For tree-based models (RandomForest)
        elif hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            sorted_indices = importances.argsort()[::-1][:top_n]
            # Tree models don't have class-specific importances
            feature_importance = {'all': [(feature_names[j], importances[j]) for j in sorted_indices]}
            return feature_importance
        
        return None
    except:
        return None

def save_model(model, model_name, models_dir="models"):
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model pipeline.
        model_name (str): Name to save the model as.
        models_dir (str): Directory to save models in.
        
    Returns:
        str: Path to the saved model.
    """
    # Create models directory if it doesn't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Create a safe filename
    safe_name = model_name.replace(" ", "_").lower()
    model_path = os.path.join(models_dir, f"{safe_name}.joblib")
    
    # Save the model
    joblib.dump(model, model_path)
    
    return model_path

def load_model(model_path):
    """
    Load a trained model from disk.
    
    Args:
        model_path (str): Path to the saved model.
        
    Returns:
        object: Loaded model.
    """
    # Check if the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the model
    model = joblib.load(model_path)
    
    return model

def get_saved_models(models_dir="models"):
    """
    Get a list of saved models.
    
    Args:
        models_dir (str): Directory to look for models in.
        
    Returns:
        list: List of model filenames.
    """
    # Check if models directory exists
    if not os.path.exists(models_dir):
        return []
    
    # Get all .joblib files
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
    
    return model_files 
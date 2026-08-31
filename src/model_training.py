from __future__ import annotations

import re
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.nlp_processing import create_hashing_vectorizer, create_tfidf_vectorizer


def get_available_models(random_state=42):
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, solver="liblinear", random_state=random_state),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(max_iter=5000, dual="auto", random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=1),
    }


def get_hyperparameter_grid(model_name):
    grids = {
        "Multinomial Naive Bayes": {"classifier__alpha": [0.25, 0.5, 1.0, 2.0]},
        "Logistic Regression": {"classifier__C": [0.25, 1.0, 4.0]},
        "Linear SVM": {"classifier__C": [0.25, 1.0, 4.0]},
        "Random Forest": {"classifier__n_estimators": [100, 200], "classifier__max_depth": [None, 30]},
    }
    common = {"vectorizer__max_features": [3000, 5000], "vectorizer__ngram_range": [(1, 1), (1, 2)]}
    return {**common, **grids[model_name]} if model_name in grids else {}


def train_model(df, text_column, sentiment_column, model_name="Logistic Regression", test_size=0.2,
                random_state=42, tune_hyperparameters=False, use_hashing_vectorizer=False,
                max_features=5000, ngram_range=(1, 2), handle_class_imbalance=True,
                n_jobs=-1, cv=3, verbose=0):
    if text_column not in df or sentiment_column not in df:
        raise KeyError("The configured text or sentiment column is missing.")
    working = df[[text_column, sentiment_column]].dropna().copy()
    if working[sentiment_column].nunique() < 2:
        raise ValueError("At least two sentiment classes are required.")
    X_train, X_test, y_train, y_test = train_test_split(
        working[text_column].astype(str), working[sentiment_column], test_size=test_size,
        random_state=random_state, stratify=working[sentiment_column])
    models = get_available_models(random_state)
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    classifier = models[model_name]
    if handle_class_imbalance and hasattr(classifier, "class_weight"):
        classifier.set_params(class_weight="balanced")
    vectorizer = create_hashing_vectorizer(2**18, ngram_range) if use_hashing_vectorizer else create_tfidf_vectorizer(max_features, ngram_range)
    model = Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])
    best_params = None
    if tune_hyperparameters:
        if use_hashing_vectorizer:
            raise ValueError("Tuning currently requires TF-IDF.")
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        search = GridSearchCV(model, get_hyperparameter_grid(model_name), cv=splitter, n_jobs=n_jobs,
                              scoring="f1_macro", verbose=verbose)
        search.fit(X_train, y_train)
        model, best_params = search.best_estimator_, search.best_params_
    else:
        model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test, best_params


def evaluate_model(model, X_test, y_test, target_accuracy=None):
    predicted = model.predict(X_test); accuracy = accuracy_score(y_test, predicted)
    return {
        "accuracy": accuracy,
        "precision": precision_score(y_test, predicted, average="weighted", zero_division=0),
        "recall": recall_score(y_test, predicted, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, predicted, average="weighted", zero_division=0),
        "macro_f1": f1_score(y_test, predicted, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, predicted),
        "confusion_matrix": confusion_matrix(y_test, predicted).tolist(),
        "classification_report": classification_report(y_test, predicted, output_dict=True, zero_division=0),
        "target_achieved": None if target_accuracy is None else accuracy >= target_accuracy,
    }


def predict_sentiment(model, text):
    return model.predict([text])[0]


def predict_sentiment_with_probability(model, text):
    if not hasattr(model, "predict_proba"):
        raise ValueError("This estimator does not expose calibrated probabilities.")
    prediction = model.predict([text])[0]
    return prediction, dict(zip(model.classes_, model.predict_proba([text])[0], strict=True))


def get_feature_importance(model, top_n=20):
    if not hasattr(model, "named_steps"):
        return None
    vectorizer, classifier = model.named_steps.get("vectorizer"), model.named_steps.get("classifier")
    if vectorizer is None or classifier is None or isinstance(vectorizer, HashingVectorizer):
        return None
    names = vectorizer.get_feature_names_out()
    if hasattr(classifier, "coef_"):
        importance = np.mean(np.abs(np.asarray(classifier.coef_)), axis=0)
    elif hasattr(classifier, "feature_importances_"):
        importance = np.asarray(classifier.feature_importances_)
    else:
        return None
    indices = np.argsort(importance)[-top_n:][::-1]
    return {"all": [(names[i], float(importance[i])) for i in indices]}


def _safe_name(name):
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip()).strip("._")
    if not value:
        raise ValueError("Invalid model name.")
    return value.lower()


def save_model(model, model_name, models_dir="models"):
    directory = Path(models_dir); directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_safe_name(model_name)}.joblib"; joblib.dump(model, path); return str(path)


def load_model(model_path):
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    warnings.warn("joblib can execute code while loading; only load trusted model artifacts.", RuntimeWarning, stacklevel=2)
    return joblib.load(path)


def get_saved_models(models_dir="models"):
    directory = Path(models_dir)
    return [path.name for path in sorted(directory.glob("*.joblib"))] if directory.is_dir() else []


def batch_predict(model, df, text_column, prediction_column="predicted_sentiment", confidence_column="confidence", batch_size=1000):
    del batch_size
    result = df.copy(); texts = result[text_column].astype(str); result[prediction_column] = model.predict(texts)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(texts); lookup = {c: i for i, c in enumerate(model.classes_)}
        result[confidence_column] = [row[lookup[p]] for p, row in zip(result[prediction_column], probabilities)]
    else:
        result[confidence_column] = np.nan
    return result

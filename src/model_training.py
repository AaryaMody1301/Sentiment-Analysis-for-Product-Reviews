from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.nlp_processing import create_hashing_vectorizer, create_tfidf_vectorizer


def get_available_models(random_state: int = 42):
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, solver="liblinear", random_state=random_state
        ),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(max_iter=5000, dual="auto", random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=random_state, n_jobs=1
        ),
    }


def get_hyperparameter_grid(model_name: str) -> dict[str, list[object]]:
    grids = {
        "Multinomial Naive Bayes": {"classifier__alpha": [0.25, 0.5, 1.0, 2.0]},
        "Logistic Regression": {"classifier__C": [0.25, 1.0, 4.0]},
        "Linear SVM": {"classifier__C": [0.25, 1.0, 4.0]},
        "Random Forest": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 30],
        },
    }
    common = {
        "vectorizer__max_features": [3000, 5000],
        "vectorizer__ngram_range": [(1, 1), (1, 2)],
    }
    return {**common, **grids[model_name]} if model_name in grids else {}


def train_model(
    df: pd.DataFrame,
    text_column: str,
    sentiment_column: str,
    model_name: str = "Logistic Regression",
    test_size: float = 0.2,
    random_state: int = 42,
    tune_hyperparameters: bool = False,
    use_hashing_vectorizer: bool = False,
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    handle_class_imbalance: bool = True,
    n_jobs: int = -1,
    cv: int = 3,
    verbose: int = 0,
):
    if text_column not in df or sentiment_column not in df:
        raise KeyError("The configured text or sentiment column is missing.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    working = df[[text_column, sentiment_column]].dropna().copy()
    if working[sentiment_column].nunique() < 2:
        raise ValueError("At least two sentiment classes are required.")

    X_train, X_test, y_train, y_test = train_test_split(
        working[text_column].astype(str),
        working[sentiment_column],
        test_size=test_size,
        random_state=random_state,
        stratify=working[sentiment_column],
    )

    models = get_available_models(random_state)
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    classifier = models[model_name]
    if handle_class_imbalance and hasattr(classifier, "class_weight"):
        classifier.set_params(class_weight="balanced")

    vectorizer = (
        create_hashing_vectorizer(2**18, ngram_range)
        if use_hashing_vectorizer
        else create_tfidf_vectorizer(max_features, ngram_range)
    )
    model = Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])

    best_params = None
    if tune_hyperparameters:
        if use_hashing_vectorizer:
            raise ValueError("Tuning currently requires TF-IDF.")
        if cv < 2:
            raise ValueError("cv must be at least 2.")
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        search = GridSearchCV(
            model,
            get_hyperparameter_grid(model_name),
            cv=splitter,
            n_jobs=n_jobs,
            scoring="f1_macro",
            verbose=verbose,
        )
        search.fit(X_train, y_train)
        model, best_params = search.best_estimator_, search.best_params_
    else:
        model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test, best_params


def evaluate_model(model, X_test, y_test, target_accuracy=None) -> dict[str, object]:
    predicted = model.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)
    return {
        "accuracy": accuracy,
        "precision": precision_score(y_test, predicted, average="weighted", zero_division=0),
        "recall": recall_score(y_test, predicted, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, predicted, average="weighted", zero_division=0),
        "macro_f1": f1_score(y_test, predicted, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, predicted),
        "confusion_matrix": confusion_matrix(y_test, predicted).tolist(),
        "classification_report": classification_report(
            y_test, predicted, output_dict=True, zero_division=0
        ),
        "target_achieved": None if target_accuracy is None else accuracy >= target_accuracy,
    }


def predict_sentiment(model, text: object):
    return model.predict([text])[0]


def predict_sentiment_with_probability(model, text: object):
    if not hasattr(model, "predict_proba"):
        raise ValueError("This estimator does not expose probability estimates.")
    prediction = model.predict([text])[0]
    return prediction, dict(zip(model.classes_, model.predict_proba([text])[0], strict=True))


def get_feature_importance(model, top_n: int = 20):
    if top_n <= 0 or not hasattr(model, "named_steps"):
        return None
    vectorizer = model.named_steps.get("vectorizer")
    classifier = model.named_steps.get("classifier")
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


def batch_predict(
    model,
    df: pd.DataFrame,
    text_column: str,
    prediction_column: str = "predicted_sentiment",
    confidence_column: str = "confidence",
    batch_size: int = 1000,
) -> pd.DataFrame:
    if text_column not in df.columns:
        raise KeyError(f"Text column '{text_column}' was not found.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    result = df.copy()
    predictions: list[object] = []
    confidence: list[float] = []
    has_probability = hasattr(model, "predict_proba")

    for start in range(0, len(result), batch_size):
        texts = result[text_column].iloc[start : start + batch_size].astype(str)
        batch_predictions = model.predict(texts)
        predictions.extend(batch_predictions.tolist())
        if has_probability:
            probabilities = model.predict_proba(texts)
            lookup = {value: index for index, value in enumerate(model.classes_)}
            confidence.extend(
                float(row[lookup[prediction]])
                for prediction, row in zip(batch_predictions, probabilities, strict=True)
            )

    result[prediction_column] = predictions
    result[confidence_column] = confidence if has_probability else np.nan
    return result

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB

from src.nlp_processing import (
    detect_columns as _detect_columns,
    normalize_sentiment_labels,
    preprocess_text,
    resolve_label_schema,
)

ProgressCallback = Callable[[float, str], None]
DEFAULT_RANDOM_STATE = 42
DEFAULT_MAX_TEST_SAMPLES = 10_000


def detect_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    return _detect_columns(df)


def _notify(callback: ProgressCallback | None, progress: float, message: str) -> None:
    if callback:
        callback(float(max(0.0, min(1.0, progress))), message)


def _preprocess_batch(
    texts,
    *,
    remove_stopwords: bool,
    perform_stemming: bool,
    perform_lemmatization: bool,
    handle_negations: bool,
) -> list[str]:
    return [
        preprocess_text(
            text,
            remove_stopwords=remove_stopwords,
            perform_stemming=perform_stemming,
            perform_lemmatization=perform_lemmatization,
            handle_negations=handle_negations,
        )
        for text in texts
    ]


def _collect_classes(
    file_path: str | os.PathLike[str],
    sentiment_column: str,
    *,
    chunksize: int,
    label_schema,
) -> tuple[list[str], int]:
    classes: set[str] = set()
    row_count = 0
    for chunk in pd.read_csv(file_path, chunksize=chunksize, usecols=[sentiment_column]):
        row_count += len(chunk)
        chunk = chunk.dropna(subset=[sentiment_column])
        if chunk.empty:
            continue
        normalized = normalize_sentiment_labels(
            chunk, sentiment_column, schema=label_schema
        )
        classes.update(normalized[sentiment_column].unique().tolist())
    return sorted(classes), row_count


def process_large_file(
    file_path,
    text_column=None,
    sentiment_column=None,
    chunksize=20_000,
    test_size=0.2,
    remove_stopwords=True,
    perform_stemming=False,
    perform_lemmatization=False,
    handle_negations=True,
    n_features=2**18,
    callback=None,
    random_state=DEFAULT_RANDOM_STATE,
    max_test_samples=DEFAULT_MAX_TEST_SAMPLES,
):
    """Train a bounded-memory classifier with a deterministic streaming holdout."""

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if chunksize < 2:
        raise ValueError("chunksize must be at least 2.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if n_features <= 0:
        raise ValueError("n_features must be positive.")
    if max_test_samples <= 0:
        raise ValueError("max_test_samples must be positive.")

    started = time.time()
    sample = pd.read_csv(path, nrows=min(5000, chunksize))
    if sample.empty:
        raise ValueError("The dataset is empty.")

    detected_text, detected_sentiment = detect_columns(sample)
    text_column = text_column or detected_text
    sentiment_column = sentiment_column or detected_sentiment
    if text_column not in sample.columns or sentiment_column not in sample.columns:
        raise ValueError(
            f"Required columns not found: text={text_column!r}, sentiment={sentiment_column!r}."
        )

    label_schema = resolve_label_schema(
        sample[sentiment_column], column_name=sentiment_column, schema="auto"
    )
    _notify(
        callback,
        0.02,
        f"Validated columns: text={text_column}, sentiment={sentiment_column}, "
        f"label_schema={label_schema.value}",
    )

    classes, total_rows = _collect_classes(
        path,
        sentiment_column,
        chunksize=chunksize,
        label_schema=label_schema,
    )
    if len(classes) < 2:
        raise ValueError(f"Need at least two sentiment classes; found {classes}.")
    _notify(callback, 0.08, f"Validated {total_rows:,} rows across classes: {classes}")

    vectorizer = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        ngram_range=(1, 2),
        norm="l2",
    )
    model = MultinomialNB(alpha=1.5)
    rng = np.random.default_rng(random_state)

    test_texts: list[str] = []
    test_labels: list[str] = []
    rows_seen = 0
    rows_trained = 0
    first_fit = True

    for chunk_number, chunk in enumerate(
        pd.read_csv(path, chunksize=chunksize), start=1
    ):
        rows_seen += len(chunk)
        missing = {text_column, sentiment_column} - set(chunk.columns)
        if missing:
            raise ValueError(
                f"Required columns missing from chunk {chunk_number}: {sorted(missing)}"
            )

        chunk = chunk.dropna(subset=[text_column, sentiment_column]).copy()
        if chunk.empty:
            continue
        chunk = normalize_sentiment_labels(
            chunk, sentiment_column, schema=label_schema
        )
        processed_texts = _preprocess_batch(
            chunk[text_column].tolist(),
            remove_stopwords=remove_stopwords,
            perform_stemming=perform_stemming,
            perform_lemmatization=perform_lemmatization,
            handle_negations=handle_negations,
        )
        labels = chunk[sentiment_column].to_numpy()

        if len(labels) == 1:
            holdout_mask = np.array([False])
        else:
            holdout_mask = rng.random(len(labels)) < test_size
            if not holdout_mask.any():
                holdout_mask[int(rng.integers(0, len(labels)))] = True
            if holdout_mask.all():
                holdout_mask[int(rng.integers(0, len(labels)))] = False

        train_mask = ~holdout_mask
        train_texts = [
            text
            for text, keep in zip(processed_texts, train_mask, strict=True)
            if keep
        ]
        train_labels = labels[train_mask]
        if train_texts:
            vectors = vectorizer.transform(train_texts)
            if first_fit:
                model.partial_fit(vectors, train_labels, classes=classes)
                first_fit = False
            else:
                model.partial_fit(vectors, train_labels)
            rows_trained += len(train_texts)

        remaining = max_test_samples - len(test_texts)
        if remaining > 0:
            holdout_indices = np.flatnonzero(holdout_mask)[:remaining]
            test_texts.extend(processed_texts[index] for index in holdout_indices)
            test_labels.extend(labels[holdout_indices].tolist())

        _notify(
            callback,
            0.08 + 0.82 * (rows_seen / max(total_rows, 1)),
            f"Processed chunk {chunk_number}: {rows_seen:,}/{total_rows:,} rows",
        )

    if first_fit:
        raise ValueError("No usable training rows were found.")

    if not test_texts:
        metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "macro_f1": 0.0,
            "balanced_accuracy": 0.0,
            "confusion_matrix": [[0 for _ in classes] for _ in classes],
            "labels": classes,
        }
    else:
        predictions = model.predict(vectorizer.transform(test_texts))
        metrics = {
            "accuracy": accuracy_score(test_labels, predictions),
            "precision": precision_score(
                test_labels, predictions, average="weighted", zero_division=0
            ),
            "recall": recall_score(
                test_labels, predictions, average="weighted", zero_division=0
            ),
            "f1_score": f1_score(
                test_labels, predictions, average="weighted", zero_division=0
            ),
            "macro_f1": f1_score(
                test_labels, predictions, average="macro", zero_division=0
            ),
            "balanced_accuracy": balanced_accuracy_score(test_labels, predictions),
            "confusion_matrix": confusion_matrix(
                test_labels, predictions, labels=classes
            ).tolist(),
            "labels": classes,
        }

    metrics.update(
        {
            "rows_scanned": total_rows,
            "rows_trained": rows_trained,
            "rows_tested": len(test_texts),
            "random_state": random_state,
            "label_schema": label_schema.value,
        }
    )
    _notify(
        callback,
        1.0,
        f"Processing complete in {time.time() - started:.1f}s. "
        f"Accuracy: {metrics['accuracy']:.4f}",
    )
    return model, vectorizer, metrics


def predict_batch(
    model,
    vectorizer,
    data,
    text_column,
    remove_stopwords=True,
    perform_stemming=False,
    perform_lemmatization=False,
    handle_negations=True,
    callback=None,
):
    """Predict a DataFrame or CSV path in bounded chunks."""

    if isinstance(data, (str, os.PathLike)):
        results = []
        for index, chunk in enumerate(pd.read_csv(data, chunksize=10_000), start=1):
            _notify(callback, 0.5, f"Processing prediction chunk {index}")
            results.append(
                process_prediction_chunk(
                    model,
                    vectorizer,
                    chunk,
                    text_column,
                    remove_stopwords,
                    perform_stemming,
                    perform_lemmatization,
                    handle_negations,
                )
            )
        output = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    else:
        output = process_prediction_chunk(
            model,
            vectorizer,
            data,
            text_column,
            remove_stopwords,
            perform_stemming,
            perform_lemmatization,
            handle_negations,
        )
    _notify(callback, 1.0, "Prediction complete")
    return output


def process_prediction_chunk(
    model,
    vectorizer,
    chunk,
    text_column,
    remove_stopwords,
    perform_stemming,
    perform_lemmatization,
    handle_negations,
):
    if text_column not in chunk.columns:
        raise ValueError(f"Text column '{text_column}' not found in data")
    result = chunk.copy()
    result["processed_text"] = _preprocess_batch(
        result[text_column].tolist(),
        remove_stopwords=remove_stopwords,
        perform_stemming=perform_stemming,
        perform_lemmatization=perform_lemmatization,
        handle_negations=handle_negations,
    )
    vectors = vectorizer.transform(result["processed_text"])
    result["prediction"] = model.predict(vectors)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vectors)
        for index, class_name in enumerate(model.classes_):
            result[f"confidence_{class_name}"] = probabilities[:, index]
    return result

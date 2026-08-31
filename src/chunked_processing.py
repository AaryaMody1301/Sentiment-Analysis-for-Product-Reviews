from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
from typing import Callable

import joblib
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
    """Detect likely review-text and sentiment columns."""
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
            chunk,
            sentiment_column,
            schema=label_schema,
        )
        classes.update(normalized[sentiment_column].unique().tolist())
    return sorted(classes), row_count


def process_large_file(
    file_path,
    text_column=None,
    sentiment_column=None,
    chunksize=20000,
    test_size=0.2,
    remove_stopwords=True,
    perform_stemming=False,
    perform_lemmatization=True,
    handle_negations=True,
    n_features=2**18,
    callback=None,
    random_state=DEFAULT_RANDOM_STATE,
    max_test_samples=DEFAULT_MAX_TEST_SAMPLES,
):
    """Train an incremental sentiment model with a deterministic streaming split.

    The file is scanned once to validate the label contract and discover the full
    class set, then streamed a second time for training. Each row is assigned to
    train or holdout data by a seeded RNG. Holdout rows are never passed to
    ``partial_fit``.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if chunksize < 2:
        raise ValueError("chunksize must be at least 2.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if n_features <= 0:
        raise ValueError("n_features must be positive.")

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
        sample[sentiment_column],
        column_name=sentiment_column,
        schema="auto",
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

    reader = pd.read_csv(path, chunksize=chunksize)
    for chunk_number, chunk in enumerate(reader, start=1):
        rows_seen += len(chunk)
        missing = {text_column, sentiment_column} - set(chunk.columns)
        if missing:
            raise ValueError(f"Required columns missing from chunk {chunk_number}: {sorted(missing)}")

        chunk = chunk.dropna(subset=[text_column, sentiment_column]).copy()
        if chunk.empty:
            continue

        # Normalize labels before any split/sampling so raw ratings can never be
        # compared against normalized class names.
        chunk = normalize_sentiment_labels(
            chunk,
            sentiment_column,
            schema=label_schema,
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
        train_texts = [text for text, keep in zip(processed_texts, train_mask, strict=True) if keep]
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

        progress = 0.08 + 0.82 * (rows_seen / max(total_rows, 1))
        _notify(
            callback,
            progress,
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
        _notify(callback, 0.93, f"Evaluating on {len(test_texts):,} untouched holdout rows")
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
    elapsed = time.time() - started
    _notify(
        callback,
        1.0,
        f"Processing complete in {elapsed:.1f}s. Accuracy: {metrics['accuracy']:.4f}",
    )
    return model, vectorizer, metrics


def predict_batch(
    model,
    vectorizer,
    data,
    text_column,
    remove_stopwords=True,
    perform_stemming=False,
    perform_lemmatization=True,
    handle_negations=True,
    callback=None,
):
    """Predict sentiment for a DataFrame or CSV path in bounded chunks."""
    if isinstance(data, (str, os.PathLike)):
        results = []
        for index, chunk in enumerate(pd.read_csv(data, chunksize=10000), start=1):
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
        if not results:
            return pd.DataFrame()
        output = pd.concat(results, ignore_index=True)
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
    """Predict one in-memory chunk without mutating the caller's DataFrame."""
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


def save_chunked_model(model, vectorizer, metrics, model_name, directory="models"):
    """Persist a chunked model bundle. Only load artifacts from trusted sources."""
    model_dir = Path(directory) / model_name.lower().replace(" ", "_")
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")
    joblib.dump(vectorizer, model_dir / "vectorizer.joblib")
    joblib.dump(metrics, model_dir / "metrics.joblib")
    info = {
        "name": model_name,
        "type": type(model).__name__,
        "n_features": getattr(vectorizer, "n_features", None),
        "accuracy": metrics.get("accuracy"),
        "labels": list(getattr(model, "classes_", [])),
        "random_state": metrics.get("random_state"),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    joblib.dump(info, model_dir / "info.joblib")
    return str(model_dir)


def load_chunked_model(model_dir):
    """Load a trusted joblib model bundle."""
    warnings.warn(
        "joblib/pickle artifacts can execute code when loaded. Load only files you trust.",
        UserWarning,
        stacklevel=2,
    )
    path = Path(model_dir)
    model = joblib.load(path / "model.joblib")
    vectorizer = joblib.load(path / "vectorizer.joblib")
    metrics = joblib.load(path / "metrics.joblib") if (path / "metrics.joblib").exists() else None
    info = joblib.load(path / "info.joblib") if (path / "info.joblib").exists() else None
    return model, vectorizer, metrics, info


def get_chunked_models(directory="models"):
    """Return model directories containing a persisted model artifact."""
    path = Path(directory)
    if not path.exists():
        return []
    return sorted(
        str(item)
        for item in path.iterdir()
        if item.is_dir() and (item / "model.joblib").exists()
    )

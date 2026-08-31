from __future__ import annotations

import re
import warnings
from functools import lru_cache
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.model_training import get_available_models
from src.nlp_processing import (
    LabelSchema,
    create_tfidf_vectorizer,
    normalize_sentiment_series,
    preprocess_text,
    resolve_label_schema,
)

BUNDLE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PreprocessingConfig:
    """Immutable preprocessing contract captured at training time."""

    remove_stopwords: bool = True
    perform_stemming: bool = False
    perform_lemmatization: bool = False
    handle_negations: bool = True

    def apply(self, text: object) -> str:
        if self.perform_lemmatization:
            _ensure_wordnet_available()
        return preprocess_text(
            text,
            remove_stopwords=self.remove_stopwords,
            perform_stemming=self.perform_stemming,
            perform_lemmatization=self.perform_lemmatization,
            handle_negations=self.handle_negations,
        )


@lru_cache(maxsize=1)
def _ensure_wordnet_available() -> None:
    try:
        from nltk.stem import WordNetLemmatizer

        WordNetLemmatizer().lemmatize("products")
    except LookupError as exc:
        raise RuntimeError(
            "This inference contract requires NLTK WordNet, but the resource is not "
            "installed. Run 'python -m nltk.downloader wordnet' before training or "
            "prediction, or disable lemmatization."
        ) from exc


@dataclass
class InferenceBundle:
    """Model plus the immutable contract required to reproduce inference."""

    model: Any
    model_name: str
    preprocessing: PreprocessingConfig
    label_schema: str
    calibration_method: str | None
    random_state: int
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = BUNDLE_SCHEMA_VERSION

    @property
    def classes(self) -> tuple[str, ...]:
        classes = getattr(self.model, "classes_", None)
        if classes is None:
            raise ValueError("The fitted model does not expose class labels.")
        return tuple(str(value) for value in classes)

    @property
    def has_probability_estimates(self) -> bool:
        return hasattr(self.model, "predict_proba")

    @property
    def confidence_kind(self) -> str:
        if self.calibration_method:
            return f"calibrated_{self.calibration_method}"
        if self.has_probability_estimates:
            return "native_probability"
        return "unavailable"

    def preprocess_many(self, texts: Iterable[object]) -> list[str]:
        return [self.preprocessing.apply(text) for text in texts]

    def predict(self, texts: Sequence[object] | object) -> np.ndarray:
        values = _coerce_texts(texts)
        return np.asarray(self.model.predict(self.preprocess_many(values)))

    def predict_proba(self, texts: Sequence[object] | object) -> np.ndarray:
        if not self.has_probability_estimates:
            raise ValueError(
                "This inference bundle does not expose probability estimates. "
                "Train with calibration enabled to obtain confidence values."
            )
        values = _coerce_texts(texts)
        return np.asarray(self.model.predict_proba(self.preprocess_many(values)), dtype=float)

    def predict_frame(self, texts: Sequence[object] | object) -> pd.DataFrame:
        values = _coerce_texts(texts)
        predictions = self.predict(values)
        result = pd.DataFrame(
            {
                "text": [str(value) for value in values],
                "predicted_sentiment": predictions,
            }
        )
        if not self.has_probability_estimates:
            result["confidence"] = np.nan
            return result

        probabilities = self.predict_proba(values)
        classes = np.asarray(self.classes)
        winning_indices = np.argmax(probabilities, axis=1)
        result["confidence"] = probabilities[np.arange(len(probabilities)), winning_indices]
        for index, class_name in enumerate(classes):
            result[f"probability_{class_name}"] = probabilities[:, index]
        return result


def _coerce_texts(texts: Sequence[object] | object) -> list[object]:
    if isinstance(texts, (str, bytes)) or np.isscalar(texts):
        return [texts]
    try:
        return list(texts)
    except TypeError:
        return [texts]


def _validate_calibration_cv(y_train: pd.Series, requested_cv: int) -> None:
    if requested_cv < 2:
        raise ValueError("calibration_cv must be at least 2.")
    smallest_class = int(y_train.value_counts().min())
    if smallest_class < requested_cv:
        raise ValueError(
            "Each training class must contain at least calibration_cv examples. "
            f"Smallest class has {smallest_class}; requested calibration_cv={requested_cv}."
        )


def train_inference_bundle(
    df: pd.DataFrame,
    text_column: str,
    sentiment_column: str,
    *,
    model_name: str = "Logistic Regression",
    preprocessing: PreprocessingConfig | None = None,
    label_schema: str | LabelSchema = "auto",
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 10_000,
    ngram_range: tuple[int, int] = (1, 2),
    handle_class_imbalance: bool = True,
    calibrate: bool = True,
    calibration_method: str = "sigmoid",
    calibration_cv: int = 3,
) -> tuple[InferenceBundle, dict[str, Any], pd.DataFrame]:
    """Train an inference-safe bundle and evaluate it on an untouched holdout split.

    Calibration, when enabled, is fitted only inside the training split via
    cross-validation. The holdout split is never used to fit the vectorizer,
    classifier, or calibrator.
    """

    if text_column not in df.columns or sentiment_column not in df.columns:
        raise KeyError("The configured text or sentiment column is missing.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if calibration_method not in {"sigmoid", "isotonic", "temperature"}:
        raise ValueError("calibration_method must be sigmoid, isotonic, or temperature.")

    working = df[[text_column, sentiment_column]].dropna().copy()
    if working.empty:
        raise ValueError("No complete text/label rows are available for training.")

    resolved_schema = resolve_label_schema(
        working[sentiment_column],
        sentiment_column,
        label_schema,
    )
    normalized = normalize_sentiment_series(
        working[sentiment_column],
        schema=resolved_schema,
        column_name=sentiment_column,
    )
    if normalized.nunique() < 2:
        raise ValueError("At least two sentiment classes are required.")

    raw_text = working[text_column].astype(str)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        raw_text,
        normalized,
        test_size=test_size,
        random_state=random_state,
        stratify=normalized,
    )

    config = preprocessing or PreprocessingConfig()
    X_train = [config.apply(value) for value in X_train_raw]
    X_test = [config.apply(value) for value in X_test_raw]

    models = get_available_models(random_state)
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    classifier = models[model_name]
    if handle_class_imbalance and hasattr(classifier, "class_weight"):
        classifier.set_params(class_weight="balanced")

    base_model = Pipeline(
        [
            (
                "vectorizer",
                create_tfidf_vectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                ),
            ),
            ("classifier", classifier),
        ]
    )

    if calibrate:
        _validate_calibration_cv(y_train, calibration_cv)
        splitter = StratifiedKFold(
            n_splits=calibration_cv,
            shuffle=True,
            random_state=random_state,
        )
        model = CalibratedClassifierCV(
            estimator=base_model,
            method=calibration_method,
            cv=splitter,
            ensemble=False,
        )
    else:
        model = base_model

    model.fit(X_train, y_train)
    bundle = InferenceBundle(
        model=model,
        model_name=model_name,
        preprocessing=config,
        label_schema=resolved_schema.value,
        calibration_method=calibration_method if calibrate else None,
        random_state=random_state,
        metadata={
            "training_rows": int(len(X_train)),
            "holdout_rows": int(len(X_test)),
            "test_size": float(test_size),
            "max_features": int(max_features),
            "ngram_range": list(ngram_range),
            "preprocessing": asdict(config),
            "confidence_kind": (
                f"calibrated_{calibration_method}"
                if calibrate
                else ("native_probability" if hasattr(model, "predict_proba") else "unavailable")
            ),
        },
    )

    metrics = evaluate_inference_bundle(bundle, X_test_raw.tolist(), y_test.tolist())
    errors = build_error_analysis(bundle, X_test_raw.tolist(), y_test.tolist())
    return bundle, metrics, errors


def expected_calibration_error(
    y_true: Sequence[object],
    y_pred: Sequence[object],
    confidence: Sequence[float],
    *,
    n_bins: int = 10,
) -> float:
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    true = np.asarray(y_true, dtype=object)
    predicted = np.asarray(y_pred, dtype=object)
    conf = np.asarray(confidence, dtype=float)
    if not (len(true) == len(predicted) == len(conf)):
        raise ValueError("y_true, y_pred, and confidence must have equal length.")
    if len(true) == 0:
        raise ValueError("Calibration error requires at least one example.")

    correctness = (true == predicted).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = float(len(conf))
    ece = 0.0
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        if index == n_bins - 1:
            mask = (conf >= lower) & (conf <= upper)
        else:
            mask = (conf >= lower) & (conf < upper)
        if not mask.any():
            continue
        ece += (float(mask.sum()) / total) * abs(
            float(correctness[mask].mean()) - float(conf[mask].mean())
        )
    return float(ece)


def _selective_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
) -> dict[str, dict[str, float | int | None]]:
    output: dict[str, dict[str, float | int | None]] = {}
    for threshold in (0.5, 0.7, 0.8, 0.9):
        mask = confidence >= threshold
        count = int(mask.sum())
        output[f"{threshold:.1f}"] = {
            "count": count,
            "coverage": round(float(count / len(confidence)), 6),
            "accuracy": (
                round(float(accuracy_score(y_true[mask], y_pred[mask])), 6)
                if count
                else None
            ),
        }
    return output


def evaluate_inference_bundle(
    bundle: InferenceBundle,
    texts: Sequence[object],
    y_true: Sequence[object],
) -> dict[str, Any]:
    if len(texts) != len(y_true):
        raise ValueError("texts and y_true must have equal length.")
    if len(texts) == 0:
        raise ValueError("Evaluation requires at least one example.")

    truth = np.asarray([str(value) for value in y_true], dtype=object)
    predicted = np.asarray([str(value) for value in bundle.predict(texts)], dtype=object)
    labels = sorted(set(truth).union(predicted))
    metrics: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(truth, predicted)), 6),
        "macro_precision": round(
            float(precision_score(truth, predicted, average="macro", zero_division=0)), 6
        ),
        "macro_recall": round(
            float(recall_score(truth, predicted, average="macro", zero_division=0)), 6
        ),
        "macro_f1": round(float(f1_score(truth, predicted, average="macro", zero_division=0)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(truth, predicted)), 6),
        "labels": labels,
        "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
        "confidence_kind": bundle.confidence_kind,
    }

    if bundle.has_probability_estimates:
        probabilities = bundle.predict_proba(texts)
        class_order = list(bundle.classes)
        winning = np.argmax(probabilities, axis=1)
        confidence = probabilities[np.arange(len(probabilities)), winning]
        metrics.update(
            {
                "log_loss": round(float(log_loss(truth, probabilities, labels=class_order)), 6),
                "mean_confidence": round(float(np.mean(confidence)), 6),
                "expected_calibration_error": round(
                    expected_calibration_error(truth, predicted, confidence), 6
                ),
                "selective_accuracy": _selective_metrics(truth, predicted, confidence),
            }
        )
    else:
        metrics.update(
            {
                "log_loss": None,
                "mean_confidence": None,
                "expected_calibration_error": None,
                "selective_accuracy": {},
            }
        )
    return metrics


def build_error_analysis(
    bundle: InferenceBundle,
    texts: Sequence[object],
    y_true: Sequence[object],
) -> pd.DataFrame:
    if len(texts) != len(y_true):
        raise ValueError("texts and y_true must have equal length.")
    predictions = bundle.predict_frame(texts)
    result = predictions.rename(columns={"text": "review_text"}).copy()
    result["true_sentiment"] = [str(value) for value in y_true]
    result["correct"] = result["predicted_sentiment"].astype(str) == result["true_sentiment"]
    result["error_type"] = np.where(
        result["correct"],
        "correct",
        result["true_sentiment"].astype(str)
        + " -> "
        + result["predicted_sentiment"].astype(str),
    )
    # Surface the most confident mistakes first, then uncertain correct predictions.
    result["_error_rank"] = (~result["correct"]).astype(int)
    result = result.sort_values(
        ["_error_rank", "confidence"],
        ascending=[False, False],
        na_position="last",
        kind="stable",
    ).drop(columns=["_error_rank"])
    return result.reset_index(drop=True)


def _safe_name(name: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip()).strip("._")
    if not value:
        raise ValueError("Invalid inference bundle name.")
    return value.lower()


def save_inference_bundle(
    bundle: InferenceBundle,
    bundle_name: str,
    models_dir: str | Path = "models",
) -> str:
    directory = Path(models_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_safe_name(bundle_name)}.inference.joblib"
    joblib.dump(bundle, path)
    return str(path)


def load_inference_bundle(path: str | Path) -> InferenceBundle:
    bundle_path = Path(path)
    if not bundle_path.is_file():
        raise FileNotFoundError(bundle_path)
    warnings.warn(
        "joblib/pickle artifacts can execute code when loaded. Load only files you trust.",
        UserWarning,
        stacklevel=2,
    )
    bundle = joblib.load(bundle_path)
    if not isinstance(bundle, InferenceBundle):
        raise TypeError("The artifact is not an InferenceBundle.")
    if bundle.schema_version != BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported inference bundle schema {bundle.schema_version}; "
            f"expected {BUNDLE_SCHEMA_VERSION}."
        )
    return bundle

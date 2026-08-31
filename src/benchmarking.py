from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

DATASET_ID = "mteb/amazon_polarity"
DATASET_REVISION = "ec149c1fe36043668a50804214d4597804001f6f"
DATASET_LICENSE = "apache-2.0"
DATASET_TRAIN_ROWS = 3_599_994
DATASET_TEST_ROWS = 400_000
LABEL_NAMES = {0: "negative", 1: "positive"}
BENCHMARK_SCHEMA_VERSION = 1
DEFAULT_SEED = 42


@dataclass(frozen=True)
class BenchmarkProfile:
    name: str
    train_per_class: int
    test_per_class: int
    shuffle_buffer: int
    max_features: int

    @property
    def train_rows(self) -> int:
        return self.train_per_class * len(LABEL_NAMES)

    @property
    def test_rows(self) -> int:
        return self.test_per_class * len(LABEL_NAMES)


PROFILES = {
    "smoke": BenchmarkProfile(
        name="smoke",
        train_per_class=500,
        test_per_class=250,
        shuffle_buffer=5_000,
        max_features=5_000,
    ),
    "phase2": BenchmarkProfile(
        name="phase2",
        train_per_class=25_000,
        test_per_class=5_000,
        shuffle_buffer=50_000,
        max_features=50_000,
    ),
}


def collect_balanced_rows(
    rows: Iterable[Mapping[str, object]], per_class: int
) -> list[dict[str, object]]:
    if per_class <= 0:
        raise ValueError("per_class must be positive.")

    counts = {label: 0 for label in LABEL_NAMES}
    selected: list[dict[str, object]] = []

    for row in rows:
        if "label" not in row or "text" not in row:
            raise KeyError("Benchmark rows must contain 'label' and 'text'.")

        try:
            label = int(row["label"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid Amazon Polarity label: {row['label']!r}") from exc
        if label not in LABEL_NAMES:
            raise ValueError(f"Unexpected Amazon Polarity label: {label!r}")
        if counts[label] >= per_class:
            continue

        text = str(row["text"]).strip()
        if not text:
            continue

        label_text = row.get("label_text")
        if label_text is not None and str(label_text).strip().lower() != LABEL_NAMES[label]:
            raise ValueError(
                f"Label contract mismatch: {label!r} does not match {label_text!r}."
            )

        selected.append({"text": text, "label": label})
        counts[label] += 1
        if all(count == per_class for count in counts.values()):
            break

    if any(count != per_class for count in counts.values()):
        raise RuntimeError(
            "The stream ended before the balanced benchmark quota was reached: "
            f"{counts}."
        )
    return selected


def fingerprint_rows(rows: Iterable[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        payload = json.dumps(
            {"label": int(row["label"]), "text": str(row["text"])},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, byteorder="big"))
        digest.update(payload)
    return digest.hexdigest()


def audit_split_integrity(
    train_rows: Iterable[Mapping[str, object]],
    test_rows: Iterable[Mapping[str, object]],
) -> dict[str, int]:
    train_list = list(train_rows)
    test_list = list(test_rows)
    train_texts = [str(row["text"]) for row in train_list]
    test_texts = [str(row["text"]) for row in test_list]
    train_unique = set(train_texts)
    test_unique = set(test_texts)
    return {
        "train_rows": len(train_list),
        "test_rows": len(test_list),
        "train_unique_texts": len(train_unique),
        "test_unique_texts": len(test_unique),
        "train_duplicate_texts": len(train_list) - len(train_unique),
        "test_duplicate_texts": len(test_list) - len(test_unique),
        "cross_split_text_overlap": len(train_unique.intersection(test_unique)),
    }


def class_counts(rows: Iterable[Mapping[str, object]]) -> dict[str, int]:
    counts = {name: 0 for name in LABEL_NAMES.values()}
    for row in rows:
        label = int(row["label"])
        if label not in LABEL_NAMES:
            raise ValueError(f"Unexpected label while counting classes: {label!r}")
        counts[LABEL_NAMES[label]] += 1
    return counts


def build_vectorizer(max_features: int) -> TfidfVectorizer:
    if max_features <= 0:
        raise ValueError("max_features must be positive.")
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )


def build_classifiers(seed: int = DEFAULT_SEED):
    return {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "multinomial_nb": MultinomialNB(alpha=1.0),
        "logistic_regression": LogisticRegression(
            max_iter=2_000,
            solver="liblinear",
            random_state=seed,
        ),
        "linear_svm": LinearSVC(
            C=1.0,
            max_iter=5_000,
            dual="auto",
            random_state=seed,
        ),
    }


def evaluate_predictions(y_true, y_pred) -> dict[str, object]:
    labels = sorted(LABEL_NAMES)
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    per_class = {
        LABEL_NAMES[label]: {
            "precision": round(float(precision[index]), 6),
            "recall": round(float(recall[index]), 6),
            "f1": round(float(per_class_f1[index]), 6),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_precision": round(
            float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 6
        ),
        "macro_recall": round(
            float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 6
        ),
        "macro_f1": round(
            float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6
        ),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": per_class,
    }


def benchmark_rows(
    train_rows: Iterable[Mapping[str, object]],
    test_rows: Iterable[Mapping[str, object]],
    *,
    max_features: int,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict[str, object]]:
    train_list = list(train_rows)
    test_list = list(test_rows)
    if not train_list or not test_list:
        raise ValueError("Both train and test rows are required.")

    train_text = [str(row["text"]) for row in train_list]
    test_text = [str(row["text"]) for row in test_list]
    y_train = np.asarray([int(row["label"]) for row in train_list])
    y_test = np.asarray([int(row["label"]) for row in test_list])
    if set(np.unique(y_train)) != set(LABEL_NAMES):
        raise ValueError("Training rows must contain both Amazon Polarity classes.")
    if set(np.unique(y_test)) != set(LABEL_NAMES):
        raise ValueError("Test rows must contain both Amazon Polarity classes.")

    vectorizer = build_vectorizer(max_features)
    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)

    results: dict[str, dict[str, object]] = {}
    for name, classifier in build_classifiers(seed).items():
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        results[name] = evaluate_predictions(y_test, predictions)
    return results


def _load_balanced_split(
    split: str,
    *,
    per_class: int,
    seed: int,
    shuffle_buffer: int,
) -> list[dict[str, object]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Benchmark dependencies are not installed. Run "
            "'python -m pip install -e \".[benchmark]\"'."
        ) from exc

    stream = load_dataset(
        DATASET_ID,
        split=split,
        revision=DATASET_REVISION,
        streaming=True,
    )
    stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer)
    return collect_balanced_rows(stream, per_class)


def build_benchmark_result(
    profile: BenchmarkProfile,
    train_rows: Iterable[Mapping[str, object]],
    test_rows: Iterable[Mapping[str, object]],
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    train_list = list(train_rows)
    test_list = list(test_rows)
    if len(train_list) != profile.train_rows or len(test_list) != profile.test_rows:
        raise ValueError("The selected rows do not match the benchmark profile size.")

    integrity = audit_split_integrity(train_list, test_list)
    if integrity["cross_split_text_overlap"] != 0:
        raise ValueError("Train/test text overlap detected in the selected benchmark rows.")
    if integrity["train_duplicate_texts"] != 0 or integrity["test_duplicate_texts"] != 0:
        raise ValueError("Duplicate text detected inside a selected benchmark split.")

    model_results = benchmark_rows(
        train_list,
        test_list,
        max_features=profile.max_features,
        seed=seed,
    )
    winner = max(
        model_results,
        key=lambda name: (float(model_results[name]["macro_f1"]), name),
    )

    try:
        datasets_version = importlib.metadata.version("datasets")
    except importlib.metadata.PackageNotFoundError:
        datasets_version = None

    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark_id": f"amazon_polarity_{profile.name}_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "license": DATASET_LICENSE,
            "official_train_rows": DATASET_TRAIN_ROWS,
            "official_test_rows": DATASET_TEST_ROWS,
            "label_contract": {"0": "negative", "1": "positive"},
        },
        "selection": {
            "profile": asdict(profile),
            "seed": seed,
            "strategy": "pinned streaming shuffle followed by exact balanced per-class quota",
            "train_fingerprint_sha256": fingerprint_rows(train_list),
            "test_fingerprint_sha256": fingerprint_rows(test_list),
            "train_class_counts": class_counts(train_list),
            "test_class_counts": class_counts(test_list),
        },
        "integrity": integrity,
        "features": {
            "type": "tfidf",
            "fit_scope": "train_only",
            "lowercase": True,
            "strip_accents": "unicode",
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_df": 0.98,
            "max_features": profile.max_features,
            "sublinear_tf": True,
        },
        "metrics": {
            "primary": "macro_f1",
            "reported": [
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "balanced_accuracy",
                "per_class",
                "confusion_matrix",
            ],
        },
        "models": model_results,
        "winner_by_macro_f1": winner,
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "datasets": datasets_version,
        },
    }


def run_external_benchmark(
    profile_name: str = "phase2",
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    if profile_name not in PROFILES:
        raise ValueError(f"Unknown profile: {profile_name!r}")
    profile = PROFILES[profile_name]
    train_rows = _load_balanced_split(
        "train",
        per_class=profile.train_per_class,
        seed=seed,
        shuffle_buffer=profile.shuffle_buffer,
    )
    test_rows = _load_balanced_split(
        "test",
        per_class=profile.test_per_class,
        seed=seed,
        shuffle_buffer=profile.shuffle_buffer,
    )
    return build_benchmark_result(profile, train_rows, test_rows, seed=seed)


def write_benchmark_result(result: Mapping[str, object], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen Amazon Polarity benchmark.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="phase2")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_external_benchmark(args.profile, seed=args.seed)
    output = args.output or Path("benchmarks/results") / f"{result['benchmark_id']}.json"
    write_benchmark_result(result, output)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

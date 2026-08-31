from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.nlp_processing import (
    LabelSchema,
    normalize_sentiment_labels,
    preprocess_text,
    resolve_label_schema,
)

__all__ = [
    "LabelSchema",
    "get_available_datasets",
    "load_dataset",
    "normalize_sentiment_labels",
    "preprocess_text",
    "resolve_label_schema",
]


def load_dataset(file_path: str | Path, **read_csv_kwargs) -> pd.DataFrame:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    try:
        return pd.read_csv(path, **read_csv_kwargs)
    except Exception as exc:
        raise ValueError(
            f"Could not parse CSV dataset '{path}'. The file was not modified or silently truncated: {exc}"
        ) from exc


def get_available_datasets(directory: str | Path = "datasets") -> list[str]:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return [str(item) for item in sorted(path.glob("*.csv")) if item.is_file()]

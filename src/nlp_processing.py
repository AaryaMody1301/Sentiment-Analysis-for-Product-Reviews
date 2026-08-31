from __future__ import annotations

import re
import warnings
from enum import Enum
from typing import Iterable

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight


class LabelSchema(str, Enum):
    TEXT = "text"
    BINARY_01 = "binary_01"
    STARS_1_TO_5 = "stars_1_to_5"


_TEXT_MAP = {
    "positive": "positive",
    "pos": "positive",
    "yes": "positive",
    "good": "positive",
    "true": "positive",
    "negative": "negative",
    "neg": "negative",
    "no": "negative",
    "bad": "negative",
    "false": "negative",
    "neutral": "neutral",
    "neu": "neutral",
    "maybe": "neutral",
    "ok": "neutral",
    "okay": "neutral",
}
_TEXT_COLUMN_HINTS = ("sentiment", "label", "class", "polarity", "target")
_STAR_COLUMN_HINTS = ("rating", "stars", "star", "score")
_WORDNET_AVAILABLE: bool | None = None
_WORDNET_WARNING_EMITTED = False


def _normalized_scalar(value: object) -> str:
    if pd.isna(value):
        raise ValueError("Sentiment labels cannot be missing.")
    return str(value).strip().lower()


def infer_label_schema(values: Iterable[object], column_name: str = "") -> LabelSchema:
    series = pd.Series(list(values)).dropna()
    if series.empty:
        raise ValueError("Cannot infer a sentiment label schema from an empty column.")

    normalized = {_normalized_scalar(value) for value in series.unique()}
    if normalized and normalized.issubset(_TEXT_MAP):
        return LabelSchema.TEXT

    column = column_name.strip().lower()
    numeric_values: set[int] = set()
    for value in series.unique():
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Unrecognized sentiment labels. Supply an explicit label schema."
            ) from exc
        if not numeric.is_integer():
            raise ValueError(
                "Numeric sentiment labels must be integer-valued and use an explicit schema."
            )
        numeric_values.add(int(numeric))

    if numeric_values.issubset({0, 1}) and any(hint in column for hint in _TEXT_COLUMN_HINTS):
        return LabelSchema.BINARY_01
    if numeric_values.issubset({1, 2, 3, 4, 5}) and any(
        hint in column for hint in _STAR_COLUMN_HINTS
    ):
        return LabelSchema.STARS_1_TO_5

    raise ValueError(
        "Numeric labels are ambiguous. Choose 'binary_01' for 0/1 targets or "
        "'stars_1_to_5' for product ratings."
    )


def resolve_label_schema(
    values: Iterable[object], column_name: str = "", schema: str | LabelSchema = "auto"
) -> LabelSchema:
    if schema == "auto":
        return infer_label_schema(values, column_name)
    try:
        return LabelSchema(schema)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in LabelSchema)
        raise ValueError(f"Unknown label schema '{schema}'. Expected one of: {allowed}.") from exc


def normalize_sentiment_series(
    series: pd.Series, schema: str | LabelSchema = "auto", column_name: str | None = None
) -> pd.Series:
    resolved = resolve_label_schema(series, column_name or str(series.name or ""), schema)

    def map_value(value: object) -> str:
        if resolved is LabelSchema.TEXT:
            key = _normalized_scalar(value)
            if key not in _TEXT_MAP:
                raise ValueError(f"Unsupported text sentiment label: {value!r}")
            return _TEXT_MAP[key]

        try:
            numeric = int(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Expected a numeric sentiment label, got {value!r}.") from exc

        if resolved is LabelSchema.BINARY_01:
            if numeric not in (0, 1):
                raise ValueError(f"binary_01 accepts only 0 or 1, got {value!r}.")
            return "positive" if numeric == 1 else "negative"

        if numeric not in (1, 2, 3, 4, 5):
            raise ValueError(f"stars_1_to_5 accepts only 1 through 5, got {value!r}.")
        if numeric <= 2:
            return "negative"
        if numeric == 3:
            return "neutral"
        return "positive"

    return series.map(map_value)


def normalize_sentiment_labels(
    df: pd.DataFrame, sentiment_column: str, schema: str | LabelSchema = "auto"
) -> pd.DataFrame:
    if sentiment_column not in df.columns:
        raise KeyError(f"Sentiment column '{sentiment_column}' was not found.")
    normalized = df.copy()
    normalized[sentiment_column] = normalize_sentiment_series(
        normalized[sentiment_column], schema=schema, column_name=sentiment_column
    )
    return normalized


def preprocess_text(
    text: object,
    remove_stopwords: bool = True,
    perform_stemming: bool = False,
    perform_lemmatization: bool = True,
    handle_negations: bool = True,
) -> str:
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    if not text:
        return ""

    text = re.sub(r"n't\b", " not", text)
    if handle_negations:
        text = re.sub(r"\bnot\s+([a-z0-9_]+)", r"not_\1", text)
    tokens = re.findall(r"[a-z0-9_]+", text)

    if remove_stopwords:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        stop_words = set(ENGLISH_STOP_WORDS)
        stop_words.discard("not")
        tokens = [
            token
            for token in tokens
            if token.startswith("not_") or token == "not" or token not in stop_words
        ]

    if perform_lemmatization:
        global _WORDNET_AVAILABLE, _WORDNET_WARNING_EMITTED
        lemmatizer = WordNetLemmatizer()
        if _WORDNET_AVAILABLE is not False:
            try:
                tokens = [lemmatizer.lemmatize(token, pos="v") for token in tokens]
                tokens = [lemmatizer.lemmatize(token) for token in tokens]
                _WORDNET_AVAILABLE = True
            except LookupError:
                _WORDNET_AVAILABLE = False
        if _WORDNET_AVAILABLE is False and not _WORDNET_WARNING_EMITTED:
            warnings.warn(
                "NLTK wordnet data is unavailable; continuing without lemmatization. "
                "Run 'python -m nltk.downloader wordnet' to enable it.",
                RuntimeWarning,
                stacklevel=2,
            )
            _WORDNET_WARNING_EMITTED = True
    elif perform_stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]

    return " ".join(tokens)


def create_tfidf_vectorizer(
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    use_idf: bool = True,
) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        use_idf=use_idf,
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )


def create_hashing_vectorizer(
    n_features: int = 2**18, ngram_range: tuple[int, int] = (1, 2)
) -> HashingVectorizer:
    return HashingVectorizer(
        n_features=n_features,
        ngram_range=ngram_range,
        alternate_sign=False,
        norm="l2",
    )


def detect_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    if df.empty and len(df.columns) == 0:
        return None, None

    text_patterns = ("review", "text", "comment", "feedback", "description", "content", "message")
    sentiment_patterns = ("sentiment", "label", "rating", "score", "class", "polarity", "star")

    text_column = next(
        (col for col in df.columns if any(pattern in str(col).lower() for pattern in text_patterns)),
        None,
    )
    sentiment_column = next(
        (
            col
            for col in df.columns
            if col != text_column
            and any(pattern in str(col).lower() for pattern in sentiment_patterns)
        ),
        None,
    )

    if text_column is None:
        string_columns = list(df.select_dtypes(include=["object", "string"]).columns)
        if string_columns:
            text_column = max(
                string_columns,
                key=lambda col: df[col].astype(str).str.len().mean(),
            )

    if sentiment_column is None:
        candidates = [col for col in df.columns if col != text_column]
        if candidates:
            sentiment_column = min(candidates, key=lambda col: df[col].nunique(dropna=True))

    return text_column, sentiment_column


def compute_class_weights(y: Iterable[object]) -> dict[object, float]:
    values = np.asarray(list(y))
    classes = np.unique(values)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=values)
    return dict(zip(classes, weights, strict=True))


def generate_feature_names(vectorizer: object, top_n: int = 20):
    del top_n
    if hasattr(vectorizer, "get_feature_names_out"):
        return vectorizer.get_feature_names_out()
    return None

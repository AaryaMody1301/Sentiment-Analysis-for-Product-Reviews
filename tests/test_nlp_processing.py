import pandas as pd
import pytest

from src.nlp_processing import LabelSchema, normalize_sentiment_labels, preprocess_text, resolve_label_schema


def test_preprocess_text_basic():
    assert preprocess_text("This is a TEST with punctuation!!!", perform_lemmatization=False) == "test punctuation"


def test_preprocess_text_negation():
    processed = preprocess_text("This product is not good and isn't working.", handle_negations=True, perform_lemmatization=False)
    assert "not_good" in processed
    assert "not_working" in processed


def test_star_rating_contract():
    df = pd.DataFrame({"rating": [1, 2, 3, 4, 5]})
    normalized = normalize_sentiment_labels(df, "rating")
    assert normalized["rating"].tolist() == ["negative", "negative", "neutral", "positive", "positive"]


def test_binary_contract():
    df = pd.DataFrame({"label": [0, 1]})
    assert normalize_sentiment_labels(df, "label")["label"].tolist() == ["negative", "positive"]


def test_ambiguous_numeric_labels_raise():
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_label_schema([0, 1], column_name="value")


def test_unknown_text_label_raises():
    df = pd.DataFrame({"sentiment": ["positive", "mystery"]})
    with pytest.raises(ValueError, match="Unsupported"):
        normalize_sentiment_labels(df, "sentiment", schema=LabelSchema.TEXT)

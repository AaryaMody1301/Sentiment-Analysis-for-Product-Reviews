from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB

from src.chunked_processing import detect_columns, predict_batch, process_large_file


def create_test_csv(path: Path, rows: int = 100) -> Path:
    frame = pd.DataFrame(
        {
            "review_text": [
                f"This is {'a positive' if i % 3 == 0 else 'a negative' if i % 3 == 1 else 'a neutral'} review {i}"
                for i in range(rows)
            ],
            "rating": [5 if i % 3 == 0 else (1 if i % 3 == 1 else 3) for i in range(rows)],
            "id": list(range(rows)),
        }
    )
    frame.to_csv(path, index=False)
    return path


def test_detect_columns():
    frame = pd.DataFrame({"review_text": ["test review"], "sentiment": [1]})
    assert detect_columns(frame) == ("review_text", "sentiment")


def test_process_large_file_is_deterministic(tmp_path):
    path = create_test_csv(tmp_path / "reviews.csv")
    first = process_large_file(path, "review_text", "rating", chunksize=10, random_state=17)
    second = process_large_file(path, "review_text", "rating", chunksize=10, random_state=17)
    model, vectorizer, metrics = first
    assert isinstance(model, MultinomialNB)
    assert isinstance(vectorizer, HashingVectorizer)
    assert metrics["macro_f1"] == second[2]["macro_f1"]
    assert metrics["label_schema"] == "stars_1_to_5"
    assert metrics["rows_scanned"] == 100


def test_predict_batch(tmp_path):
    path = create_test_csv(tmp_path / "reviews.csv")
    model, vectorizer, _ = process_large_file(path, "review_text", "rating", chunksize=10)
    test_data = pd.DataFrame(
        {"review_text": ["positive test review", "negative test review", "neutral test review"]}
    )
    result = predict_batch(model, vectorizer, test_data, "review_text")
    assert len(result) == 3
    assert "prediction" in result.columns
    for class_name in model.classes_:
        assert f"confidence_{class_name}" in result.columns

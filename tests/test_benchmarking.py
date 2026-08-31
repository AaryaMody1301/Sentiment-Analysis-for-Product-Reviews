import math

from src.benchmarking import (
    BenchmarkProfile,
    audit_split_integrity,
    benchmark_rows,
    build_benchmark_result,
    collect_balanced_rows,
    fingerprint_rows,
)


def _rows():
    return [
        {"label": 1, "label_text": "positive", "text": "excellent battery life"},
        {"label": 0, "label_text": "negative", "text": "terrible battery life"},
        {"label": 1, "label_text": "positive", "text": "great sound quality"},
        {"label": 0, "label_text": "negative", "text": "awful sound quality"},
        {"label": 1, "label_text": "positive", "text": "excellent and reliable"},
        {"label": 0, "label_text": "negative", "text": "terrible and unreliable"},
    ]


def test_collect_balanced_rows_is_exact_and_preserves_label_contract():
    selected = collect_balanced_rows(_rows(), per_class=2)
    assert len(selected) == 4
    assert sum(row["label"] == 0 for row in selected) == 2
    assert sum(row["label"] == 1 for row in selected) == 2


def test_fingerprint_is_order_sensitive_and_deterministic():
    rows = collect_balanced_rows(_rows(), per_class=2)
    assert fingerprint_rows(rows) == fingerprint_rows(list(rows))
    assert fingerprint_rows(rows) != fingerprint_rows(list(reversed(rows)))


def test_integrity_audit_detects_cross_split_overlap():
    train = [{"label": 0, "text": "same"}, {"label": 1, "text": "different"}]
    test = [{"label": 0, "text": "same"}, {"label": 1, "text": "other"}]
    audit = audit_split_integrity(train, test)
    assert audit["cross_split_text_overlap"] == 1


def test_benchmark_rows_reports_complete_metric_contract():
    train = _rows() * 8
    test = [
        {"label": 1, "text": "excellent sound and battery"},
        {"label": 0, "text": "terrible sound and battery"},
        {"label": 1, "text": "great reliable quality"},
        {"label": 0, "text": "awful unreliable quality"},
    ]
    results = benchmark_rows(train, test, max_features=100, seed=42)
    assert set(results) == {
        "dummy_most_frequent",
        "multinomial_nb",
        "logistic_regression",
        "linear_svm",
    }
    for metrics in results.values():
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["macro_f1"] <= 1.0
        assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
        assert len(metrics["confusion_matrix"]) == 2
        assert set(metrics["per_class"]) == {"negative", "positive"}
        assert all(math.isfinite(value) for value in [metrics["accuracy"], metrics["macro_f1"]])


def test_build_result_records_reproducibility_and_no_overlap():
    profile = BenchmarkProfile(
        name="unit",
        train_per_class=2,
        test_per_class=1,
        shuffle_buffer=10,
        max_features=50,
    )
    train = [
        {"label": 0, "text": "bad product bad"},
        {"label": 1, "text": "good product good"},
        {"label": 0, "text": "awful item awful"},
        {"label": 1, "text": "great item great"},
    ]
    test = [
        {"label": 0, "text": "bad awful"},
        {"label": 1, "text": "good great"},
    ]
    result = build_benchmark_result(profile, train, test, seed=42)
    assert result["selection"]["train_class_counts"] == {"negative": 2, "positive": 2}
    assert result["selection"]["test_class_counts"] == {"negative": 1, "positive": 1}
    assert result["integrity"]["cross_split_text_overlap"] == 0
    assert result["features"]["fit_scope"] == "train_only"
    assert result["metrics"]["primary"] == "macro_f1"
    assert result["winner_by_macro_f1"] in result["models"]

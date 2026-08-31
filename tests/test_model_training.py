import numpy as np
import pandas as pd
import pytest

from src.model_training import batch_predict, evaluate_model, get_feature_importance, predict_sentiment_with_probability, train_model


def dataset():
    positive = [f"excellent durable product works great {i}" for i in range(20)]
    negative = [f"terrible broken product waste money {i}" for i in range(20)]
    return pd.DataFrame({"text": positive + negative, "sentiment": ["positive"] * 20 + ["negative"] * 20})


def test_fixed_seed_is_deterministic():
    first = train_model(dataset(), "text", "sentiment", random_state=7)
    second = train_model(dataset(), "text", "sentiment", random_state=7)
    a = evaluate_model(first[0], first[2], first[4]); b = evaluate_model(second[0], second[2], second[4])
    assert a["accuracy"] == b["accuracy"]
    assert a["macro_f1"] == b["macro_f1"]
    assert a["target_achieved"] is None


def test_evaluation_reports_imbalance_aware_metrics():
    model, _, X_test, _, y_test, _ = train_model(dataset(), "text", "sentiment")
    metrics = evaluate_model(model, X_test, y_test)
    assert "macro_f1" in metrics and "balanced_accuracy" in metrics


def test_binary_linear_feature_importance_is_safe():
    model, *_ = train_model(dataset(), "text", "sentiment")
    importance = get_feature_importance(model, top_n=5)
    assert list(importance) == ["all"] and len(importance["all"]) == 5


def test_svm_does_not_fake_probability():
    model, *_ = train_model(dataset(), "text", "sentiment", model_name="Linear SVM")
    with pytest.raises(ValueError, match="calibrated probabilities"):
        predict_sentiment_with_probability(model, "great product")
    output = batch_predict(model, pd.DataFrame({"text": ["great", "bad"]}), "text")
    assert np.isnan(output["confidence"]).all()

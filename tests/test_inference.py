import numpy as np
import pandas as pd
import pytest

from src.inference import (
    PreprocessingConfig,
    build_error_analysis,
    expected_calibration_error,
    train_inference_bundle,
)
from src.safe_persistence import load_safe_inference_bundle, save_safe_inference_bundle


def dataset(rows_per_class=30):
    positive = [f"excellent durable product works great value {i}" for i in range(rows_per_class)]
    negative = [f"terrible broken product waste money awful {i}" for i in range(rows_per_class)]
    return pd.DataFrame(
        {
            "review": positive + negative,
            "sentiment": ["positive"] * rows_per_class + ["negative"] * rows_per_class,
        }
    )


def config():
    return PreprocessingConfig(
        remove_stopwords=False,
        perform_stemming=False,
        perform_lemmatization=False,
        handle_negations=True,
    )


def test_bundle_reuses_training_preprocessing_contract():
    bundle, metrics, _ = train_inference_bundle(
        dataset(), "review", "sentiment", preprocessing=config(), calibrate=False, random_state=7
    )
    raw = "not great product"
    processed = bundle.preprocessing.apply(raw)
    assert bundle.predict(raw)[0] == bundle.model.predict([processed])[0]
    assert bundle.preprocessing.handle_negations is True
    assert metrics["confidence_kind"] == "native_probability"


def test_calibrated_linear_svm_exposes_real_probabilities():
    bundle, metrics, _ = train_inference_bundle(
        dataset(),
        "review",
        "sentiment",
        model_name="Linear SVM",
        preprocessing=config(),
        calibrate=True,
        calibration_method="sigmoid",
        calibration_cv=3,
        random_state=11,
    )
    probabilities = bundle.predict_proba(["great value", "awful waste"])
    assert probabilities.shape == (2, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert metrics["confidence_kind"] == "calibrated_sigmoid"
    assert metrics["log_loss"] is not None
    assert metrics["expected_calibration_error"] is not None


def test_uncalibrated_svm_never_fakes_confidence():
    bundle, metrics, _ = train_inference_bundle(
        dataset(),
        "review",
        "sentiment",
        model_name="Linear SVM",
        preprocessing=config(),
        calibrate=False,
    )
    frame = bundle.predict_frame(["great", "bad"])
    assert frame["confidence"].isna().all()
    assert metrics["confidence_kind"] == "unavailable"
    assert metrics["log_loss"] is None
    with pytest.raises(ValueError, match="calibration enabled"):
        bundle.predict_proba("great")


def test_error_analysis_surfaces_contract_fields():
    bundle, _, errors = train_inference_bundle(
        dataset(), "review", "sentiment", preprocessing=config(), calibrate=True
    )
    assert {
        "review_text",
        "true_sentiment",
        "predicted_sentiment",
        "confidence",
        "correct",
        "error_type",
    }.issubset(errors.columns)
    assert len(errors) == 12
    rebuilt = build_error_analysis(bundle, ["great", "terrible"], ["positive", "negative"])
    assert len(rebuilt) == 2


def test_expected_calibration_error_known_example():
    value = expected_calibration_error(["p", "n"], ["p", "p"], [0.9, 0.9], n_bins=2)
    assert value == pytest.approx(0.4)


def test_expected_calibration_error_rejects_invalid_confidence():
    with pytest.raises(ValueError, match="between 0 and 1"):
        expected_calibration_error(["p"], ["p"], [1.2])


def test_safe_bundle_round_trip_preserves_preprocessing(tmp_path):
    bundle, _, _ = train_inference_bundle(
        dataset(), "review", "sentiment", preprocessing=config(), calibrate=False
    )
    path = save_safe_inference_bundle(bundle, "safe-roundtrip", tmp_path)
    loaded = load_safe_inference_bundle(path)
    assert loaded.preprocessing == bundle.preprocessing
    assert loaded.confidence_kind == bundle.confidence_kind
    assert loaded.predict("excellent product")[0] == bundle.predict("excellent product")[0]


def test_calibration_cv_requires_enough_examples_per_class():
    with pytest.raises(ValueError, match="calibration_cv"):
        train_inference_bundle(
            dataset(rows_per_class=3),
            "review",
            "sentiment",
            preprocessing=config(),
            calibrate=True,
            calibration_cv=3,
            test_size=0.5,
        )

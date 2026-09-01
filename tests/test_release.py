import json
from pathlib import Path

import pandas as pd

from src.inference import PreprocessingConfig, train_inference_bundle
from src.release import (
    EXPECTED_TEST_FINGERPRINT,
    EXPECTED_TRAIN_FINGERPRINT,
    benchmark_evidence_issues,
    dataset_provenance,
    release_issues,
    sha256_bytes,
    verify_artifact_manifest,
    write_release_sidecars,
)
from src.safe_persistence import (
    REVIEWED_SKLEARN_INTERNAL_TYPES,
    inspect_safe_inference_bundle,
    load_safe_inference_bundle,
    save_safe_inference_bundle,
    unapproved_safe_inference_types,
)


def dataset(rows_per_class=20):
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


def test_safe_skops_round_trip_has_no_unknown_types(tmp_path):
    bundle, _, _ = train_inference_bundle(
        dataset(), "review", "sentiment", preprocessing=config(), calibrate=False, random_state=17
    )
    path = save_safe_inference_bundle(bundle, "safe-test", tmp_path)
    assert path.endswith(".inference.skops")
    assert inspect_safe_inference_bundle(path) == ()
    loaded = load_safe_inference_bundle(path)
    assert loaded.preprocessing == bundle.preprocessing
    assert loaded.predict("excellent product")[0] == bundle.predict("excellent product")[0]


def test_safe_skops_round_trip_supports_calibrated_svm(tmp_path):
    bundle, metrics, _ = train_inference_bundle(
        dataset(rows_per_class=30),
        "review",
        "sentiment",
        model_name="Linear SVM",
        preprocessing=config(),
        calibrate=True,
        calibration_method="sigmoid",
        calibration_cv=3,
        random_state=19,
    )
    path = save_safe_inference_bundle(bundle, "calibrated-svm", tmp_path)
    reported = set(inspect_safe_inference_bundle(path))
    assert reported
    assert reported.issubset(REVIEWED_SKLEARN_INTERNAL_TYPES)
    assert unapproved_safe_inference_types(path) == ()
    loaded = load_safe_inference_bundle(path)
    assert loaded.predict_proba(["excellent value", "awful waste"]).shape == (2, 2)
    assert metrics["confidence_kind"] == "calibrated_sigmoid"


def test_dataset_provenance_is_content_addressed():
    raw = b"review,sentiment\ngreat,positive\n"
    provenance = dataset_provenance("uploads/reviews.csv", raw, rows=1)
    assert provenance["source_name"] == "reviews.csv"
    assert provenance["sha256"] == sha256_bytes(raw)
    assert provenance["bytes"] == len(raw)
    assert provenance["rows"] == 1


def test_release_sidecars_record_artifact_and_model_contract(tmp_path):
    bundle, metrics, _ = train_inference_bundle(
        dataset(), "review", "sentiment", preprocessing=config(), calibrate=False, random_state=23
    )
    training_data = dataset_provenance("reviews.csv", b"frozen bytes", rows=40)
    bundle.metadata["training_data"] = training_data
    artifact = save_safe_inference_bundle(bundle, "release-test", tmp_path)
    manifest_path, card_path = write_release_sidecars(
        bundle, metrics, artifact, training_data=training_data
    )

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    assert manifest["artifact"]["format"] == "skops"
    assert len(manifest["artifact"]["sha256"]) == 64
    assert manifest["training_data"]["sha256"] == training_data["sha256"]
    assert manifest["model"]["label_schema"] == "text"
    assert "scikit-learn" in manifest["runtime_versions"]

    ok, message = verify_artifact_manifest(artifact)
    assert ok is True
    assert "matches" in message

    card = Path(card_path).read_text(encoding="utf-8")
    assert "## Intended use" in card
    assert "## Out-of-scope and limitations" in card
    assert "## Training data provenance" in card
    assert "## Evaluation" in card
    assert "static reviewed allowlist" in card
    assert "joblib" not in card.lower()


def test_manifest_verification_detects_artifact_change(tmp_path):
    bundle, metrics, _ = train_inference_bundle(
        dataset(), "review", "sentiment", preprocessing=config(), calibrate=False
    )
    artifact = save_safe_inference_bundle(bundle, "tamper-test", tmp_path)
    write_release_sidecars(bundle, metrics, artifact)
    Path(artifact).write_bytes(Path(artifact).read_bytes() + b"tamper")
    ok, message = verify_artifact_manifest(artifact)
    assert ok is False
    assert "does not match" in message


def valid_benchmark_payload():
    models = {
        "dummy_most_frequent": (0.5, 0.333333, 0.5),
        "multinomial_nb": (0.8878, 0.887799, 0.8878),
        "logistic_regression": (0.907, 0.907, 0.907),
        "linear_svm": (0.9088, 0.9088, 0.9088),
    }
    return {
        "schema_version": 1,
        "benchmark_id": "amazon_polarity_phase2_v1",
        "dataset": {
            "id": "mteb/amazon_polarity",
            "revision": "ec149c1fe36043668a50804214d4597804001f6f",
            "license": "apache-2.0",
            "official_train_rows": 3599994,
            "official_test_rows": 400000,
            "label_contract": {"0": "negative", "1": "positive"},
        },
        "integrity": {
            "train_rows": 50000,
            "train_unique_texts": 50000,
            "train_duplicate_texts": 0,
            "test_rows": 10000,
            "test_unique_texts": 10000,
            "test_duplicate_texts": 0,
            "cross_split_text_overlap": 0,
        },
        "selection": {
            "seed": 42,
            "profile": {
                "name": "phase2",
                "shuffle_buffer": 50000,
                "train_per_class": 25000,
                "test_per_class": 5000,
                "max_features": 50000,
            },
            "train_class_counts": {"negative": 25000, "positive": 25000},
            "test_class_counts": {"negative": 5000, "positive": 5000},
            "train_fingerprint_sha256": EXPECTED_TRAIN_FINGERPRINT,
            "test_fingerprint_sha256": EXPECTED_TEST_FINGERPRINT,
        },
        "features": {
            "type": "tfidf",
            "fit_scope": "train_only",
            "lowercase": True,
            "strip_accents": "unicode",
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_df": 0.98,
            "max_features": 50000,
            "sublinear_tf": True,
        },
        "metrics": {"primary": "macro_f1"},
        "models": {
            name: {
                "accuracy": values[0],
                "macro_f1": values[1],
                "balanced_accuracy": values[2],
                "confusion_matrix": [[2500, 2500], [2500, 2500]],
            }
            for name, values in models.items()
        },
        "winner_by_macro_f1": "linear_svm",
        "runtime": {"datasets": "5.0.1"},
    }


def test_benchmark_evidence_validator_rejects_tampering(tmp_path):
    path = tmp_path / "result.json"
    payload = valid_benchmark_payload()
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert benchmark_evidence_issues(path) == []
    payload["selection"]["train_fingerprint_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert any("training fingerprint" in issue for issue in benchmark_evidence_issues(path))


def _write_candidate_tree(root: Path, *, version="0.1.0"):
    for relative in (
        "README.md",
        "SECURITY.md",
        "LICENSE",
        "CONTRIBUTING.md",
        "docs/INFERENCE.md",
        "docs/MODEL_CARD.md",
        "benchmarks/protocols/amazon_polarity_phase2_v1.json",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("release file\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "x"\nversion = "{version}"\ndependencies = ["skops==0.14.0"]\n',
        encoding="utf-8",
    )


def test_candidate_release_check_rejects_generated_and_compiled_artifacts(tmp_path):
    _write_candidate_tree(tmp_path)
    assert release_issues(tmp_path, mode="candidate") == []
    models = tmp_path / "models"
    models.mkdir()
    (models / "unsafe.joblib").write_bytes(b"not a real model")
    cache = tmp_path / "src" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "module.pyc").write_bytes(b"bytecode")
    issues = release_issues(tmp_path, mode="candidate")
    assert any("generated model artifact" in issue for issue in issues)
    assert any("compiled Python artifact" in issue for issue in issues)


def test_candidate_release_check_rejects_serialized_artifact_outside_models(tmp_path):
    _write_candidate_tree(tmp_path)
    (tmp_path / "unexpected.pkl").write_bytes(b"not a real model")
    issues = release_issues(tmp_path, mode="candidate")
    assert any("generated model artifact" in issue for issue in issues)


def test_release_mode_requires_valid_evidence_and_final_version(tmp_path):
    _write_candidate_tree(tmp_path)
    benchmark = tmp_path / "benchmarks" / "results" / "amazon_polarity_phase2_v1.json"
    benchmark.parent.mkdir(parents=True)
    benchmark.write_text("{}", encoding="utf-8")
    issues = release_issues(tmp_path, mode="release")
    assert any("invalid release benchmark evidence" in issue for issue in issues)
    assert any("version 1.0.0" in issue for issue in issues)


def test_release_mode_accepts_exact_frozen_evidence(tmp_path):
    _write_candidate_tree(tmp_path, version="1.0.0")
    benchmark = tmp_path / "benchmarks" / "results" / "amazon_polarity_phase2_v1.json"
    benchmark.parent.mkdir(parents=True)
    benchmark.write_text(json.dumps(valid_benchmark_payload()), encoding="utf-8")
    assert release_issues(tmp_path, mode="release") == []

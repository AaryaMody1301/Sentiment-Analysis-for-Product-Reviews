import json
from pathlib import Path

import pandas as pd

from src.inference import PreprocessingConfig, train_inference_bundle
from src.release import (
    dataset_provenance,
    release_issues,
    sha256_bytes,
    verify_artifact_manifest,
    write_release_sidecars,
)
from src.safe_persistence import (
    inspect_safe_inference_bundle,
    load_safe_inference_bundle,
    save_safe_inference_bundle,
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
    assert loaded.confidence_kind == bundle.confidence_kind
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
    assert inspect_safe_inference_bundle(path) == ()
    loaded = load_safe_inference_bundle(path)
    probabilities = loaded.predict_proba(["excellent value", "awful waste"])
    assert probabilities.shape == (2, 2)
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
    assert "Legacy `.joblib`" in card


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


def _write_candidate_tree(root: Path):
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
        '[project]\nname = "x"\nversion = "0.1.0"\ndependencies = ["skops==0.14.0"]\n',
        encoding="utf-8",
    )


def test_candidate_release_check_rejects_generated_models(tmp_path):
    _write_candidate_tree(tmp_path)
    assert release_issues(tmp_path, mode="candidate") == []
    models = tmp_path / "models"
    models.mkdir()
    (models / "unsafe.joblib").write_bytes(b"not a real model")
    issues = release_issues(tmp_path, mode="candidate")
    assert any("generated model artifact" in issue for issue in issues)


def test_release_mode_requires_evidence_and_final_version(tmp_path):
    _write_candidate_tree(tmp_path)
    issues = release_issues(tmp_path, mode="release")
    assert any("benchmark evidence" in issue for issue in issues)
    assert any("version 1.0.0" in issue for issue in issues)

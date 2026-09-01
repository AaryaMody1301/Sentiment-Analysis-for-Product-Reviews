from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

from src.inference import InferenceBundle

MANIFEST_SCHEMA_VERSION = 1
RELEASE_BENCHMARK_RESULT = Path("benchmarks/results/amazon_polarity_phase2_v1.json")
RELEASE_BENCHMARK_PROTOCOL = Path("benchmarks/protocols/amazon_polarity_phase2_v1.json")
GENERATED_MODEL_SUFFIXES = {".joblib", ".pkl", ".pickle", ".skops"}
REQUIRED_RELEASE_FILES = (
    Path("README.md"),
    Path("SECURITY.md"),
    Path("LICENSE"),
    Path("CONTRIBUTING.md"),
    Path("pyproject.toml"),
    Path("docs/INFERENCE.md"),
    Path("docs/MODEL_CARD.md"),
    RELEASE_BENCHMARK_PROTOCOL,
)

EXPECTED_BENCHMARK_ID = "amazon_polarity_phase2_v1"
EXPECTED_DATASET_ID = "mteb/amazon_polarity"
EXPECTED_DATASET_REVISION = "ec149c1fe36043668a50804214d4597804001f6f"
EXPECTED_TRAIN_FINGERPRINT = "ec4fc2ad1b734b6d43221fda8a67e6be5162eeec4426921860ff8181e928e944"
EXPECTED_TEST_FINGERPRINT = "00c1205e35fb1e5862e3fe9ea769e1acded6c0089cddb955d37e22ebbe042550"
EXPECTED_MODEL_METRICS = {
    "dummy_most_frequent": {"accuracy": 0.5, "macro_f1": 0.333333, "balanced_accuracy": 0.5},
    "multinomial_nb": {"accuracy": 0.8878, "macro_f1": 0.887799, "balanced_accuracy": 0.8878},
    "logistic_regression": {"accuracy": 0.907, "macro_f1": 0.907, "balanced_accuracy": 0.907},
    "linear_svm": {"accuracy": 0.9088, "macro_f1": 0.9088, "balanced_accuracy": 0.9088},
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_provenance(name: str, data: bytes, *, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source_name": Path(name).name,
        "sha256": sha256_bytes(data),
        "bytes": len(data),
    }
    if rows is not None:
        result["rows"] = int(rows)
    return result


def runtime_versions() -> dict[str, str]:
    packages = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit-learn": "scikit-learn",
        "nltk": "nltk",
        "skops": "skops",
        "streamlit": "streamlit",
    }
    versions = {"python": sys.version.split()[0]}
    for display_name, distribution in packages.items():
        try:
            versions[display_name] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[display_name] = "not-installed"
    return versions


def current_code_revision() -> str | None:
    env_sha = os.environ.get("GITHUB_SHA")
    if env_sha:
        return env_sha
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def build_release_manifest(
    bundle: InferenceBundle,
    metrics: dict[str, Any] | None,
    artifact_path: str | Path,
    *,
    training_data: dict[str, Any] | None = None,
    benchmark_result: str | Path | None = None,
    code_revision: str | None = None,
) -> dict[str, Any]:
    artifact = Path(artifact_path)
    if not artifact.is_file():
        raise FileNotFoundError(artifact)

    benchmark: dict[str, Any] | None = None
    if benchmark_result is not None:
        benchmark_path = Path(benchmark_result)
        if benchmark_path.is_file():
            benchmark = {
                "path": benchmark_path.as_posix(),
                "sha256": sha256_file(benchmark_path),
            }

    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact": {
            "path": artifact.name,
            "format": artifact.suffix.lstrip("."),
            "sha256": sha256_file(artifact),
            "bytes": artifact.stat().st_size,
        },
        "model": {
            "name": bundle.model_name,
            "classes": list(bundle.classes),
            "label_schema": bundle.label_schema,
            "confidence_kind": bundle.confidence_kind,
            "calibration_method": bundle.calibration_method,
            "random_state": bundle.random_state,
            "bundle_schema_version": bundle.schema_version,
            "metadata": dict(bundle.metadata),
        },
        "training_data": training_data or bundle.metadata.get("training_data"),
        "evaluation": metrics or {},
        "benchmark_evidence": benchmark,
        "code_revision": code_revision or current_code_revision(),
        "runtime_versions": runtime_versions(),
    }


def _metric_line(metrics: dict[str, Any], key: str, label: str) -> str:
    value = metrics.get(key)
    if value is None:
        return f"- {label}: not available"
    if isinstance(value, (int, float)):
        return f"- {label}: {value:.6f}"
    return f"- {label}: {value}"


def render_model_card(manifest: dict[str, Any]) -> str:
    model = manifest["model"]
    metrics = manifest.get("evaluation") or {}
    training_data = manifest.get("training_data") or {}
    artifact = manifest["artifact"]
    benchmark = manifest.get("benchmark_evidence")
    benchmark_line = (
        f"Benchmark evidence: `{benchmark['path']}` (SHA-256 `{benchmark['sha256']}`)."
        if benchmark
        else "Benchmark evidence was not attached to this artifact. Do not treat holdout metrics as a repository-wide benchmark claim."
    )
    lines = [
        f"# Model Card: {model['name']}",
        "",
        "## Summary",
        "Classical product-review sentiment classifier packaged with an immutable preprocessing contract and explicit confidence semantics.",
        "",
        "## Intended use",
        "- Analyze product-review sentiment in exploratory or application workflows.",
        "- Use only with label semantics compatible with the recorded label schema.",
        "- Treat confidence as probabilistic only when `confidence_kind` records a native or calibrated probability.",
        "",
        "## Out-of-scope and limitations",
        "- Not validated for medical, legal, safety-critical, or high-stakes automated decisions.",
        "- Performance can shift across languages, domains, writing styles, class balance, and time.",
        "- Sarcasm, mixed sentiment, context-dependent language, and distribution shift remain known failure modes.",
        "- A local holdout score is not evidence of performance on the full Amazon Polarity dataset.",
        "",
        "## Training data provenance",
        f"- Source name: {training_data.get('source_name', 'not recorded')}",
        f"- SHA-256: {training_data.get('sha256', 'not recorded')}",
        f"- Rows: {training_data.get('rows', 'not recorded')}",
        "",
        "## Model and inference contract",
        f"- Model: {model['name']}",
        f"- Classes: {', '.join(model['classes'])}",
        f"- Label schema: {model['label_schema']}",
        f"- Confidence kind: {model['confidence_kind']}",
        f"- Calibration method: {model.get('calibration_method') or 'none'}",
        f"- Random seed: {model['random_state']}",
        "",
        "## Evaluation",
        _metric_line(metrics, "macro_f1", "Macro F1"),
        _metric_line(metrics, "accuracy", "Accuracy"),
        _metric_line(metrics, "balanced_accuracy", "Balanced accuracy"),
        _metric_line(metrics, "log_loss", "Log loss"),
        _metric_line(metrics, "expected_calibration_error", "Expected calibration error"),
        "",
        benchmark_line,
        "",
        "## Artifact and reproducibility",
        f"- Artifact: `{artifact['path']}`",
        f"- Artifact SHA-256: `{artifact['sha256']}`",
        f"- Code revision: `{manifest.get('code_revision') or 'not recorded'}`",
        "- Runtime versions are recorded in the companion manifest JSON.",
        "",
        "## Security",
        "The application loads preferred `.skops` artifacts only after serialized-type inspection. Types outside the static reviewed allowlist are rejected.",
        "",
    ]
    return "\n".join(lines)


def verify_artifact_manifest(artifact_path: str | Path) -> tuple[bool, str]:
    artifact = Path(artifact_path)
    manifest_path = artifact.with_name(f"{artifact.name}.manifest.json")
    if not manifest_path.is_file():
        return False, "companion manifest is missing"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"could not read companion manifest: {exc}"
    expected = manifest.get("artifact", {}).get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        return False, "companion manifest does not contain a valid artifact SHA-256"
    actual = sha256_file(artifact)
    if actual != expected:
        return False, "artifact SHA-256 does not match the companion manifest"
    return True, "artifact SHA-256 matches the companion manifest"


def write_release_sidecars(
    bundle: InferenceBundle,
    metrics: dict[str, Any] | None,
    artifact_path: str | Path,
    *,
    training_data: dict[str, Any] | None = None,
    benchmark_result: str | Path | None = None,
) -> tuple[str, str]:
    artifact = Path(artifact_path)
    manifest = build_release_manifest(
        bundle,
        metrics,
        artifact,
        training_data=training_data,
        benchmark_result=benchmark_result,
    )
    manifest_path = artifact.with_name(f"{artifact.name}.manifest.json")
    card_path = artifact.with_name(f"{artifact.name}.model-card.md")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    card_path.write_text(render_model_card(manifest), encoding="utf-8")
    return str(manifest_path), str(card_path)


def _project_version(root: Path) -> str | None:
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return None
    with pyproject.open("rb") as handle:
        data = tomllib.load(handle)
    return data.get("project", {}).get("version")


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, str(exc)
    if not isinstance(value, dict):
        return None, "top-level value must be a JSON object"
    return value, None


def benchmark_evidence_issues(path: str | Path) -> list[str]:
    """Validate the exact frozen evidence used for the v1.0.0 performance claim."""

    benchmark_path = Path(path)
    if not benchmark_path.is_file():
        return [f"benchmark evidence is missing: {benchmark_path.as_posix()}"]
    data, error = _read_json(benchmark_path)
    if data is None:
        return [f"benchmark evidence is not valid JSON: {error}"]

    issues: list[str] = []

    def expect(actual: object, expected: object, label: str) -> None:
        if actual != expected:
            issues.append(f"{label} must be {expected!r}; found {actual!r}")

    expect(data.get("schema_version"), 1, "benchmark schema_version")
    expect(data.get("benchmark_id"), EXPECTED_BENCHMARK_ID, "benchmark_id")

    dataset = data.get("dataset") if isinstance(data.get("dataset"), dict) else {}
    expect(dataset.get("id"), EXPECTED_DATASET_ID, "dataset id")
    expect(dataset.get("revision"), EXPECTED_DATASET_REVISION, "dataset revision")
    expect(dataset.get("license"), "apache-2.0", "dataset license")
    expect(dataset.get("official_train_rows"), 3_599_994, "official train rows")
    expect(dataset.get("official_test_rows"), 400_000, "official test rows")
    expect(dataset.get("label_contract"), {"0": "negative", "1": "positive"}, "label contract")

    integrity = data.get("integrity") if isinstance(data.get("integrity"), dict) else {}
    expected_integrity = {
        "train_rows": 50_000,
        "train_unique_texts": 50_000,
        "train_duplicate_texts": 0,
        "test_rows": 10_000,
        "test_unique_texts": 10_000,
        "test_duplicate_texts": 0,
        "cross_split_text_overlap": 0,
    }
    for key, expected in expected_integrity.items():
        expect(integrity.get(key), expected, f"integrity.{key}")

    selection = data.get("selection") if isinstance(data.get("selection"), dict) else {}
    expect(selection.get("seed"), 42, "selection seed")
    expect(
        selection.get("train_class_counts"),
        {"negative": 25_000, "positive": 25_000},
        "training class counts",
    )
    expect(
        selection.get("test_class_counts"),
        {"negative": 5_000, "positive": 5_000},
        "test class counts",
    )
    expect(
        selection.get("train_fingerprint_sha256"),
        EXPECTED_TRAIN_FINGERPRINT,
        "training fingerprint",
    )
    expect(
        selection.get("test_fingerprint_sha256"),
        EXPECTED_TEST_FINGERPRINT,
        "test fingerprint",
    )
    profile = selection.get("profile") if isinstance(selection.get("profile"), dict) else {}
    expected_profile = {
        "name": "phase2",
        "shuffle_buffer": 50_000,
        "train_per_class": 25_000,
        "test_per_class": 5_000,
        "max_features": 50_000,
    }
    for key, expected in expected_profile.items():
        expect(profile.get(key), expected, f"selection.profile.{key}")

    features = data.get("features") if isinstance(data.get("features"), dict) else {}
    expected_features = {
        "type": "tfidf",
        "fit_scope": "train_only",
        "lowercase": True,
        "strip_accents": "unicode",
        "ngram_range": [1, 2],
        "min_df": 2,
        "max_df": 0.98,
        "max_features": 50_000,
        "sublinear_tf": True,
    }
    for key, expected in expected_features.items():
        expect(features.get(key), expected, f"features.{key}")

    metric_contract = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
    expect(metric_contract.get("primary"), "macro_f1", "primary metric")

    models = data.get("models") if isinstance(data.get("models"), dict) else {}
    for model_name, expected_metrics in EXPECTED_MODEL_METRICS.items():
        model = models.get(model_name)
        if not isinstance(model, dict):
            issues.append(f"missing benchmark model result: {model_name}")
            continue
        for metric_name, expected in expected_metrics.items():
            actual = model.get(metric_name)
            if not isinstance(actual, (int, float)) or abs(float(actual) - expected) > 1e-9:
                issues.append(
                    f"{model_name}.{metric_name} must be {expected}; found {actual!r}"
                )
        matrix = model.get("confusion_matrix")
        if (
            not isinstance(matrix, list)
            or len(matrix) != 2
            or any(not isinstance(row, list) or len(row) != 2 for row in matrix)
            or sum(sum(int(value) for value in row) for row in matrix) != 10_000
        ):
            issues.append(f"{model_name}.confusion_matrix must be a 2x2 matrix totaling 10000")

    expect(data.get("winner_by_macro_f1"), "linear_svm", "winner_by_macro_f1")
    runtime = data.get("runtime") if isinstance(data.get("runtime"), dict) else {}
    expect(runtime.get("datasets"), "5.0.1", "benchmark datasets version")
    return issues


def _tracked_paths(base: Path) -> list[Path]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(base), "ls-files"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        completed = None
    if completed is not None and completed.returncode == 0:
        return [base / line for line in completed.stdout.splitlines() if line.strip()]
    return [path for path in base.rglob("*") if path.is_file()]


def release_issues(root: str | Path = ".", *, mode: str = "candidate") -> list[str]:
    if mode not in {"candidate", "release"}:
        raise ValueError("mode must be candidate or release")
    base = Path(root)
    issues: list[str] = []

    for relative in REQUIRED_RELEASE_FILES:
        if not (base / relative).is_file():
            issues.append(f"missing required release file: {relative.as_posix()}")

    for path in sorted(_tracked_paths(base)):
        try:
            relative = path.relative_to(base)
        except ValueError:
            relative = path
        normalized = relative.as_posix()
        if path.suffix.lower() in GENERATED_MODEL_SUFFIXES and "models/" in f"/{normalized}":
            issues.append(f"generated model artifact is tracked in repository tree: {relative}")
        if path.suffix.lower() == ".pyc" or "__pycache__" in relative.parts:
            issues.append(f"compiled Python artifact is tracked in repository tree: {relative}")

    legacy_app = base / "src/app.py"
    if legacy_app.exists():
        issues.append("legacy src/app.py monolith must be retired before release")

    pyproject = base / "pyproject.toml"
    if pyproject.is_file() and "skops==0.14.0" not in pyproject.read_text(encoding="utf-8"):
        issues.append("pyproject.toml must pin skops==0.14.0 for the release artifact contract")
    readme = base / "README.md"
    if readme.is_file() and "92%" in readme.read_text(encoding="utf-8"):
        issues.append("README contains the unsupported legacy 92% performance claim")

    if mode == "release":
        benchmark = base / RELEASE_BENCHMARK_RESULT
        issues.extend(
            f"invalid release benchmark evidence: {issue}"
            for issue in benchmark_evidence_issues(benchmark)
        )
        version = _project_version(base)
        if version != "1.0.0":
            issues.append(f"release mode requires project version 1.0.0; found {version!r}")
    return issues


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate release-hardening requirements.")
    parser.add_argument("--check", choices=["candidate", "release"], default="candidate")
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    issues = release_issues(args.root, mode=args.check)
    if issues:
        print(f"Release {args.check} check failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print(f"Release {args.check} check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

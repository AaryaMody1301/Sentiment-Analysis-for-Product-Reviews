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
GENERATED_MODEL_SUFFIXES = {".joblib", ".pkl", ".pickle", ".skops"}
REQUIRED_RELEASE_FILES = (
    Path("README.md"),
    Path("SECURITY.md"),
    Path("LICENSE"),
    Path("CONTRIBUTING.md"),
    Path("pyproject.toml"),
    Path("docs/INFERENCE.md"),
    Path("docs/MODEL_CARD.md"),
    Path("benchmarks/protocols/amazon_polarity_phase2_v1.json"),
)


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
        "python": None,
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit-learn": "scikit-learn",
        "nltk": "nltk",
        "joblib": "joblib",
        "skops": "skops",
        "streamlit": "streamlit",
    }
    versions = {"python": sys.version.split()[0]}
    for display_name, distribution in packages.items():
        if distribution is None:
            continue
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
            benchmark = {"path": benchmark_path.as_posix(), "sha256": sha256_file(benchmark_path)}

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
        f"# Model Card: {model['name']}", "", "## Summary",
        "Classical product-review sentiment classifier packaged with an immutable preprocessing contract and explicit confidence semantics.",
        "", "## Intended use",
        "- Analyze product-review sentiment in exploratory or application workflows.",
        "- Use only with label semantics compatible with the recorded label schema.",
        "- Treat confidence as probabilistic only when `confidence_kind` records a native or calibrated probability.",
        "", "## Out-of-scope and limitations",
        "- Not validated for medical, legal, safety-critical, or high-stakes automated decisions.",
        "- Performance can shift across languages, domains, writing styles, class balance, and time.",
        "- Sarcasm, mixed sentiment, context-dependent language, and distribution shift remain known failure modes.",
        "- A local holdout score is not evidence of performance on the full Amazon Polarity dataset.",
        "", "## Training data provenance",
        f"- Source name: {training_data.get('source_name', 'not recorded')}",
        f"- SHA-256: {training_data.get('sha256', 'not recorded')}",
        f"- Rows: {training_data.get('rows', 'not recorded')}",
        "", "## Model and inference contract",
        f"- Model: {model['name']}",
        f"- Classes: {', '.join(model['classes'])}",
        f"- Label schema: {model['label_schema']}",
        f"- Confidence kind: {model['confidence_kind']}",
        f"- Calibration method: {model.get('calibration_method') or 'none'}",
        f"- Random seed: {model['random_state']}",
        "", "## Evaluation",
        _metric_line(metrics, "macro_f1", "Macro F1"),
        _metric_line(metrics, "accuracy", "Accuracy"),
        _metric_line(metrics, "balanced_accuracy", "Balanced accuracy"),
        _metric_line(metrics, "log_loss", "Log loss"),
        _metric_line(metrics, "expected_calibration_error", "Expected calibration error"),
        "", benchmark_line, "", "## Artifact and reproducibility",
        f"- Artifact: `{artifact['path']}`",
        f"- Artifact SHA-256: `{artifact['sha256']}`",
        f"- Code revision: `{manifest.get('code_revision') or 'not recorded'}`",
        "- Runtime versions are recorded in the companion manifest JSON.",
        "", "## Security",
        "The preferred `.skops` artifact is inspected for serialized types that are not trusted by default before this application loads it. Legacy `.joblib` artifacts use pickle semantics and must only be loaded from fully trusted sources.",
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
    manifest = build_release_manifest(bundle, metrics, artifact, training_data=training_data, benchmark_result=benchmark_result)
    manifest_path = artifact.with_name(f"{artifact.name}.manifest.json")
    card_path = artifact.with_name(f"{artifact.name}.model-card.md")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    card_path.write_text(render_model_card(manifest), encoding="utf-8")
    return str(manifest_path), str(card_path)


def _project_version(root: Path) -> str | None:
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return None
    with pyproject.open("rb") as handle:
        data = tomllib.load(handle)
    return data.get("project", {}).get("version")


def release_issues(root: str | Path = ".", *, mode: str = "candidate") -> list[str]:
    if mode not in {"candidate", "release"}:
        raise ValueError("mode must be candidate or release")
    base = Path(root)
    issues: list[str] = []
    for relative in REQUIRED_RELEASE_FILES:
        if not (base / relative).is_file():
            issues.append(f"missing required release file: {relative.as_posix()}")

    tracked_artifacts: list[Path] = []
    try:
        completed = subprocess.run(["git", "-C", str(base), "ls-files", "models"], check=False, capture_output=True, text=True, timeout=2)
    except (OSError, subprocess.SubprocessError):
        completed = None
    if completed is not None and completed.returncode == 0:
        tracked_artifacts = [base / line for line in completed.stdout.splitlines() if line.strip()]
    else:
        models_dir = base / "models"
        if models_dir.is_dir():
            tracked_artifacts = [path for path in models_dir.rglob("*") if path.is_file()]
    for path in sorted(tracked_artifacts):
        if path.suffix.lower() in GENERATED_MODEL_SUFFIXES:
            try:
                relative = path.relative_to(base)
            except ValueError:
                relative = path
            issues.append(f"generated model artifact is tracked in repository tree: {relative}")

    pyproject = base / "pyproject.toml"
    if pyproject.is_file() and "skops==0.14.0" not in pyproject.read_text(encoding="utf-8"):
        issues.append("pyproject.toml must pin skops==0.14.0 for the release artifact contract")
    readme = base / "README.md"
    if readme.is_file() and "92%" in readme.read_text(encoding="utf-8"):
        issues.append("README contains the unsupported legacy 92% performance claim")

    if mode == "release":
        benchmark = base / RELEASE_BENCHMARK_RESULT
        if not benchmark.is_file():
            issues.append(f"release benchmark evidence is missing: {RELEASE_BENCHMARK_RESULT.as_posix()}")
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

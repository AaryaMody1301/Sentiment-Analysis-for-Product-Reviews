from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import skops.io as sio

from src.inference import BUNDLE_SCHEMA_VERSION, InferenceBundle, PreprocessingConfig

SAFE_ARTIFACT_SCHEMA_VERSION = 1

# skops 0.14 reports these fitted scikit-learn calibration/CV implementation
# classes as unknown even though they are part of the framework model produced
# by CalibratedClassifierCV. Keep this static and exact: never trust the entire
# set returned by get_untrusted_types(). New names must be reviewed explicitly.
REVIEWED_SKLEARN_INTERNAL_TYPES = frozenset(
    {
        "sklearn.calibration._CalibratedClassifier",
        "sklearn.calibration._SigmoidCalibration",
        "sklearn.calibration._TemperatureScaling",
        "sklearn.model_selection._split.StratifiedKFold",
    }
)


def _safe_name(name: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip()).strip("._")
    if not value:
        raise ValueError("Invalid inference bundle name.")
    return value.lower()


def _payload(bundle: InferenceBundle) -> dict[str, Any]:
    """Convert a bundle to builtins + sklearn objects for skops persistence."""
    return {
        "artifact_schema_version": SAFE_ARTIFACT_SCHEMA_VERSION,
        "bundle_schema_version": int(bundle.schema_version),
        "model": bundle.model,
        "model_name": bundle.model_name,
        "preprocessing": asdict(bundle.preprocessing),
        "label_schema": bundle.label_schema,
        "calibration_method": bundle.calibration_method,
        "random_state": int(bundle.random_state),
        "metadata": dict(bundle.metadata),
    }


def inspect_safe_inference_bundle(path: str | Path) -> tuple[str, ...]:
    """Return every type that skops does not trust by default."""
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    return tuple(sorted(sio.get_untrusted_types(file=artifact_path)))


def reviewed_safe_inference_types(path: str | Path) -> tuple[str, ...]:
    return tuple(
        value
        for value in inspect_safe_inference_bundle(path)
        if value in REVIEWED_SKLEARN_INTERNAL_TYPES
    )


def unapproved_safe_inference_types(path: str | Path) -> tuple[str, ...]:
    return tuple(
        value
        for value in inspect_safe_inference_bundle(path)
        if value not in REVIEWED_SKLEARN_INTERNAL_TYPES
    )


def save_safe_inference_bundle(
    bundle: InferenceBundle,
    bundle_name: str,
    models_dir: str | Path = "models",
) -> str:
    """Persist with skops and reject every non-reviewed serialized type."""
    directory = Path(models_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_safe_name(bundle_name)}.inference.skops"
    sio.dump(_payload(bundle), path)

    unapproved = unapproved_safe_inference_types(path)
    if unapproved:
        path.unlink(missing_ok=True)
        joined = ", ".join(unapproved)
        raise RuntimeError(
            "The generated skops artifact contains serialized types outside the reviewed "
            f"allowlist: {joined}. The file was removed instead of expanding trust automatically."
        )
    return str(path)


def _validate_payload(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("The artifact payload is not a mapping.")

    required = {
        "artifact_schema_version",
        "bundle_schema_version",
        "model",
        "model_name",
        "preprocessing",
        "label_schema",
        "calibration_method",
        "random_state",
        "metadata",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"The artifact is missing required fields: {', '.join(missing)}")
    if payload["artifact_schema_version"] != SAFE_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported safe artifact schema "
            f"{payload['artifact_schema_version']}; expected {SAFE_ARTIFACT_SCHEMA_VERSION}."
        )
    if payload["bundle_schema_version"] != BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported inference bundle schema "
            f"{payload['bundle_schema_version']}; expected {BUNDLE_SCHEMA_VERSION}."
        )
    if not isinstance(payload["preprocessing"], dict):
        raise TypeError("The preprocessing contract must be a mapping.")
    if not isinstance(payload["metadata"], dict):
        raise TypeError("The metadata field must be a mapping.")
    return payload


def load_safe_inference_bundle(path: str | Path) -> InferenceBundle:
    """Load only default-trusted types plus the static reviewed sklearn allowlist."""
    artifact_path = Path(path)
    unapproved = unapproved_safe_inference_types(artifact_path)
    if unapproved:
        joined = ", ".join(unapproved)
        raise ValueError(
            "Refusing to load this artifact because it contains serialized types outside "
            f"the reviewed allowlist: {joined}. Review those types before changing policy."
        )

    reviewed = reviewed_safe_inference_types(artifact_path)
    payload = _validate_payload(
        sio.load(artifact_path, trusted=list(reviewed) if reviewed else None)
    )
    preprocessing = PreprocessingConfig(**payload["preprocessing"])
    return InferenceBundle(
        model=payload["model"],
        model_name=str(payload["model_name"]),
        preprocessing=preprocessing,
        label_schema=str(payload["label_schema"]),
        calibration_method=(
            None if payload["calibration_method"] is None else str(payload["calibration_method"])
        ),
        random_state=int(payload["random_state"]),
        metadata=dict(payload["metadata"]),
        schema_version=int(payload["bundle_schema_version"]),
    )

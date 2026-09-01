# Security Policy

## Model artifacts

The supported v1 inference format is `*.inference.skops` with `skops==0.14.0`. Before loading one, the application calls `skops.io.get_untrusted_types()` and compares the result with a static, exact allowlist of reviewed scikit-learn calibration implementation types.

The allowlist is intentionally narrow: `_CalibratedClassifier`, `_SigmoidCalibration`, `_TemperatureScaling`, and `StratifiedKFold`. These are framework objects created by this project's calibrated training path. Any other reported type is rejected. The application never trusts the full set requested by an arbitrary artifact automatically.

`skops` reduces the code-execution risk associated with pickle-family persistence, but an artifact should still come from a trusted source with reviewed provenance. A newly reported serialized type must be reviewed explicitly before the static policy changes.

The supported Streamlit application does **not** expose `.joblib`, `.pkl`, or `.pickle` model loading.

## Data and provenance

CSV parsing fails explicitly rather than silently skipping malformed rows. Numeric sentiment labels must follow an explicit or safely inferred schema, and numeric schemas reject fractional/non-finite values rather than coercing them to another class.

Inference bundles can be accompanied by a JSON manifest and model card recording the artifact hash, training-data fingerprint, model/inference contract, evaluation results, code revision when discoverable, and runtime dependency versions. A hash identifies content; it does not establish that an untrusted source is safe.

## Repository hygiene

Generated model artifacts, compiled Python bytecode, logs, local datasets, and temporary archives must not be committed. The release gate checks the tracked tree for serialized models and `__pycache__`/`.pyc` artifacts.

## Reporting

Do not publish exploit details or malicious serialized artifacts in a public issue. Contact the repository owner privately with a minimal reproduction and the affected commit.

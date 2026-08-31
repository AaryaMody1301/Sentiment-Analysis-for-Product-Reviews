# Security Policy

## Preferred model artifacts

Phase 4 uses `skops==0.14.0` for preferred inference artifacts (`*.inference.skops`). Before loading one, the application calls `skops.io.get_untrusted_types()` and compares the result with a static, exact allowlist of reviewed scikit-learn calibration implementation types.

The allowlist is intentionally narrow: `_CalibratedClassifier`, `_SigmoidCalibration`, `_TemperatureScaling`, and `StratifiedKFold`. These are framework objects created by this project's `CalibratedClassifierCV` training path. Any other type reported by skops is rejected. The application never trusts the full set returned by an arbitrary artifact automatically.

`skops` is safer than pickle-based persistence, but it is still a deserialization format. Review an artifact's source and provenance before using it. A future new type must be reviewed explicitly before the static policy changes.

## Legacy joblib compatibility

The project retains `.joblib` loading only for backwards compatibility with locally created models. Joblib uses Python pickle semantics and loading a malicious artifact can execute arbitrary code.

Only load a `.joblib`, `.pkl`, or `.pickle` file if you created it yourself or otherwise fully trust its origin. Never accept legacy serialized model files from untrusted users. Generated model artifacts are ignored by Git and are not part of the source release.

## Data and provenance

CSV parsing fails explicitly rather than silently skipping malformed rows. Numeric sentiment labels must follow an explicit or safely inferred schema; unknown labels are rejected instead of being converted to a default class.

Preferred inference bundles can be accompanied by a JSON manifest and model card recording the artifact hash, training-data fingerprint, model/inference contract, evaluation results, code revision when discoverable, and runtime dependency versions. A hash identifies content; it does not establish that an untrusted source is safe.

## Reporting

Do not publish exploit details or malicious serialized artifacts in a public issue. Contact the repository owner privately with a minimal reproduction and the affected commit.

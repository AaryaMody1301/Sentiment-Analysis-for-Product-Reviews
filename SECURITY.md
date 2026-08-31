# Security Policy

## Preferred model artifacts

Phase 4 uses `skops==0.14.0` for preferred inference artifacts (`*.inference.skops`). Before the application loads one of these files, it calls `skops.io.get_untrusted_types()` and refuses the artifact if any serialized type is not trusted by default. The application does not automatically approve unknown types.

`skops` is safer than pickle-based persistence, but it is still a deserialization format. Review an artifact's source and provenance before using it, and do not weaken the trust check merely to make an unfamiliar file load.

## Legacy joblib compatibility

The project retains `.joblib` loading only for backwards compatibility with locally created models. Joblib uses Python pickle semantics and loading a malicious artifact can execute arbitrary code.

Only load a `.joblib`, `.pkl`, or `.pickle` file if you created it yourself or otherwise fully trust its origin. Never accept legacy serialized model files from untrusted users. Generated model artifacts are ignored by Git and are not part of the source release.

## Data and provenance

CSV parsing fails explicitly rather than silently skipping malformed rows. Numeric sentiment labels must follow an explicit or safely inferred schema; unknown labels are rejected instead of being converted to a default class.

Preferred inference bundles can be accompanied by a JSON manifest and model card recording the artifact hash, training-data fingerprint, model/inference contract, evaluation results, code revision when discoverable, and runtime dependency versions. A hash identifies content; it does not establish that an untrusted source is safe.

## Reporting

Do not publish exploit details or malicious serialized artifacts in a public issue. Contact the repository owner privately with a minimal reproduction and the affected commit.

# Security Policy

## Model artifacts

This project currently keeps `joblib` compatibility for locally trained scikit-learn models. `joblib` uses Python pickle semantics and can execute arbitrary code while loading an artifact.

Only load a `.joblib` file if you created it yourself or otherwise fully trust its origin. Do not accept model files from untrusted users and do not commit generated model artifacts to this repository.

## Datasets

CSV parsing fails explicitly rather than silently skipping malformed rows. Numeric sentiment labels must follow an explicit or safely inferred schema; unknown labels are rejected instead of being converted to a default class.

## Reporting

Do not publish exploit details or malicious serialized artifacts in a public issue. Contact the repository owner privately with a minimal reproduction and affected commit.

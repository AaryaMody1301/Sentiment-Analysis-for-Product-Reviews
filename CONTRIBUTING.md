# Contributing to Sentiment Analysis for Product Reviews

Contributions should preserve the repository's evidence-first and security-first contracts: explicit label semantics, deterministic evaluation where practical, train-only fitting, honest confidence semantics, and no committed generated model artifacts.

## Development setup

```bash
git clone https://github.com/AaryaMody1301/Sentiment-Analysis-for-Product-Reviews.git
cd Sentiment-Analysis-for-Product-Reviews
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the same checks used by CI:

```bash
python -m compileall -q src main.py pages tests
python -m pytest -q
python -m src.release --check candidate
python -m pip check
```

The external Phase 2 benchmark needs the optional benchmark dependencies:

```bash
python -m pip install -e ".[benchmark]"
python -m src.benchmarking --profile phase2
```

Do not use the smoke benchmark for performance claims.

## Pull requests

Keep changes focused and add regression tests for behavior changes. Update documentation when an interface, persistence contract, label contract, benchmark protocol, or security assumption changes.

Do not commit generated `.joblib`, `.pkl`, `.pickle`, or `.skops` model artifacts. Preferred `.skops` artifacts and their provenance sidecars are runtime outputs, not source files.

Do not loosen the safe persistence loader by automatically trusting types reported by `skops.io.get_untrusted_types()`. Unknown types require explicit human review outside the application.

## Evidence and performance claims

`datasets/sample_reviews.csv` is demonstration data only. Performance claims must point to reproducible benchmark evidence with an immutable dataset revision, selection protocol, split-integrity checks, sample fingerprints, metrics, and runtime metadata.

A local holdout score from an uploaded CSV must not be described as a score on the full Amazon Polarity dataset.

## Security reports

Do not attach malicious serialized artifacts to public issues. Follow the reporting guidance in `SECURITY.md`.

# Model Card Contract

This repository is a training and inference application, not a repository for one pre-trained release model. It therefore does not publish a single performance-bearing model card in source control.

Phase 4 generates a model card and a JSON provenance manifest next to every preferred `.inference.skops` artifact. Those sidecars are local runtime artifacts and are intentionally ignored by Git together with the model itself.

## Required per-artifact fields

Every generated model card records:

- intended use and explicit out-of-scope/high-stakes use;
- known limitations and distribution-shift cautions;
- training-data source name, byte size, row count when known, and SHA-256 fingerprint;
- model family, classes, label schema, preprocessing contract, random seed, and confidence semantics;
- holdout metrics produced by the inference training workflow;
- benchmark evidence only when an immutable benchmark result is explicitly attached;
- artifact SHA-256, code revision when discoverable, and a companion runtime-version manifest;
- the persistence security boundary between preferred `.skops` artifacts and legacy pickle/joblib compatibility files.

Preferred artifacts are inspected with `skops.io.get_untrusted_types()`. The loader accepts default-trusted types plus only the exact scikit-learn calibration/CV implementation names documented in `SECURITY.md`; any other reported type is rejected rather than automatically trusted.

## Evidence rules

A model card must not turn a local holdout result into a claim about the full Amazon Polarity dataset. Repository-level benchmark claims require the committed Phase 2 result, its frozen dataset revision, selection protocol, sample fingerprints, integrity checks, and runtime metadata.

Generated model cards are descriptive provenance records, not guarantees that a model is appropriate for a new domain or decision context.

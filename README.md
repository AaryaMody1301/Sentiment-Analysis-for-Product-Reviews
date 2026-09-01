# Sentiment Analysis for Product Reviews

A Streamlit-based NLP application for exploring, training, comparing, and using classical sentiment classifiers on product-review data with explicit correctness, evidence, inference, and artifact-security contracts.

## Current status

**v1.0.0 release-ready.** Phases 1-5 are complete and the full release candidate has passed the frozen external benchmark, normal CI, and the strict release gate on Python 3.11 and 3.13.

The application includes deterministic training/evaluation, explicit numeric-label schemas, reliable inference bundles, calibrated confidence, error analysis, preferred `.skops` persistence, artifact provenance/model cards, CI on Python 3.11 and 3.13, and release validation.

## Label contracts

Numeric labels are not interpreted by value alone:

| Schema | Meaning |
| --- | --- |
| `binary_01` | `0 = negative`, `1 = positive` |
| `stars_1_to_5` | `1-2 = negative`, `3 = neutral`, `4-5 = positive` |

`auto` inference accepts binary values only in clear label/sentiment-style columns and star values only in rating/star/score-style columns. Ambiguous numeric columns raise an error instead of risking incorrect ground truth. Unknown text labels are rejected rather than silently converted to neutral.

## Models

The application supports Logistic Regression, Multinomial Naive Bayes, Linear SVM, and Random Forest. TF-IDF is used for normal training and HashingVectorizer supports bounded-memory processing.

The frozen external benchmark uses Dummy Classifier, Multinomial Naive Bayes, Logistic Regression, and Linear SVM. Random Forest is intentionally outside the sparse-text benchmark because that benchmark prioritizes strong, resource-efficient classical baselines.

## Evidence-backed Amazon Polarity benchmark

The benchmark source is MTEB `amazon_polarity`, pinned to revision `ec149c1fe36043668a50804214d4597804001f6f`. The frozen profile uses 50,000 balanced training reviews and 10,000 balanced test reviews with seed `42`, train-only TF-IDF, and up to 50,000 unigram/bigram features.

| Model | Accuracy | Macro F1 | Balanced accuracy |
| --- | ---: | ---: | ---: |
| Dummy, most frequent | 0.5000 | 0.333333 | 0.5000 |
| Multinomial Naive Bayes | 0.8878 | 0.887799 | 0.8878 |
| Logistic Regression | 0.9070 | 0.907000 | 0.9070 |
| **Linear SVM** | **0.9088** | **0.908800** | **0.9088** |

Linear SVM ranks first by macro F1, but only by `0.0018` over Logistic Regression. That is treated as a narrow result on this frozen subset, not evidence of universal superiority.

The successful GitHub Actions evidence run was `33471236375` on September 1, 2026. Integrity checks found zero duplicate texts in either selected split and zero train/test text overlap. The selected sequence fingerprints are:

- train: `ec4fc2ad1b734b6d43221fda8a67e6be5162eeec4426921860ff8181e928e944`;
- test: `00c1205e35fb1e5862e3fe9ea769e1acded6c0089cddb955d37e22ebbe042550`.

The complete result, including per-class metrics, confusion matrices, runtime versions, and selection metadata, is committed at [`benchmarks/results/amazon_polarity_phase2_v1.json`](benchmarks/results/amazon_polarity_phase2_v1.json). See [`benchmarks/README.md`](benchmarks/README.md) and the frozen [`protocol`](benchmarks/protocols/amazon_polarity_phase2_v1.json).

This is explicitly the **50,000-train / 10,000-test subset benchmark**, not a result on the full 3.6M-review training split.

The benchmark can be reproduced manually with:

```bash
python -m pip install -e ".[benchmark]"
python -m src.benchmarking --profile phase2
```

The smaller `smoke` profile is for integration checks only and must not be used for release performance claims.

## Reliable inference

`src/inference.py` defines an `InferenceBundle`: the fitted estimator, resolved label schema, random seed, feature settings, and immutable preprocessing settings travel together. Raw prediction text is transformed with the same settings used during training instead of later UI choices.

Confidence-aware training can wrap the complete TF-IDF + classifier pipeline in cross-validated calibration. Calibration is fit only inside the training split; the holdout remains untouched. Estimators without real probability estimates return no confidence instead of a synthetic decision-score probability.

Evaluation includes macro F1, balanced accuracy, accuracy, macro precision/recall, log loss when probabilities exist, expected calibration error, confidence-threshold coverage/selective accuracy, and row-level error analysis. See [`docs/INFERENCE.md`](docs/INFERENCE.md).

## Safe persistence and provenance

Preferred inference artifacts use `skops==0.14.0` and end in `.inference.skops`. The application inspects each preferred artifact with `skops.io.get_untrusted_types()`. Default-trusted types are accepted, and the only additional accepted names are a static reviewed allowlist of scikit-learn calibration internals produced by this project's calibrated inference path. Any other reported type is rejected; the application never auto-trusts the full set requested by an arbitrary file.

When a preferred bundle is saved, the Reliable Inference page also writes:

- a JSON manifest with artifact SHA-256, runtime versions, model/inference metadata, code revision when discoverable, evaluation metrics, and training-data fingerprint; and
- a generated model card covering intended use, limitations, provenance, evaluation, benchmark-evidence status, reproducibility, and security.

See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) for the model-card contract.

Legacy `.joblib` persistence remains available only for backwards compatibility with fully trusted local files. Joblib uses pickle semantics and can execute code while loading a malicious artifact. See [`SECURITY.md`](SECURITY.md).

## Setup

Python 3.11+ is required.

```bash
git clone https://github.com/AaryaMody1301/Sentiment-Analysis-for-Product-Reviews.git
cd Sentiment-Analysis-for-Product-Reviews
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .
```

Optional WordNet lemmatization:

```bash
python -m nltk.downloader wordnet
```

Run the app:

```bash
streamlit run main.py
```

## Development and release gates

```bash
python -m pip install -e ".[dev]"
python -m compileall -q src main.py pages tests
python -m pytest -q
python -m src.release --check candidate
python -m pip check
```

CI runs those checks on Python 3.11 and 3.13. The stricter release validation is:

```bash
python -m src.release --check release
```

Release mode requires the committed frozen benchmark evidence, project version `1.0.0`, release/security documentation, the pinned safe-artifact dependency, and a repository tree free of generated serialized models.

## Data and evidence

`datasets/sample_reviews.csv` is a demonstration dataset, not a benchmark. Large/raw datasets remain outside Git. CSV parsing is strict: malformed input raises an error instead of silently dropping records.

Performance claims require committed benchmark evidence with identified data, immutable revision, selection protocol, fingerprints, split-integrity checks, metrics, and runtime metadata.

## Roadmap

- **Phase 1 - complete:** correctness, deterministic behavior, packaging, tests, CI
- **Phase 2 - complete:** frozen external benchmark protocol and committed evidence
- **Phase 3 - complete:** inference contract, calibrated confidence, error analysis, Reliable Inference page
- **Phase 4 - complete:** safe persistence, provenance/model cards, release validation, generated-artifact cleanup
- **Phase 5 - complete:** integrated v1.0.0 evidence, final CI, and strict release validation

## License

MIT. See [`LICENSE`](LICENSE).

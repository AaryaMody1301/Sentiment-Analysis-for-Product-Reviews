# Sentiment Analysis for Product Reviews

A Streamlit-based NLP application for exploring, training, comparing, and using classical sentiment classifiers on product-review data with explicit correctness, evidence, inference, and artifact-security contracts.

## Current status

**Phase 4: release hardening (stacked on Phase 3).** Phase 1 established correctness and deterministic foundations. Phase 2 adds a frozen external benchmark and remains behind its benchmark-evidence gate. Phase 3 adds inference-safe preprocessing contracts, calibrated confidence, and error analysis. Phase 4 prepares the repository for a v1.0 release without bypassing those earlier gates.

Phase 4 adds safe-by-default `.skops` inference artifacts, per-artifact provenance manifests/model cards, release validation, package metadata, and cleanup of previously committed generated model binaries. The project version remains pre-1.0 until the Phase 2 evidence result is committed and the stacked phases are merged in order.

## Label contracts

Numeric labels are not interpreted by value alone:

| Schema | Meaning |
| --- | --- |
| `binary_01` | `0 = negative`, `1 = positive` |
| `stars_1_to_5` | `1-2 = negative`, `3 = neutral`, `4-5 = positive` |

`auto` inference accepts binary values only in clear label/sentiment-style columns and star values only in rating/star/score-style columns. Ambiguous numeric columns raise an error instead of risking incorrect ground truth. Unknown text labels are rejected rather than silently converted to neutral.

## Models

The application supports Logistic Regression, Multinomial Naive Bayes, Linear SVM, and Random Forest. TF-IDF is used for normal training and HashingVectorizer supports bounded-memory processing.

The Phase 2 external benchmark uses Dummy Classifier, Multinomial Naive Bayes, Logistic Regression, and Linear SVM. Random Forest is intentionally outside the frozen sparse-text benchmark because that benchmark prioritizes strong, resource-efficient classical baselines.

## Reproducible Phase 2 benchmark

The benchmark source is MTEB `amazon_polarity`, pinned to revision `ec149c1fe36043668a50804214d4597804001f6f`. The frozen Phase 2 profile uses:

- 50,000 training reviews: 25,000 per class;
- 10,000 test reviews: 5,000 per class;
- fixed seed `42` and pinned `datasets==5.0.1`;
- train-only TF-IDF with up to 50,000 unigram/bigram features;
- macro F1 as the primary metric;
- accuracy, macro precision/recall, balanced accuracy, per-class metrics, and a confusion matrix;
- selected-sample SHA-256 fingerprints plus duplicate and train/test-overlap checks.

This is the **Phase 2 subset benchmark**, not a result on the full 3.6M-review training split. See [benchmarks/README.md](benchmarks/README.md) and the frozen [protocol](benchmarks/protocols/amazon_polarity_phase2_v1.json).

```bash
python -m pip install -e ".[benchmark]"
python -m src.benchmarking --profile phase2
```

The `smoke` profile is for integration checks only and must not be used for release performance claims.

## Reliable inference

`src/inference.py` defines an `InferenceBundle`: the fitted estimator, resolved label schema, random seed, feature settings, and immutable preprocessing settings travel together. Raw prediction text is transformed with the same settings used during training instead of later UI choices.

Confidence-aware training can wrap the complete TF-IDF + classifier pipeline in cross-validated calibration. Calibration is fit only inside the training split; the holdout remains untouched. Estimators without real probability estimates return no confidence instead of a synthetic decision-score probability.

Evaluation includes macro F1, balanced accuracy, accuracy, macro precision/recall, log loss when probabilities exist, expected calibration error, confidence-threshold coverage/selective accuracy, and row-level error analysis. See [docs/INFERENCE.md](docs/INFERENCE.md).

## Safe persistence and provenance

Preferred inference artifacts use `skops==0.14.0` and end in `.inference.skops`. The application inspects each preferred artifact with `skops.io.get_untrusted_types()`. Default-trusted types are accepted, and the only additional accepted names are a static reviewed allowlist of scikit-learn calibration internals produced by this project's `CalibratedClassifierCV` path. Any other reported type is rejected; the application never auto-trusts the full set requested by an arbitrary file.

When a preferred bundle is saved, the Reliable Inference page also writes:

- a JSON manifest with artifact SHA-256, runtime versions, model/inference metadata, code revision when discoverable, evaluation metrics, and training-data fingerprint; and
- a generated model card covering intended use, limitations, provenance, evaluation, benchmark-evidence status, reproducibility, and security.

See [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for the model-card contract.

Legacy `.joblib` persistence remains available only for backwards compatibility with fully trusted local files. Joblib uses pickle semantics and can execute code while loading a malicious artifact. See [SECURITY.md](SECURITY.md).

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

CI runs these checks on Python 3.11 and 3.13. A separate manual **Release gate** runs the stricter `python -m src.release --check release`; that check intentionally fails until the committed Phase 2 benchmark result exists and the package version is explicitly set to `1.0.0`.

## Data and evidence

`datasets/sample_reviews.csv` is a demonstration dataset, not a benchmark. Large/raw datasets remain outside Git. CSV parsing is strict: malformed input raises an error instead of silently dropping records.

Performance claims require committed benchmark evidence with identified data, immutable revision, selection protocol, fingerprints, split-integrity checks, metrics, and runtime metadata.

## Roadmap

- **Phase 1 - complete:** correctness, deterministic behavior, packaging, tests, CI
- **Phase 2 - evidence gate active:** reproducible external benchmark and evidence-backed model comparison
- **Phase 3 - code complete / stacked:** inference contract, calibrated confidence, error analysis, product migration
- **Phase 4 - active / stacked:** safe persistence, provenance/model cards, release validation, generated-artifact cleanup, v1.0 readiness

## License

MIT. See [LICENSE](LICENSE).

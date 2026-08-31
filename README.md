# Sentiment Analysis for Product Reviews

A Streamlit-based NLP application for exploring, training, comparing, and using classical sentiment classifiers on product-review data.

## Current status

**Phase 3: inference and product reliability (stacked on Phase 2).** Phase 1 established the correctness foundation. Phase 2 adds the frozen external benchmark and remains behind its evidence gate until the benchmark artifact is committed. Phase 3 is being developed on top of that branch without bypassing the gate.

Phase 3 adds an inference bundle that captures the training-time preprocessing contract, supports cross-validated probability calibration, reports calibration-aware metrics, and produces structured error analysis. The legacy Streamlit workflow remains available while prediction paths are migrated incrementally.

## Label contracts

Numeric labels are not interpreted by value alone:

| Schema | Meaning |
| --- | --- |
| `binary_01` | `0 = negative`, `1 = positive` |
| `stars_1_to_5` | `1-2 = negative`, `3 = neutral`, `4-5 = positive` |

`auto` inference accepts binary values only in clear label/sentiment-style columns and star values only in rating/star/score-style columns. Ambiguous numeric columns raise an error instead of risking incorrect ground truth. Unknown text labels are rejected rather than silently converted to neutral.

## Models

The application supports Logistic Regression, Multinomial Naive Bayes, Linear SVM, and Random Forest. TF-IDF is used for normal training and HashingVectorizer supports bounded-memory processing.

The Phase 2 external benchmark uses a baseline ladder of Dummy Classifier, Multinomial Naive Bayes, Logistic Regression, and Linear SVM. Random Forest is not part of the frozen sparse-text benchmark because the benchmark prioritizes strong, resource-efficient classical baselines.

## Reproducible Phase 2 benchmark

The benchmark source is MTEB `amazon_polarity`, pinned to revision `ec149c1fe36043668a50804214d4597804001f6f`. The source dataset contains 3,599,994 training reviews and 400,000 test reviews with binary labels (`0 = negative`, `1 = positive`).

The frozen Phase 2 profile uses a reproducible, balanced subset:

- 50,000 training reviews: 25,000 per class
- 10,000 test reviews: 5,000 per class
- fixed seed `42`
- pinned `datasets==5.0.1`
- train-only TF-IDF fitting with up to 50,000 unigram/bigram features
- primary metric: macro F1
- supporting metrics: accuracy, macro precision/recall, balanced accuracy, per-class metrics, confusion matrix
- selected-sample SHA-256 fingerprints and train/test overlap checks recorded in every result

This is deliberately described as the **Phase 2 subset benchmark**, not as a result on the full 3.6M-review training split. See [benchmarks/README.md](benchmarks/README.md) and the frozen [protocol](benchmarks/protocols/amazon_polarity_phase2_v1.json).

Run it with:

```bash
python -m pip install -e ".[benchmark]"
python -m src.benchmarking --profile phase2
```

A smaller `smoke` profile is available for integration checks but must not be used for release performance claims.

## Inference reliability (Phase 3)

`src/inference.py` introduces an `InferenceBundle`: the fitted estimator, label schema, random seed, and immutable preprocessing settings travel together. Raw prediction text is therefore transformed with the same settings used during training instead of whatever UI checkboxes happen to be selected later.

For confidence-aware inference, Phase 3 can wrap the complete TF-IDF + classifier pipeline in cross-validated `CalibratedClassifierCV`. Calibration is fit only inside the training split; the holdout remains untouched. Uncalibrated estimators without `predict_proba` return no confidence rather than a synthetic score.

Evaluation now includes:

- macro F1, balanced accuracy, accuracy, macro precision and recall
- log loss when probabilities exist
- expected calibration error (ECE)
- confidence-threshold coverage and selective accuracy at 0.5, 0.7, 0.8 and 0.9
- row-level error analysis with true label, prediction, confidence and error type

See [docs/INFERENCE.md](docs/INFERENCE.md) for the contract and migration rules.

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

## Development

```bash
python -m pip install -e ".[dev]"
python -m compileall -q src main.py pages tests
python -m pytest -q
python -m pip check
```

CI runs these checks on Python 3.11 and 3.13.

## Data and evidence

`datasets/sample_reviews.csv` is a demonstration dataset, not a benchmark. Large/raw datasets remain outside Git. Performance claims require a committed benchmark result with identified data, immutable revision, selection protocol, fingerprints, split-integrity checks, metrics, and runtime metadata.

CSV parsing is strict: malformed input raises an error instead of silently dropping records.

## Model persistence and trust

Saved `.joblib` models are a compatibility feature. `joblib` uses pickle semantics, so loading an untrusted artifact can execute code. Only load models you created or fully trust. See [SECURITY.md](SECURITY.md).

## Roadmap

- **Phase 1 - complete:** correctness, deterministic behavior, packaging, tests, CI
- **Phase 2 - evidence gate active:** reproducible external benchmark and evidence-backed model comparison
- **Phase 3 - active (stacked):** inference contract, calibrated confidence, error analysis and product migration
- **Phase 4:** release hardening, provenance/model cards, safer persistence, v1.0.0

## License

MIT. See [LICENSE](LICENSE).

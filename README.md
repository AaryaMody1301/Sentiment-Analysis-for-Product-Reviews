# Sentiment Analysis for Product Reviews

An evidence-first Streamlit NLP application for training and using classical product-review sentiment models with explicit label, evaluation, inference, and artifact-security contracts.

## Current status

**v1.0.0 release-ready.** The repository has a frozen external benchmark, deterministic training/evaluation paths, safe inference persistence, app-level smoke tests, CI on Python 3.11 and 3.13, and a strict release gate that validates the committed benchmark evidence rather than only checking that the file exists.

The retired legacy Streamlit monolith and pickle/joblib model-management UI are not part of the v1 application path. `main.py` is a small release landing page and `pages/1_Reliable_Inference.py` is the supported training/inference experience.

## Label contracts

Numeric labels are not interpreted by value alone:

| Schema | Meaning |
| --- | --- |
| `binary_01` | `0 = negative`, `1 = positive` |
| `stars_1_to_5` | `1-2 = negative`, `3 = neutral`, `4-5 = positive` |

`auto` inference accepts binary values only in clear label/sentiment-style columns and star values only in rating/star/score-style columns. Ambiguous numeric columns raise an error. Explicit numeric schemas also require finite integer-valued inputs, so values such as `0.5` or `2.9` are rejected rather than truncated.

## Models

The application supports Logistic Regression, Multinomial Naive Bayes, Linear SVM, and Random Forest. TF-IDF is used for normal training. `src/chunked_processing.py` provides a deterministic HashingVectorizer + MultinomialNB path for bounded-memory CSV processing without serialized-model persistence.

The frozen external benchmark uses Dummy Classifier, Multinomial Naive Bayes, Logistic Regression, and Linear SVM. Random Forest is intentionally outside the sparse-text benchmark because that benchmark prioritizes strong, resource-efficient classical baselines.

## Evidence-backed Amazon Polarity benchmark

The benchmark source is MTEB `amazon_polarity`, pinned to revision `ec149c1fe36043668a50804214d4597804001f6f`. The frozen profile uses 50,000 balanced training reviews and 10,000 balanced test reviews with seed `42`, train-only TF-IDF, and up to 50,000 unigram/bigram features.

| Model | Accuracy | Macro F1 | Balanced accuracy |
| --- | ---: | ---: | ---: |
| Dummy, most frequent | 0.5000 | 0.333333 | 0.5000 |
| Multinomial Naive Bayes | 0.8878 | 0.887799 | 0.8878 |
| Logistic Regression | 0.9070 | 0.907000 | 0.9070 |
| **Linear SVM** | **0.9088** | **0.908800** | **0.9088** |

Linear SVM ranks first by macro F1, but only by `0.0018` over Logistic Regression. That is a narrow result on this frozen subset, not evidence of universal superiority.

The successful GitHub Actions evidence run was `33471236375` on September 1, 2026. Integrity checks found zero duplicate texts in either selected split and zero train/test text overlap. The selected sequence fingerprints are:

- train: `ec4fc2ad1b734b6d43221fda8a67e6be5162eeec4426921860ff8181e928e944`;
- test: `00c1205e35fb1e5862e3fe9ea769e1acded6c0089cddb955d37e22ebbe042550`.

The complete result is committed at [`benchmarks/results/amazon_polarity_phase2_v1.json`](benchmarks/results/amazon_polarity_phase2_v1.json). See [`benchmarks/README.md`](benchmarks/README.md) and the frozen [`protocol`](benchmarks/protocols/amazon_polarity_phase2_v1.json).

This is explicitly the **50,000-train / 10,000-test subset benchmark**, not a result on the full 3.6M-review training split.

Reproduce it manually with:

```bash
python -m pip install -e ".[benchmark]"
python -m src.benchmarking --profile phase2
```

The smaller `smoke` profile is for integration checks only and must not be used for release performance claims.

## Reliable inference

`src/inference.py` defines an `InferenceBundle`: the fitted estimator, resolved label schema, random seed, feature settings, and immutable preprocessing settings travel together. Raw prediction text is transformed with the same settings used during training.

Confidence-aware training can wrap the complete TF-IDF + classifier pipeline in cross-validated calibration. Calibration is fit only inside the training split; the holdout remains untouched. Estimators without real probability estimates return no confidence instead of a synthetic decision-score probability.

Evaluation includes macro F1, balanced accuracy, accuracy, macro precision/recall, log loss when probabilities exist, expected calibration error, confidence-threshold coverage/selective accuracy, and row-level error analysis. See [`docs/INFERENCE.md`](docs/INFERENCE.md).

## Safe persistence and provenance

Preferred inference artifacts use `skops==0.14.0` and end in `.inference.skops`. Before loading, the application inspects serialized types with `skops.io.get_untrusted_types()`. Default-trusted types are accepted, and the only additional accepted names are a static reviewed allowlist of scikit-learn calibration internals produced by this project's calibrated inference path. Any other reported type is rejected.

The v1 application does not expose pickle/joblib model loading. When a preferred bundle is saved, the Reliable Inference page also writes a JSON provenance manifest and generated model card. See [`SECURITY.md`](SECURITY.md) and [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

## Setup

Python 3.11+ is required.

```bash
git clone https://github.com/AaryaMody1301/Sentiment-Analysis-for-Product-Reviews.git
cd Sentiment-Analysis-for-Product-Reviews
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .
```

Optional WordNet lemmatization is explicit and never downloaded at import time:

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
python -m src.release --check release
python -m pip check
```

CI runs those checks on Python 3.11 and 3.13. Release mode validates the exact frozen benchmark identity, dataset revision, selection counts, fingerprints, duplicate/overlap integrity, metric snapshot and winner, in addition to version/docs/security/artifact hygiene.

## Data and evidence

`datasets/sample_reviews.csv` is demonstration data, not a benchmark. Large/raw datasets remain outside Git. CSV parsing is strict: malformed input raises an error instead of silently dropping records.

Performance claims require committed benchmark evidence with identified data, immutable revision, selection protocol, fingerprints, split-integrity checks, metrics, and runtime metadata.

## License

MIT. See [`LICENSE`](LICENSE).

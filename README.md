# Sentiment Analysis for Product Reviews

A Streamlit-based NLP application for exploring, training, comparing, and using classical sentiment classifiers on product-review data.

## Current status

**Phase 1: trustworthy foundation.** The earlier unsupported `92%+ accuracy` headline has been removed. Performance claims will return only after Phase 2 introduces a frozen, reproducible benchmark with identified data, splits, metrics, and artifacts.

Phase 1 establishes explicit label contracts, a single canonical preprocessing implementation, deterministic in-memory model training, macro F1 and balanced accuracy, strict CSV parsing, honest probability handling, modern packaging, CI, and an explicit model-artifact trust boundary.

## Label contracts

Numeric labels are not interpreted by value alone:

| Schema | Meaning |
| --- | --- |
| `binary_01` | `0 = negative`, `1 = positive` |
| `stars_1_to_5` | `1-2 = negative`, `3 = neutral`, `4-5 = positive` |

`auto` inference accepts binary values only in clear label/sentiment-style columns and star values only in rating/star/score-style columns. Ambiguous numeric columns raise an error instead of risking incorrect ground truth. Unknown text labels are rejected rather than silently converted to neutral.

## Models

The current classical model set is Logistic Regression, Multinomial Naive Bayes, Linear SVM, and Random Forest. TF-IDF is used for normal training and HashingVectorizer supports bounded-memory processing.

Phase 1 does **not** claim that any model is best. Phase 2 will establish the benchmark and baseline ladder.

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
python -m compileall -q src main.py tests
python -m pytest -q
python -m pip check
```

CI runs these checks on Python 3.11 and 3.13.

## Data and evidence

`datasets/sample_reviews.csv` is a demonstration dataset, not a benchmark. Large/raw datasets should remain outside Git and must have documented provenance before they are used for release claims. CSV parsing is strict: malformed input raises an error instead of silently dropping records.

## Model persistence and trust

Saved `.joblib` models are a compatibility feature. `joblib` uses pickle semantics, so loading an untrusted artifact can execute code. Only load models you created or fully trust. See [SECURITY.md](SECURITY.md).

## Roadmap

- **Phase 1** - correctness, deterministic behavior, packaging, tests, CI
- **Phase 2** - reproducible external benchmark and evidence-backed model comparison
- **Phase 3** - inference/product redesign, calibrated confidence, error analysis
- **Phase 4** - release hardening, provenance/model cards, safer persistence, v1.0.0

## License

MIT. See [LICENSE](LICENSE).

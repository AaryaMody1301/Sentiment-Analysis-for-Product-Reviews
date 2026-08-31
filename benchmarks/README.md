# Reproducible Benchmarks

This directory contains the evidence used for repository performance claims. Demo data under `datasets/` is never treated as benchmark evidence.

## Amazon Polarity Phase 2 v1

The Phase 2 benchmark uses the MTEB `amazon_polarity` dataset at the immutable revision `ec149c1fe36043668a50804214d4597804001f6f`.

The source dataset has an official training split of 3,599,994 reviews and a test split of 400,000 reviews. Labels are binary: `0 = negative` and `1 = positive`. The frozen Phase 2 profile intentionally uses a smaller, resource-conscious subset so the benchmark can be reproduced on ordinary development hardware:

- 25,000 negative + 25,000 positive training reviews
- 5,000 negative + 5,000 positive test reviews
- seed `42`
- streaming shuffle buffer `50,000`
- exact balanced quota after the pinned streaming shuffle
- TF-IDF fit on training text only
- maximum 50,000 unigram/bigram features

The benchmark records SHA-256 fingerprints of the selected train and test sequences, class counts, duplicate counts, cross-split overlap, runtime package versions, per-class metrics, confusion matrices, and the complete model comparison.

### Baseline ladder

1. `DummyClassifier(strategy="most_frequent")`
2. Multinomial Naive Bayes
3. Logistic Regression
4. Linear SVM

The primary ranking metric is **macro F1**. Accuracy, macro precision, macro recall, balanced accuracy, per-class precision/recall/F1, and confusion matrices are reported as supporting evidence.

Random Forest remains available in the application but is intentionally excluded from this sparse-text benchmark ladder because it is substantially less resource-efficient than the linear baselines for this representation.

## Running the benchmark

Install the optional benchmark dependency and run the frozen profile:

```bash
python -m pip install -e ".[benchmark]"
python -m src.benchmarking --profile phase2
```

For a quick integration check without producing release evidence:

```bash
python -m src.benchmarking --profile smoke
```

Results are written under `benchmarks/results/`. A result is evidence for this repository only when its dataset revision, selection fingerprints, profile, and runtime metadata are present. The Phase 2 subset result must not be described as a score on the full 3.6M-review training set.

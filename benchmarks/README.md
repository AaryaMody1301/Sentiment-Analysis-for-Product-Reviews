# Reproducible Benchmarks

This directory contains the evidence used for repository performance claims. Demo data under `datasets/` is never treated as benchmark evidence.

## Amazon Polarity Phase 2 v1

The frozen benchmark uses the MTEB `amazon_polarity` dataset at immutable revision `ec149c1fe36043668a50804214d4597804001f6f`.

The source dataset has an official training split of 3,599,994 reviews and a test split of 400,000 reviews. Labels are binary: `0 = negative` and `1 = positive`. The release benchmark intentionally uses a smaller, resource-conscious subset:

- 25,000 negative + 25,000 positive training reviews;
- 5,000 negative + 5,000 positive test reviews;
- seed `42`;
- streaming shuffle buffer `50,000`;
- exact balanced quota after the pinned streaming shuffle;
- TF-IDF fit on training text only;
- maximum 50,000 unigram/bigram features.

### Committed result

The successful evidence run was GitHub Actions run `33471236375` on September 1, 2026. The uploaded `amazon-polarity-benchmark` artifact had ID `9786614134` and archive digest `sha256:0ac71926503639969842a9775a21ec6bf2f8729bbaf7cc3e3635976323bf755e`.

| Model | Accuracy | Macro F1 | Balanced accuracy |
| --- | ---: | ---: | ---: |
| Dummy, most frequent | 0.5000 | 0.333333 | 0.5000 |
| Multinomial Naive Bayes | 0.8878 | 0.887799 | 0.8878 |
| Logistic Regression | 0.9070 | 0.907000 | 0.9070 |
| **Linear SVM** | **0.9088** | **0.908800** | **0.9088** |

Linear SVM ranks first by the frozen primary metric, macro F1. Its margin over Logistic Regression is only `0.0018`, so the result supports a narrow benchmark win rather than a claim of broad practical superiority.

Integrity checks for the selected subset all passed:

- training duplicate texts: `0`;
- test duplicate texts: `0`;
- train/test text overlap: `0`;
- training sequence fingerprint: `ec4fc2ad1b734b6d43221fda8a67e6be5162eeec4426921860ff8181e928e944`;
- test sequence fingerprint: `00c1205e35fb1e5862e3fe9ea769e1acded6c0089cddb955d37e22ebbe042550`.

The successful run used CPython `3.13.15`, `datasets 5.0.1`, NumPy `2.5.2`, and scikit-learn `1.9.0`. The complete evidence, including per-class metrics and confusion matrices, is committed at [`results/amazon_polarity_phase2_v1.json`](results/amazon_polarity_phase2_v1.json).

This is explicitly a **50,000-train / 10,000-test subset benchmark**. It must not be described as a score on the full 3.6M-review training split.

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

The GitHub benchmark workflow is manual after the release evidence was committed, so routine pull requests do not repeatedly download and execute the external benchmark.

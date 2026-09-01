# Inference contract

A trained classifier is not sufficient by itself: prediction also depends on label semantics and the exact preprocessing used before the vectorizer sees a review. The v1 application therefore treats those settings as part of the model contract.

## `InferenceBundle`

`src.inference.InferenceBundle` stores the fitted estimator/pipeline, immutable `PreprocessingConfig`, resolved label schema, model name, random seed, calibration method, and training/holdout metadata.

Call `bundle.predict(...)` or `bundle.predict_frame(...)` with raw review text. The bundle applies its own preprocessing contract before inference.

## Confidence semantics

Confidence is never manufactured from a decision score.

- With calibration enabled, the full TF-IDF + classifier pipeline is wrapped by `CalibratedClassifierCV` and calibrated using stratified cross-validation **inside the training split**.
- If calibration is disabled but the estimator natively implements `predict_proba`, results are labeled `native_probability`.
- If the estimator has no probability interface and calibration is disabled, confidence is unavailable and returned as `NaN`.

The untouched holdout is used only for evaluation.

## Reproducible preprocessing

Resource-independent preprocessing is the default. WordNet lemmatization is opt-in. If a bundle explicitly requires WordNet and the resource is missing, training or prediction fails clearly instead of silently changing the transform. No NLTK resources are downloaded at import time.

Explicit `binary_01` and `stars_1_to_5` schemas require finite integer-valued inputs. Fractional values are rejected rather than truncated.

## Evaluation and error analysis

`evaluate_inference_bundle` reports macro F1, balanced accuracy, accuracy, macro precision/recall, and probability-aware diagnostics when available: log loss, expected calibration error, mean confidence, and confidence-threshold coverage/selective accuracy.

`build_error_analysis` returns row-level records with raw review text, true sentiment, predicted sentiment, confidence, correctness, and error type. Confident errors are surfaced first.

## Persistence and security

Supported persistence is implemented in `src.safe_persistence` with `.inference.skops` artifacts. The loader accepts default-trusted types plus only the exact reviewed scikit-learn calibration/CV names documented in `SECURITY.md`. Any other reported type is rejected.

The v1 Streamlit application does not expose pickle/joblib model loading.

## Streamlit application

`main.py` is the release entrypoint and `pages/1_Reliable_Inference.py` is the supported training/inference page. The previous monolithic app path was retired so preprocessing, confidence, persistence, and provenance behavior cannot diverge between two user interfaces.

# Inference contract

Phase 3 separates **model evaluation** from **deployment inference**. A trained classifier is not sufficient by itself: prediction also depends on the label contract and the exact text preprocessing used before the vectorizer sees a review.

## `InferenceBundle`

`src.inference.InferenceBundle` stores:

- the fitted estimator/pipeline;
- immutable `PreprocessingConfig` settings;
- the resolved label schema;
- model name and random seed;
- calibration method, when enabled;
- training/holdout metadata and feature settings.

Call `bundle.predict(...)` or `bundle.predict_frame(...)` with raw review text. The bundle applies its own preprocessing contract before inference.

## Confidence semantics

Confidence is never manufactured from a decision score.

- With calibration enabled, the full TF-IDF + classifier pipeline is wrapped by `CalibratedClassifierCV` and calibrated using stratified cross-validation **inside the training split**. The untouched holdout is used only for evaluation.
- If calibration is disabled but the estimator natively implements `predict_proba`, results are labeled `native_probability`, not calibrated confidence.
- If the estimator has no probability interface and calibration is disabled, confidence is unavailable and returned as `NaN` in prediction tables.

The default calibration method is sigmoid. This is a conservative choice for classical text models and avoids isotonic calibration's tendency to overfit when calibration samples are limited.

## Reproducible preprocessing

The Phase 3 preprocessing contract defaults to resource-independent preprocessing. WordNet lemmatization is opt-in. If a bundle explicitly requires WordNet and the resource is missing, training or prediction fails clearly instead of silently changing the text transform.

## Evaluation and error analysis

`evaluate_inference_bundle` reports standard classification metrics plus probability-aware diagnostics when available:

- log loss;
- expected calibration error (ECE);
- mean predicted confidence;
- coverage and accuracy for confidence thresholds 0.5, 0.7, 0.8 and 0.9.

`build_error_analysis` returns row-level records with raw review text, true sentiment, predicted sentiment, confidence, correctness and error type. Confident errors are surfaced first so failure modes can be inspected before adding model complexity.

## Persistence and security

`save_inference_bundle` and `load_inference_bundle` use joblib for compatibility. Joblib/pickle artifacts can execute code while loading. Only load artifacts created by this project or another fully trusted source. Safer persistence is a Phase 4 release-hardening item.

## Streamlit migration

The existing app currently trains on a `processed_text` column and preprocesses new prediction text using the current UI controls. Phase 3 adds a separate **Reliable Inference** page that trains from raw text and stores the preprocessing contract inside the bundle. This avoids a risky rewrite of the legacy monolith while giving users a trustworthy path immediately.

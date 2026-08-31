from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import streamlit as st

from src.inference import (
    PreprocessingConfig,
    load_inference_bundle,
    save_inference_bundle,
    train_inference_bundle,
)
from src.model_training import get_available_models

st.set_page_config(page_title="Reliable Inference", page_icon="R", layout="wide")
st.title("Reliable Sentiment Inference")
st.caption(
    "Train and use a model with a fixed preprocessing contract and explicit confidence semantics."
)

source_tab, bundle_tab = st.tabs(["Train bundle", "Load trusted bundle"])

with source_tab:
    uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"], key="reliable_csv")
    if uploaded is not None:
        try:
            frame = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            frame = None
    else:
        frame = None

    if frame is not None and not frame.empty:
        st.dataframe(frame.head(10), use_container_width=True)
        columns = list(frame.columns)
        left, right = st.columns(2)
        with left:
            text_column = st.selectbox("Review text column", columns)
        with right:
            sentiment_candidates = [column for column in columns if column != text_column]
            sentiment_column = st.selectbox("Sentiment label column", sentiment_candidates)

        schema_labels = {
            "Auto (safe inference only)": "auto",
            "Text labels": "text",
            "Binary 0/1": "binary_01",
            "Ratings 1-5": "stars_1_to_5",
        }
        schema_name = st.selectbox("Label schema", list(schema_labels))

        st.subheader("Preprocessing contract")
        prep_a, prep_b = st.columns(2)
        with prep_a:
            remove_stopwords = st.checkbox("Remove stopwords", value=True)
            stemming = st.checkbox("Apply stemming", value=False)
        with prep_b:
            lemmatization = st.checkbox(
                "Apply WordNet lemmatization",
                value=False,
                help="Requires the WordNet resource in both training and inference environments.",
            )
            negations = st.checkbox("Preserve negations", value=True)

        if stemming and lemmatization:
            st.info("Lemmatization takes precedence over stemming in the canonical preprocessor.")

        st.subheader("Model and confidence")
        model_col, confidence_col = st.columns(2)
        with model_col:
            model_name = st.selectbox("Model", list(get_available_models()))
            max_features = st.select_slider(
                "Maximum TF-IDF features",
                options=[3000, 5000, 10000, 20000, 50000],
                value=10000,
            )
        with confidence_col:
            calibrate = st.checkbox(
                "Calibrate probability estimates",
                value=True,
                help="Uses cross-validation inside the training split. The holdout remains untouched.",
            )
            calibration_method = st.selectbox(
                "Calibration method",
                ["sigmoid", "temperature", "isotonic"],
                disabled=not calibrate,
            )
            calibration_cv = st.slider(
                "Calibration folds",
                min_value=2,
                max_value=5,
                value=3,
                disabled=not calibrate,
            )

        split_col, seed_col = st.columns(2)
        with split_col:
            test_size = st.slider("Holdout fraction", 0.1, 0.4, 0.2, 0.05)
        with seed_col:
            random_state = st.number_input("Random seed", min_value=0, value=42, step=1)

        if st.button("Train reliable inference bundle", type="primary"):
            try:
                config = PreprocessingConfig(
                    remove_stopwords=remove_stopwords,
                    perform_stemming=stemming,
                    perform_lemmatization=lemmatization,
                    handle_negations=negations,
                )
                bundle, metrics, errors = train_inference_bundle(
                    frame,
                    text_column,
                    sentiment_column,
                    model_name=model_name,
                    preprocessing=config,
                    label_schema=schema_labels[schema_name],
                    test_size=float(test_size),
                    random_state=int(random_state),
                    max_features=int(max_features),
                    calibrate=calibrate,
                    calibration_method=calibration_method,
                    calibration_cv=int(calibration_cv),
                )
            except Exception as exc:
                st.error(f"Training failed: {exc}")
            else:
                st.session_state["reliable_bundle"] = bundle
                st.session_state["reliable_metrics"] = metrics
                st.session_state["reliable_errors"] = errors
                st.success("Inference bundle trained with an untouched holdout evaluation.")

with bundle_tab:
    model_dir = Path("models")
    saved = sorted(model_dir.glob("*.inference.joblib")) if model_dir.is_dir() else []
    if not saved:
        st.info("No saved inference bundles were found in models/.")
    else:
        selected = st.selectbox("Trusted local bundle", saved, format_func=lambda path: path.name)
        st.warning(
            "Joblib uses pickle semantics. Load only an artifact created by this project or another fully trusted source."
        )
        if st.button("Load selected trusted bundle"):
            try:
                with warnings.catch_warnings(record=True):
                    bundle = load_inference_bundle(selected)
            except Exception as exc:
                st.error(f"Could not load bundle: {exc}")
            else:
                st.session_state["reliable_bundle"] = bundle
                st.session_state.pop("reliable_metrics", None)
                st.session_state.pop("reliable_errors", None)
                st.success(f"Loaded {selected.name}")

bundle = st.session_state.get("reliable_bundle")
metrics = st.session_state.get("reliable_metrics")
errors = st.session_state.get("reliable_errors")

if bundle is not None:
    st.divider()
    st.subheader("Active inference contract")
    contract = {
        "model": bundle.model_name,
        "label_schema": bundle.label_schema,
        "confidence_kind": bundle.confidence_kind,
        "random_state": bundle.random_state,
        **bundle.metadata.get("preprocessing", {}),
    }
    st.json(contract)

    if metrics is not None:
        st.subheader("Untouched holdout evaluation")
        metric_columns = st.columns(4)
        metric_columns[0].metric("Macro F1", f"{metrics['macro_f1']:.3f}")
        metric_columns[1].metric("Accuracy", f"{metrics['accuracy']:.3f}")
        metric_columns[2].metric("Balanced accuracy", f"{metrics['balanced_accuracy']:.3f}")
        ece = metrics.get("expected_calibration_error")
        metric_columns[3].metric("ECE", "N/A" if ece is None else f"{ece:.3f}")
        st.caption(f"Confidence semantics: {metrics['confidence_kind']}")

        if metrics.get("selective_accuracy"):
            selective = pd.DataFrame.from_dict(metrics["selective_accuracy"], orient="index")
            selective.index.name = "minimum_confidence"
            st.write("Selective accuracy by confidence threshold")
            st.dataframe(selective, use_container_width=True)

        if errors is not None:
            mistakes = errors.loc[~errors["correct"]]
            st.write(f"Holdout errors: {len(mistakes)} of {len(errors)}")
            if not mistakes.empty:
                st.dataframe(mistakes.head(50), use_container_width=True)

    st.subheader("Predict raw reviews")
    single_review = st.text_area("Review text", key="reliable_single_review")
    if st.button("Analyze review"):
        if not single_review.strip():
            st.error("Enter a review first.")
        else:
            st.dataframe(bundle.predict_frame(single_review), use_container_width=True)

    batch_reviews = st.text_area(
        "Batch reviews (one review per line)",
        key="reliable_batch_reviews",
    )
    if st.button("Analyze batch"):
        reviews = [line.strip() for line in batch_reviews.splitlines() if line.strip()]
        if not reviews:
            st.error("Enter at least one review.")
        else:
            st.dataframe(bundle.predict_frame(reviews), use_container_width=True)

    bundle_name = st.text_input("Bundle name", value="sentiment-inference")
    if st.button("Save trusted local bundle"):
        try:
            path = save_inference_bundle(bundle, bundle_name)
        except Exception as exc:
            st.error(f"Could not save bundle: {exc}")
        else:
            st.success(f"Saved to {path}")

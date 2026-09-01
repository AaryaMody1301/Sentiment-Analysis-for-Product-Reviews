from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Product Review Sentiment Analysis",
    layout="wide",
)

st.title("Product Review Sentiment Analysis")
st.caption(
    "Evidence-first classical NLP with explicit label contracts, reproducible evaluation, "
    "and safe inference artifacts."
)

st.success(
    "v1.0.0 release candidate: frozen Amazon Polarity evidence, Python 3.11/3.13 CI, "
    "and strict release validation are part of the repository."
)

left, right = st.columns(2)
with left:
    st.subheader("Reliable inference")
    st.write(
        "Train from raw review text, preserve preprocessing settings inside the inference "
        "contract, calibrate confidence without touching the holdout, and save inspected "
        "`.skops` artifacts with provenance sidecars."
    )
    st.page_link(
        "pages/1_Reliable_Inference.py",
        label="Open Reliable Inference"
    )

with right:
    st.subheader("Frozen benchmark")
    st.metric("Linear SVM macro F1", "0.9088")
    st.write(
        "Measured on the frozen 50,000-train / 10,000-test Amazon Polarity subset. "
        "The margin over Logistic Regression is 0.0018, so the result is intentionally "
        "described as a narrow benchmark win."
    )

st.divider()
st.subheader("Release guarantees")
st.markdown(
    """
- Numeric labels require explicit, validated semantics; fractional values are rejected.
- TF-IDF is fit on training data only and holdout data is never used for fitting or calibration.
- Preferred model artifacts use `skops` with a static reviewed-type allowlist.
- Benchmark claims require the committed frozen evidence, fingerprints, and integrity checks.
- Generated models, bytecode, logs, and local datasets are excluded from the source release.
"""
)

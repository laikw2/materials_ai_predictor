"""Streamlit app for polymer cross-presentation prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from common import MODELS_DIR, load_bundle


def score_or_proba(estimator, X):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    return estimator.predict(X)


@st.cache_resource
def load_models():
    reg_bundle = load_bundle(MODELS_DIR / "best_regression_model.joblib")
    clf_path = MODELS_DIR / "best_classifier_model.joblib"
    clf_bundle = load_bundle(clf_path) if clf_path.exists() else None
    return reg_bundle, clf_bundle


def nullable_number(label: str, default: float | None, help_text: str = ""):
    text_default = "" if default is None or np.isnan(default) else str(round(float(default), 6))
    value = st.text_input(label, value=text_default, help=help_text)
    if value.strip() == "":
        return np.nan
    try:
        return float(value)
    except ValueError:
        st.warning(f"{label} is not numeric, so it will be treated as missing.")
        return np.nan


def main() -> None:
    st.set_page_config(page_title="Polymer Cross-Presentation Predictor", layout="wide")
    st.title("Polymer Cross-Presentation Predictor")
    st.caption("Decision-support tool for prioritizing polymer designs. Wetlab validation is still required.")

    try:
        reg_bundle, clf_bundle = load_models()
    except FileNotFoundError:
        st.error("No trained model found. Run `python src/train_regression.py` first.")
        st.stop()

    reg = reg_bundle["pipeline"]
    meta = reg_bundle["metadata"]
    features = meta["features"]
    ranges = meta.get("train_descriptor_ranges", {})

    st.subheader("Descriptor Inputs")
    st.write("Blank fields are treated as missing values and imputed by the trained pipeline median. They are not converted to zero.")

    cols = st.columns(3)
    values = {}
    for i, feature in enumerate(features):
        rng = ranges.get(feature, {})
        default = None
        help_text = ""
        if rng:
            default = (rng["min"] + rng["max"]) / 2
            help_text = f"Training range: {rng['min']:.4g} to {rng['max']:.4g}"
        with cols[i % 3]:
            if feature.startswith("R_"):
                selected = st.selectbox(feature, ["missing", 0, 1], index=0)
                values[feature] = np.nan if selected == "missing" else float(selected)
            else:
                values[feature] = nullable_number(feature, default, help_text)

    X = pd.DataFrame([values], columns=features)

    warnings = []
    for feature, value in values.items():
        if np.isnan(value) or feature not in ranges:
            continue
        lo, hi = ranges[feature]["min"], ranges[feature]["max"]
        if value < lo or value > hi:
            warnings.append(f"`{feature}`={value:.4g} is outside training range {lo:.4g}-{hi:.4g}.")

    if st.button("Predict Cross-Presentation"):
        pred = float(reg.predict(X)[0])
        st.metric("Predicted cross_presentation_pct", f"{pred:.3f}")

        if clf_bundle is not None:
            clf = clf_bundle["pipeline"]
            clf_meta = clf_bundle["metadata"]
            score = float(score_or_proba(clf, X)[0])
            label = int(clf.predict(X)[0])
            st.metric("High-efficiency score", f"{score:.3f}")
            st.write(f"Median-split class: **{'High' if label == 1 else 'Low'}** using threshold {clf_meta['threshold']:.3f}.")

        if warnings:
            st.warning("Prediction is extrapolative for some descriptors:\n\n" + "\n".join(f"- {w}" for w in warnings))

        st.info(
            "Use this output to prioritize candidate synthesis, not to replace wetlab testing. "
            "The training dataset has only 32 materials, so uncertainty is high."
        )


if __name__ == "__main__":
    main()

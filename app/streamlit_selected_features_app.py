"""Compact Streamlit app that asks only for features used by the final model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from common import MODELS_DIR, get_selected_feature_names, load_bundle


DERIVED_FEATURES = {
    "pka_sq": ["pka"],
    "hydrophobic_activation": ["LogP", "pka", "endosomal_alignment"],
}

MISSING_INDICATOR_FEATURES = {
    "missing_carbon_length": "carbon_length",
    "missing_hydrophobic_density": "hydrophobic_density",
    "missing_dp_per_carbon": "dp_per_carbon",
    "missing_chain_dp_interaction": "chain_dp_interaction",
}

FEATURE_LABELS = {
    "pka": "pKa",
    "LogP": "LogP",
    "R_C10": "R group: C10",
    "R_C12": "R group: C12",
    "endosomal_alignment": "Endosomal alignment",
    "carbon_length": "Carbon length",
    "hydrophobic_density": "Hydrophobic density",
    "dp_per_carbon": "DP per carbon",
    "chain_dp_interaction": "Chain-DP interaction",
}

FEATURE_DESCRIPTIONS = {
    "pka": "Acid dissociation / buffering descriptor.",
    "LogP": "Hydrophobicity descriptor.",
    "endosomal_alignment": "Alignment with the endosomal activation window.",
    "R_C10": "Use 1 if the candidate uses C10 R-group, otherwise 0.",
    "R_C12": "Use 1 if the candidate uses C12 R-group, otherwise 0.",
    "carbon_length": "Numeric alkyl carbon chain length.",
    "hydrophobic_density": "Numeric normalized hydrophobicity per carbon descriptor.",
    "dp_per_carbon": "Numeric polymerization degree per carbon descriptor.",
    "chain_dp_interaction": "Numeric chain length multiplied by DP descriptor.",
}


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


def nullable_number(label: str, default: float | None, help_text: str = "") -> float:
    text_default = "" if default is None or np.isnan(default) else str(round(float(default), 6))
    value = st.text_input(label, value=text_default, help=help_text)
    if value.strip() == "":
        return np.nan
    try:
        return float(value)
    except ValueError:
        st.warning(f"{label} is not numeric, so it will be treated as missing.")
        return np.nan


def required_raw_inputs(selected_features: list[str]) -> list[str]:
    required: set[str] = set()
    for feature in selected_features:
        if feature in DERIVED_FEATURES:
            required.update(DERIVED_FEATURES[feature])
        elif feature in MISSING_INDICATOR_FEATURES:
            required.add(MISSING_INDICATOR_FEATURES[feature])
        else:
            required.add(feature)
    preferred_order = [
        "pka",
        "LogP",
        "endosomal_alignment",
        "R_C10",
        "R_C12",
        "carbon_length",
        "hydrophobic_density",
        "dp_per_carbon",
        "chain_dp_interaction",
    ]
    ordered = [feature for feature in preferred_order if feature in required]
    ordered.extend(sorted(str(feature) for feature in required if feature not in ordered and not str(feature).startswith("missing_")))
    return ordered


def fill_full_schema(values: dict[str, float], all_features: list[str]) -> pd.DataFrame:
    row = {feature: np.nan for feature in all_features}
    row.update(values)

    if not np.isnan(row.get("pka", np.nan)):
        row["pka_sq"] = row["pka"] ** 2

    if all(not np.isnan(row.get(col, np.nan)) for col in ["LogP", "pka", "endosomal_alignment"]):
        row["hydrophobic_activation"] = row["LogP"] * (row["pka"] - 6.6) * row["endosomal_alignment"]

    return pd.DataFrame([row], columns=all_features)


def main() -> None:
    st.set_page_config(page_title="Compact Materials AI Predictor", layout="wide")
    st.title("Compact Materials AI Predictor")
    st.caption("This version asks only for inputs needed by the final feature-selected regression model.")

    try:
        reg_bundle, clf_bundle = load_models()
    except FileNotFoundError:
        st.error("No trained model found. Ensure `models/best_regression_model.joblib` is committed.")
        st.stop()

    reg = reg_bundle["pipeline"]
    meta = reg_bundle["metadata"]
    all_features = meta["features"]
    selected_features = get_selected_feature_names(reg, all_features)
    raw_inputs = required_raw_inputs(selected_features)
    ranges = meta.get("train_descriptor_ranges", {})

    st.subheader("Final Model")
    st.write(f"Regression model: **{meta.get('best_model', 'unknown')}**")
    with st.expander("Show internal selected model features"):
        st.write(
            "`missing_*` entries are automatic missing-value indicators created by the trained imputer. "
            "Users do not key these in as 0/1; they are generated automatically when the related numeric field is blank."
        )
        st.code("\n".join(selected_features))

    st.subheader("Required Inputs")
    st.write(
        "Blank fields are treated as missing and handled by the trained model's median imputer. "
        "They are not converted to zero."
    )

    cols = st.columns(2)
    values: dict[str, float] = {}
    for i, feature in enumerate(raw_inputs):
        label = FEATURE_LABELS.get(feature, feature)
        rng = ranges.get(feature, {})
        help_text = ""
        default = None
        if rng:
            default = (rng["min"] + rng["max"]) / 2
            help_text = f"Training range: {rng['min']:.4g} to {rng['max']:.4g}"

        with cols[i % 2]:
            if feature.startswith("R_"):
                selected = st.selectbox(label, ["missing", 0, 1], index=0, help=FEATURE_DESCRIPTIONS.get(feature, ""))
                values[feature] = np.nan if selected == "missing" else float(selected)
            else:
                combined_help = FEATURE_DESCRIPTIONS.get(feature, "")
                if help_text:
                    combined_help = f"{combined_help} {help_text}".strip()
                values[feature] = nullable_number(label, default, combined_help)

    X = fill_full_schema(values, all_features)

    warnings = []
    for feature, value in values.items():
        if np.isnan(value) or feature not in ranges:
            continue
        lo, hi = ranges[feature]["min"], ranges[feature]["max"]
        if value < lo or value > hi:
            warnings.append(f"`{FEATURE_LABELS.get(feature, feature)}`={value:.4g} is outside training range {lo:.4g}-{hi:.4g}.")

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
            "Use this output for candidate prioritization only. The model was trained on a small materials dataset, "
            "so new wetlab validation is still required."
        )


if __name__ == "__main__":
    main()

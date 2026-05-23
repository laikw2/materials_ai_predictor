"""Prediction helper for saved polymer models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import MODELS_DIR, load_bundle, load_dataset


def score_or_proba(estimator, X):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    return estimator.predict(X)


def load_example_row() -> pd.DataFrame:
    from common import DATA_PATH, get_feature_matrix

    df = load_dataset(DATA_PATH)
    return get_feature_matrix(df).iloc[[0]].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default=None, help="CSV containing descriptor columns.")
    parser.add_argument("--example", action="store_true", help="Predict the first dataset row as a smoke test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    regression_bundle = load_bundle(MODELS_DIR / "best_regression_model.joblib")
    reg = regression_bundle["pipeline"]
    metadata = regression_bundle["metadata"]
    features = metadata["features"]

    if args.example:
        X = load_example_row()
    elif args.input_csv:
        X = pd.read_csv(args.input_csv)
    else:
        raise SystemExit("Pass --example or --input-csv.")

    for col in features:
        if col not in X.columns:
            X[col] = np.nan
    X = X[features]
    pred = reg.predict(X)
    output = pd.DataFrame({"predicted_cross_presentation_pct": pred})

    classifier_path = MODELS_DIR / "best_classifier_model.joblib"
    if Path(classifier_path).exists():
        clf_bundle = load_bundle(classifier_path)
        clf = clf_bundle["pipeline"]
        score = score_or_proba(clf, X)
        label = clf.predict(X)
        output["high_efficiency_score"] = score
        output["predicted_high_efficiency_class"] = label

    print(output.to_string(index=False))


if __name__ == "__main__":
    main()

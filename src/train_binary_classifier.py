"""Train secondary median-split classifiers for ROC/AUC style metrics."""

from __future__ import annotations

import argparse
import warnings
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from common import (
    DATA_PATH,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    build_pipeline,
    classifier_model_spaces,
    combine_grids,
    ensure_dirs,
    get_feature_matrix,
    get_selected_feature_names,
    load_dataset,
    make_selector_grid,
    plot_bar,
    plot_learning_curve,
    precision_recall_curve,
    roc_curve,
    safe_ap,
    safe_auc,
    save_bundle,
    write_json,
)
from feature_config import DEFAULT_HIGH_EFFICIENCY_THRESHOLD, TARGET_REGRESSION

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--n-iter", type=int, default=8, help="Randomized inner-CV configurations per outer fold.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers for inner CV search. -1 uses all cores.")
    return parser.parse_args()


def score_or_proba(estimator, X):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    return estimator.predict(X)


def plot_confusion(cm: np.ndarray, path) -> None:
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Best Classifier Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.xticks([0, 1], ["Low", "High"])
    plt.yticks([0, 1], ["Low", "High"])
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_roc_pr(y_true: np.ndarray, scores: np.ndarray, prefix: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, scores)
    precision, recall, _ = precision_recall_curve(y_true, scores)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC={safe_auc(y_true, scores):.3f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Best Classifier ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{prefix}_roc.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR-AUC={safe_ap(y_true, scores):.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Best Classifier Precision-Recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{prefix}_pr_auc.png", dpi=180)
    plt.close()


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    ensure_dirs()
    df = load_dataset(args.data)
    X = get_feature_matrix(df)
    y_cont = pd.to_numeric(df[TARGET_REGRESSION], errors="coerce")
    keep = y_cont.notna()
    X = X.loc[keep].reset_index(drop=True)
    y_cont = y_cont.loc[keep].reset_index(drop=True)
    threshold = float(args.threshold if args.threshold is not None else y_cont.median())
    y = (y_cont >= threshold).astype(int)

    selector_grid = make_selector_grid("classification", X.shape[1])
    outer_cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    rows = []
    oof_scores = {}
    oof_labels = {}
    best_estimators = {}
    feature_counts = {}

    for model_name, (model, model_grid) in classifier_model_spaces().items():
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Training classifier: {model_name}", flush=True)
        pipe = build_pipeline(model)
        grid = combine_grids(selector_grid, model_grid)
        scores = np.full(len(y), np.nan, dtype=float)
        labels = np.full(len(y), -1, dtype=int)
        selected = Counter()

        for fold_id, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), start=1):
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] {model_name} outer fold {fold_id}/{args.folds}", flush=True)
            search = RandomizedSearchCV(
                estimator=clone(pipe),
                param_distributions=grid,
                n_iter=args.n_iter,
                cv=inner_cv,
                scoring="roc_auc",
                n_jobs=args.n_jobs,
                random_state=42 + fold_id,
                error_score=np.nan,
            )
            search.fit(X.iloc[train_idx], y.iloc[train_idx])
            scores[val_idx] = score_or_proba(search.best_estimator_, X.iloc[val_idx])
            labels[val_idx] = search.predict(X.iloc[val_idx])
            selected.update(get_selected_feature_names(search.best_estimator_, list(X.columns)))

        print(f"  [{datetime.now().strftime('%H:%M:%S')}] {model_name} final full-data refit", flush=True)
        final_search = RandomizedSearchCV(
            estimator=clone(pipe),
            param_distributions=grid,
            n_iter=args.n_iter,
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=args.n_jobs,
            random_state=42,
            error_score=np.nan,
        )
        final_search.fit(X, y)

        mask = ~np.isnan(scores)
        row = {
            "model": model_name,
            "threshold": threshold,
            "roc_auc": safe_auc(y.to_numpy()[mask], scores[mask]),
            "pr_auc": safe_ap(y.to_numpy()[mask], scores[mask]),
            "accuracy": float(accuracy_score(y.to_numpy()[mask], labels[mask])),
            "precision": float(precision_score(y.to_numpy()[mask], labels[mask], zero_division=0)),
            "recall": float(recall_score(y.to_numpy()[mask], labels[mask], zero_division=0)),
            "f1": float(f1_score(y.to_numpy()[mask], labels[mask], zero_division=0)),
            "best_params_full_data": str(final_search.best_params_),
        }
        rows.append(row)
        oof_scores[model_name] = scores
        oof_labels[model_name] = labels
        best_estimators[model_name] = final_search.best_estimator_
        feature_counts[model_name] = selected

    summary_df = pd.DataFrame(rows).sort_values(["roc_auc", "pr_auc", "f1"], ascending=[False, False, False])
    summary_df.to_csv(METRICS_DIR / "classifier_model_comparison.csv", index=False)
    plot_bar(summary_df.sort_values("roc_auc", ascending=False), "model", "roc_auc", "Classifier ROC-AUC Comparison", FIGURES_DIR / "classifier_roc_auc_comparison.png")

    best_name = str(summary_df.iloc[0]["model"])
    best_scores = oof_scores[best_name]
    best_labels = oof_labels[best_name]
    cm = confusion_matrix(y, best_labels)
    plot_confusion(cm, FIGURES_DIR / "best_classifier_confusion_matrix.png")
    plot_roc_pr(y.to_numpy(), best_scores, "best_classifier")

    selected_df = pd.DataFrame(feature_counts[best_name].most_common(), columns=["feature", "selection_count"])
    selected_df.to_csv(METRICS_DIR / "best_classifier_selected_feature_frequency.csv", index=False)
    plot_learning_curve(
        best_estimators[best_name],
        X,
        y,
        f"Classifier Learning Curve: {best_name}",
        FIGURES_DIR / "best_classifier_training_validation_curve.png",
        scoring="roc_auc",
    )

    metadata = {
        "task": "median_split_classification",
        "target_source": TARGET_REGRESSION,
        "threshold": threshold,
        "positive_class": f"{TARGET_REGRESSION} >= {threshold:.3f}",
        "best_model": best_name,
        "features": list(X.columns),
        "metrics": summary_df.iloc[0].to_dict(),
        "note": "Classifier is secondary and exists only to support ROC/AUC/confusion-matrix style evaluation.",
    }
    save_bundle(MODELS_DIR / "best_classifier_model.joblib", best_estimators[best_name], metadata)
    write_json(METRICS_DIR / "best_classifier_metadata.json", metadata)

    print("\nBest classifier:", best_name)
    print(summary_df.to_string(index=False))
    print("Saved:", MODELS_DIR / "best_classifier_model.joblib")


if __name__ == "__main__":
    main()

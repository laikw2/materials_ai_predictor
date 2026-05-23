"""Train leakage-free regression models for cross_presentation_pct."""

from __future__ import annotations

import argparse
import warnings
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, RepeatedKFold

from common import (
    DATA_PATH,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    build_pipeline,
    combine_grids,
    ensure_dirs,
    get_feature_matrix,
    get_selected_feature_names,
    load_dataset,
    plot_bar,
    plot_gbr_staged_loss,
    plot_learning_curve,
    plot_predicted_vs_actual,
    plot_residuals,
    regression_model_spaces,
    rmse,
    save_bundle,
    spearman_corr,
    write_json,
    make_selector_grid,
)
from feature_config import TARGET_REGRESSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--n-iter", type=int, default=8, help="Randomized inner-CV configurations per outer fold.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers for inner CV search. -1 uses all cores.")
    parser.add_argument("--quick", action="store_true", help="Use fewer repeats for fast smoke testing.")
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()
    ensure_dirs()
    if args.quick:
        args.repeats = 2

    df = load_dataset(args.data)
    X = get_feature_matrix(df)
    y = pd.to_numeric(df[TARGET_REGRESSION], errors="coerce")
    keep = y.notna()
    X = X.loc[keep].reset_index(drop=True)
    y = y.loc[keep].reset_index(drop=True)

    selector_grid = make_selector_grid("regression", X.shape[1])
    outer_cv = RepeatedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=42)
    inner_cv = KFold(n_splits=args.inner_folds, shuffle=True, random_state=42)

    all_rows = []
    oof_predictions: dict[str, np.ndarray] = {}
    feature_counts: dict[str, Counter] = {}
    best_estimators = {}

    for model_name, (model, model_grid) in regression_model_spaces().items():
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Training regression model: {model_name}", flush=True)
        pipe = build_pipeline(model)
        grid = combine_grids(selector_grid, model_grid)
        preds = np.full(len(y), np.nan, dtype=float)
        fold_rows = []
        selected = Counter()

        for fold_id, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), start=1):
            total_folds = args.folds * args.repeats
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] {model_name} outer fold {fold_id}/{total_folds}", flush=True)
            search = RandomizedSearchCV(
                estimator=clone(pipe),
                param_distributions=grid,
                n_iter=args.n_iter,
                cv=inner_cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=args.n_jobs,
                random_state=42 + fold_id,
                error_score=np.nan,
            )
            search.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = search.predict(X.iloc[val_idx])
            preds[val_idx] = pred
            selected.update(get_selected_feature_names(search.best_estimator_, list(X.columns)))
            fold_rows.append(
                {
                    "model": model_name,
                    "fold": fold_id,
                    "mae": float(np.mean(np.abs(y.iloc[val_idx].to_numpy() - pred))),
                    "rmse": rmse(y.iloc[val_idx].to_numpy(), pred),
                    "r2": float("nan") if len(val_idx) < 2 else float(r2_score(y.iloc[val_idx], pred)),
                    "best_params": str(search.best_params_),
                }
            )

        # Fit final tuned model on all data with the same inner-CV procedure.
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] {model_name} final full-data refit", flush=True)
        final_search = RandomizedSearchCV(
            estimator=clone(pipe),
            param_distributions=grid,
            n_iter=args.n_iter,
            cv=inner_cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=args.n_jobs,
            random_state=42,
            error_score=np.nan,
        )
        final_search.fit(X, y)
        best_estimators[model_name] = final_search.best_estimator_
        oof_predictions[model_name] = preds
        feature_counts[model_name] = selected

        mask = ~np.isnan(preds)
        summary = {
            "model": model_name,
            "mae": float(np.mean(np.abs(y.to_numpy()[mask] - preds[mask]))),
            "rmse": rmse(y.to_numpy()[mask], preds[mask]),
            "r2": float(r2_score(y.to_numpy()[mask], preds[mask])),
            "spearman": spearman_corr(y.to_numpy()[mask], preds[mask]),
            "best_params_full_data": str(final_search.best_params_),
        }
        all_rows.extend(fold_rows)
        all_rows.append({**summary, "fold": "OOF_SUMMARY", "best_params": summary["best_params_full_data"]})

    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(METRICS_DIR / "regression_cv_metrics.csv", index=False)
    summary_df = metrics_df[metrics_df["fold"].eq("OOF_SUMMARY")].copy()
    summary_df = summary_df.sort_values(["rmse", "mae"], ascending=[True, True])
    summary_df.to_csv(METRICS_DIR / "regression_model_comparison.csv", index=False)

    best_name = str(summary_df.iloc[0]["model"])
    best_pipeline = best_estimators[best_name]
    best_pred = oof_predictions[best_name]
    best_selected = feature_counts[best_name]

    plot_bar(summary_df, "model", "rmse", "Regression Model RMSE Comparison", FIGURES_DIR / "regression_rmse_comparison.png")
    plot_predicted_vs_actual(y.to_numpy(), best_pred, f"Best Regression OOF: {best_name}", FIGURES_DIR / "best_regression_predicted_vs_actual.png")
    plot_residuals(y.to_numpy(), best_pred, f"Best Regression Residuals: {best_name}", FIGURES_DIR / "best_regression_residuals.png")

    staged_ok = plot_gbr_staged_loss(best_pipeline, X, y, f"Training vs Validation Loss: {best_name}", FIGURES_DIR / "best_regression_training_validation_loss.png")
    if not staged_ok:
        plot_learning_curve(
            best_pipeline,
            X,
            y,
            f"Learning Curve: {best_name}",
            FIGURES_DIR / "best_regression_training_validation_loss.png",
            scoring="neg_root_mean_squared_error",
        )

    selected_df = pd.DataFrame(best_selected.most_common(), columns=["feature", "selection_count"])
    selected_df.to_csv(METRICS_DIR / "best_regression_selected_feature_frequency.csv", index=False)

    train_rmse = rmse(y.to_numpy(), best_pipeline.predict(X))
    cv_rmse = float(summary_df.iloc[0]["rmse"])
    diagnosis = "reasonable_fit"
    if train_rmse > cv_rmse * 1.15:
        diagnosis = "possible_underfitting"
    elif cv_rmse > train_rmse * 1.8:
        diagnosis = "possible_overfitting"

    metadata = {
        "task": "regression",
        "target": TARGET_REGRESSION,
        "best_model": best_name,
        "features": list(X.columns),
        "train_descriptor_ranges": {col: {"min": float(np.nanmin(X[col])), "max": float(np.nanmax(X[col]))} for col in X.columns},
        "oof_metrics": summary_df.iloc[0].to_dict(),
        "train_rmse_full_fit": train_rmse,
        "fit_diagnosis": diagnosis,
        "note": "Use for design prioritization only. Wetlab validation remains required, especially outside training ranges.",
    }
    save_bundle(MODELS_DIR / "best_regression_model.joblib", best_pipeline, metadata)
    write_json(METRICS_DIR / "best_regression_metadata.json", metadata)

    print("\nBest regression model:", best_name)
    print(summary_df.head(10).to_string(index=False))
    print("Saved:", MODELS_DIR / "best_regression_model.joblib")


if __name__ == "__main__":
    main()

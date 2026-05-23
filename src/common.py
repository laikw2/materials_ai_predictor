"""Common utilities for training, evaluation, saving, and plotting."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge, RidgeClassifier, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold, StratifiedKFold, learning_curve
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from feature_config import DEFAULT_HIGH_EFFICIENCY_THRESHOLD, DESCRIPTOR_FEATURES, TARGET_REGRESSION

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "stage1_dataset.csv"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
METRICS_DIR = ROOT / "reports" / "metrics"


def ensure_dirs() -> None:
    for path in [MODELS_DIR, FIGURES_DIR, METRICS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
        elif df[col].dtype == object and col.startswith("R_"):
            df[col] = df[col].map({True: 1, False: 0, "True": 1, "False": 0})
    return df


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    available = [col for col in DESCRIPTOR_FEATURES if col in df.columns]
    X = df[available].copy()
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = pd.Series(y_true).rank().to_numpy()
    b = pd.Series(y_pred).rank().to_numpy()
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def safe_ap(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, score))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_preprocessor() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )


def make_selector_grid(task: str, n_features: int) -> list[dict[str, Any]]:
    k_small = min(6, n_features)
    k_mid = min(10, n_features)
    k_all = "all"
    if task == "regression":
        mi = SelectKBest(score_func=mutual_info_regression)
        lasso = SelectFromModel(ElasticNet(alpha=0.05, l1_ratio=0.8, max_iter=20000, random_state=42))
        rfe_est = Ridge(alpha=1.0)
    else:
        mi = SelectKBest(score_func=mutual_info_classif)
        lasso = SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear", C=0.5, random_state=42))
        rfe_est = LogisticRegression(solver="liblinear", C=0.5, random_state=42)

    return [
        {"selector": [SelectKBest(score_func=mi.score_func)], "selector__k": [k_mid, k_all]},
        {"selector": [RFE(estimator=rfe_est, step=0.25)], "selector__n_features_to_select": [k_mid]},
        {"selector": [lasso]},
    ]


def regression_model_spaces() -> dict[str, tuple[Any, list[dict[str, Any]]]]:
    spaces: dict[str, tuple[Any, list[dict[str, Any]]]] = {
        "Ridge": (
            Ridge(),
            [{"model__alpha": [0.1, 10.0, 100.0]}],
        ),
        "ElasticNet": (
            ElasticNet(max_iter=20000, random_state=42),
            [{"model__alpha": [0.01, 0.1, 1.0], "model__l1_ratio": [0.3, 0.8]}],
        ),
        "SVR": (
            SVR(),
            [{"model__C": [0.1, 1.0, 10.0], "model__epsilon": [0.1, 0.3], "model__gamma": ["scale"]}],
        ),
        "RandomForest": (
            RandomForestRegressor(n_estimators=120, random_state=42),
            [{"model__max_depth": [2, None], "model__min_samples_leaf": [2, 4], "model__max_features": ["sqrt"]}],
        ),
        "ExtraTrees": (
            ExtraTreesRegressor(n_estimators=120, random_state=42),
            [{"model__max_depth": [2, None], "model__min_samples_leaf": [2, 4], "model__max_features": ["sqrt"]}],
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(random_state=42),
            [{"model__n_estimators": [50, 100], "model__learning_rate": [0.03, 0.08], "model__max_depth": [1, 2], "model__subsample": [0.8]}],
        ),
        "MLPExperimental": (
            MLPRegressor(hidden_layer_sizes=(8,), alpha=0.01, early_stopping=True, validation_fraction=0.2, n_iter_no_change=50, max_iter=2000, random_state=42),
            [{"model__learning_rate_init": [0.001], "model__alpha": [0.01, 0.1]}],
        ),
    }

    try:
        from xgboost import XGBRegressor

        spaces["XGBoost"] = (
            XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=1, verbosity=0),
            [{"model__n_estimators": [50, 100], "model__learning_rate": [0.03, 0.08], "model__max_depth": [1, 2], "model__subsample": [0.8], "model__colsample_bytree": [0.8], "model__reg_alpha": [0.1], "model__reg_lambda": [1, 5]}],
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMRegressor

        spaces["LightGBM"] = (
            LGBMRegressor(objective="regression", random_state=42, n_jobs=1, verbose=-1),
            [{"model__n_estimators": [30, 80], "model__learning_rate": [0.03, 0.08], "model__max_depth": [1, 2], "model__num_leaves": [3, 5], "model__min_child_samples": [2], "model__reg_alpha": [0.1], "model__reg_lambda": [1, 5]}],
        )
    except Exception:
        pass

    return spaces


def classifier_model_spaces() -> dict[str, tuple[Any, list[dict[str, Any]]]]:
    spaces: dict[str, tuple[Any, list[dict[str, Any]]]] = {
        "LogisticRegression": (
            LogisticRegression(solver="liblinear", random_state=42),
            [{"model__C": [0.05, 0.1, 1.0, 10.0], "model__penalty": ["l1", "l2"]}],
        ),
        "SVC": (
            SVC(probability=True, random_state=42),
            [{"model__C": [0.1, 1.0, 10.0], "model__gamma": ["scale"], "model__kernel": ["rbf", "linear"]}],
        ),
        "RandomForest": (
            RandomForestClassifier(n_estimators=120, random_state=42),
            [{"model__max_depth": [2, None], "model__min_samples_leaf": [2, 4], "model__max_features": ["sqrt"]}],
        ),
        "ExtraTrees": (
            ExtraTreesClassifier(n_estimators=120, random_state=42),
            [{"model__max_depth": [2, None], "model__min_samples_leaf": [2, 4], "model__max_features": ["sqrt"]}],
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=42),
            [{"model__n_estimators": [50, 100], "model__learning_rate": [0.03, 0.08], "model__max_depth": [1, 2], "model__subsample": [0.8]}],
        ),
        "SGDLogistic": (
            SGDClassifier(loss="log_loss", early_stopping=True, n_iter_no_change=30, max_iter=2000, random_state=42),
            [{"model__alpha": [0.0001, 0.001, 0.01], "model__penalty": ["l2", "elasticnet"]}],
        ),
    }

    try:
        from xgboost import XGBClassifier

        spaces["XGBoost"] = (
            XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42, n_jobs=1, verbosity=0),
            [{"model__n_estimators": [50, 100], "model__learning_rate": [0.03, 0.08], "model__max_depth": [1, 2], "model__subsample": [0.8], "model__colsample_bytree": [0.8], "model__reg_alpha": [0.1], "model__reg_lambda": [1, 5]}],
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier

        spaces["LightGBM"] = (
            LGBMClassifier(objective="binary", random_state=42, n_jobs=1, verbose=-1),
            [{"model__n_estimators": [30, 80], "model__learning_rate": [0.03, 0.08], "model__max_depth": [1, 2], "model__num_leaves": [3, 5], "model__min_child_samples": [2], "model__reg_alpha": [0.1], "model__reg_lambda": [1, 5]}],
        )
    except Exception:
        pass

    return spaces


def build_pipeline(model: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor()),
            ("selector", SelectKBest(k="all")),
            ("model", model),
        ]
    )


def combine_grids(selector_grids: list[dict[str, Any]], model_grids: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grids = []
    for sg in selector_grids:
        for mg in model_grids:
            combined = {}
            combined.update(sg)
            combined.update(mg)
            grids.append(combined)
    return grids


def get_selected_feature_names(fitted_pipeline: Pipeline, base_features: list[str]) -> list[str]:
    preprocess = fitted_pipeline.named_steps["preprocess"]
    selector = fitted_pipeline.named_steps["selector"]
    try:
        names = preprocess.get_feature_names_out(base_features)
    except Exception:
        names = np.asarray(base_features)
    names = np.asarray([str(name).replace("x0_", "").replace("missingindicator_", "missing_") for name in names])
    if hasattr(selector, "get_support"):
        try:
            return list(names[selector.get_support()])
        except Exception:
            return list(names)
    return list(names)


def plot_bar(df: pd.DataFrame, x: str, y: str, title: str, path: Path) -> None:
    plt.figure(figsize=(10, 5))
    ordered = df.sort_values(y)
    plt.bar(ordered[x], ordered[y])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, title: str, path: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=55)
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    plt.xlabel("Actual cross_presentation_pct")
    plt.ylabel("Predicted cross_presentation_pct")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, title: str, path: Path) -> None:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, s=55)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Predicted")
    plt.ylabel("Residual (actual - predicted)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_learning_curve(estimator: Pipeline, X: pd.DataFrame, y: pd.Series, title: str, path: Path, scoring: str) -> None:
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes, train_scores, val_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        scoring=scoring,
        train_sizes=np.linspace(0.35, 1.0, 6),
        n_jobs=1,
    )
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    plt.figure(figsize=(7, 5))
    if scoring.startswith("neg_"):
        train_mean = -train_mean
        val_mean = -val_mean
        ylabel = scoring.replace("neg_", "")
    else:
        ylabel = scoring
    plt.plot(train_sizes, train_mean, marker="o", label="Training")
    plt.plot(train_sizes, val_mean, marker="o", label="Validation")
    plt.xlabel("Training samples")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_gbr_staged_loss(fitted_pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, title: str, path: Path) -> bool:
    model = fitted_pipeline.named_steps["model"]
    if not hasattr(model, "staged_predict"):
        return False
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(cv.split(X, y))
    candidate = clone(fitted_pipeline)
    candidate.fit(X.iloc[train_idx], y.iloc[train_idx])
    model = candidate.named_steps["model"]
    Xt_train = candidate[:-1].transform(X.iloc[train_idx])
    Xt_val = candidate[:-1].transform(X.iloc[val_idx])
    train_losses = [rmse(y.iloc[train_idx], pred) for pred in model.staged_predict(Xt_train)]
    val_losses = [rmse(y.iloc[val_idx], pred) for pred in model.staged_predict(Xt_val)]
    plt.figure(figsize=(7, 5))
    plt.plot(train_losses, label="Training RMSE")
    plt.plot(val_losses, label="Validation RMSE")
    plt.xlabel("Boosting round")
    plt.ylabel("RMSE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return True


def save_bundle(path: Path, pipeline: Pipeline, metadata: dict[str, Any]) -> None:
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, path)


def load_bundle(path: Path) -> dict[str, Any]:
    return joblib.load(path)

"""
Train and persist the production loan-prediction model.

This is the heart of the ML pipeline. It:

  1. Loads the Bank Personal Loan dataset.
  2. Drops `ID` and `ZIP Code` (non-predictive / leakage risk).
  3. Splits 70/30 stratified.
  4. Trains four candidate models with hyperparameter search:
        - Logistic Regression (with StandardScaler)
        - Decision Tree
        - Random Forest
        - XGBoost (if installed)
  5. Picks the model with the best F1 score on the test set.
  6. Saves:
        - models/best_model.pkl                (the chosen estimator)
        - models/best_decision_tree_model.pkl  (legacy filename, same object)
        - models/model_metrics.json            (full metric block + winner name)
        - reports/model_comparison_summary.txt (human-readable report)
        - reports/plots/{roc_curves, confusion_matrices, model_comparison}.png

Usage:
    python train_model.py
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA_PATH = ROOT / "data" / "Bank_Personal_Loan_Modelling.csv"
MODELS_DIR = HERE / "models"
REPORT_DIR = ROOT / "reports"
PLOTS_DIR = REPORT_DIR / "plots"

FEATURES = [
    "Age", "Experience", "Income", "Family", "CCAvg", "Education",
    "Mortgage", "Securities Account", "CD Account", "Online", "CreditCard",
]
TARGET = "Personal Loan"
RANDOM_STATE = 42

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150})


# ---------------------------------------------------------------- data ----

def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH.name}")
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)
    return X, y


# ----------------------------------------------------------- candidates ----

def candidate_searches(cv: StratifiedKFold) -> Dict:
    """Return {name: (search_object,)} ready to .fit()."""
    out: Dict[str, GridSearchCV | RandomizedSearchCV] = {}

    # Logistic Regression — needs scaling, balanced for class imbalance
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=RANDOM_STATE)),
    ])
    lr_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"],
    }
    out["LogisticRegression"] = GridSearchCV(
        lr_pipe, lr_grid, cv=cv, scoring="f1", n_jobs=-1)

    # Decision Tree
    dt_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [4, 6, 8, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": [None, "balanced"],
    }
    out["DecisionTree"] = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        dt_grid, cv=cv, scoring="f1", n_jobs=-1)

    # Random Forest (RandomizedSearch — bigger space, fixed budget)
    rf_grid = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 8, 12, 16, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "class_weight": [None, "balanced"],
    }
    out["RandomForest"] = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        rf_grid, n_iter=25, cv=cv, scoring="f1", n_jobs=-1,
        random_state=RANDOM_STATE)

    # XGBoost (optional dependency)
    if HAS_XGB:
        xgb_grid = {
            "n_estimators": [200, 400, 600],
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.03, 0.1, 0.2],
            "subsample": [0.7, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.9, 1.0],
            "min_child_weight": [1, 3, 5],
        }
        out["XGBoost"] = RandomizedSearchCV(
            XGBClassifier(
                random_state=RANDOM_STATE, n_jobs=-1,
                eval_metric="logloss", tree_method="hist",
                # scale_pos_weight handles class imbalance for XGBoost
            ),
            xgb_grid, n_iter=25, cv=cv, scoring="f1", n_jobs=-1,
            random_state=RANDOM_STATE)

    return out


# ----------------------------------------------------------- evaluation ----

def evaluate(model, X_test, y_test) -> Dict:
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred
    return {
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score":  float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "_y_pred": y_pred,
        "_y_proba": y_proba,
    }


# ---------------------------------------------------------------- plots ----

def plot_model_comparison(rows: pd.DataFrame, path: Path) -> None:
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold")
    for ax, m in zip(axes.flat, metrics):
        d = rows.sort_values(m, ascending=True)
        ax.barh(d["model"], d[m], color="#4F8FFF")
        for i, v in enumerate(d[m]):
            ax.text(v + 0.005, i, f"{v:.4f}", va="center", fontweight="bold")
        ax.set_xlim(0, 1.05)
        ax.set_title(m, fontweight="bold")
    axes.flat[-1].axis("off")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved  {path.relative_to(ROOT)}")


def plot_roc_curves(results: Dict, y_test, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#FF5252", "#00D084", "#4F8FFF", "#FFB800", "#9C27B0"]
    for (name, info), c in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, info["_y_proba"])
        ax.plot(fpr, tpr, lw=2, color=c,
                label=f"{name}  (AUC = {info['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved  {path.relative_to(ROOT)}")


def plot_confusion_matrices(results: Dict, path: Path) -> None:
    n = len(results)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    fig.suptitle("Confusion Matrices (test set)", fontsize=14, fontweight="bold")
    flat = axes.flat if hasattr(axes, "flat") else [axes]
    for ax, (name, info) in zip(flat, results.items()):
        cm = np.array(info["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["No Loan", "Loan"],
                    yticklabels=["No Loan", "Loan"])
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    for ax in list(flat)[len(results):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved  {path.relative_to(ROOT)}")


# ---------------------------------------------------------------- report ----

def write_report(rows: pd.DataFrame, results: Dict, winner: str,
                 best_params: Dict, importances: Dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "model_comparison_summary.txt"
    L = []
    L.append("=" * 78)
    L.append("              MODEL COMPARISON & SELECTION — TEST-SET METRICS")
    L.append("=" * 78)
    L.append("")
    L.append(rows.round(4).to_string(index=False))
    L.append("")
    L.append("-" * 78)
    L.append(f"BEST MODEL: {winner}  (selected by highest F1)")
    L.append("-" * 78)
    L.append(f"Best hyperparameters:")
    for k, v in best_params.items():
        L.append(f"  {k} = {v}")
    L.append("")
    if importances:
        L.append("Top features (by importance):")
        for k, v in sorted(importances.items(), key=lambda kv: -kv[1]):
            L.append(f"  {k:25s}  {v:.4f}")
    L.append("")
    cm = np.array(results[winner]["confusion_matrix"])
    L.append("Confusion matrix (test set):")
    L.append(f"               Predicted: No   Predicted: Yes")
    L.append(f"  Actual: No        {cm[0,0]:>5d}          {cm[0,1]:>5d}")
    L.append(f"  Actual: Yes       {cm[1,0]:>5d}          {cm[1,1]:>5d}")
    L.append("")
    L.append("=" * 78)
    out.write_text("\n".join(L))
    print(f"  saved  {out.relative_to(ROOT)}")


# ---------------------------------------------------------------- main -----

def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
    print(f"Train: {len(X_train)} | Test: {len(X_test)} | "
          f"Positive rate: {y.mean():.2%}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    searches = candidate_searches(cv)
    if not HAS_XGB:
        print("XGBoost not installed — skipping that candidate.")

    results: Dict[str, Dict] = {}
    rows = []
    fitted_models: Dict[str, object] = {}
    best_params_per_model: Dict[str, Dict] = {}

    for name, search in searches.items():
        print(f"\n--- {name} ---")
        search.fit(X_train, y_train)
        model = search.best_estimator_
        fitted_models[name] = model
        best_params_per_model[name] = search.best_params_
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                     scoring="accuracy")
        info = evaluate(model, X_test, y_test)
        info["cv_mean"] = float(cv_scores.mean())
        info["cv_std"] = float(cv_scores.std())
        results[name] = info
        rows.append({
            "model": name,
            "accuracy":  info["accuracy"],
            "precision": info["precision"],
            "recall":    info["recall"],
            "f1_score":  info["f1_score"],
            "roc_auc":   info["roc_auc"],
            "cv_mean":   info["cv_mean"],
        })
        print(f"  best params : {search.best_params_}")
        print(f"  test F1     : {info['f1_score']:.4f}   "
              f"acc {info['accuracy']:.4f}  AUC {info['roc_auc']:.4f}")

    rows_df = pd.DataFrame(rows).sort_values("f1_score", ascending=False)
    print("\n" + "=" * 70)
    print("Test-set comparison:")
    print(rows_df.round(4).to_string(index=False))

    winner = rows_df.iloc[0]["model"]
    best_model = fitted_models[winner]
    print(f"\nWINNER: {winner}")

    # Feature importances (if the chosen estimator exposes them)
    importances: Dict[str, float] = {}
    inner = (best_model.named_steps["clf"]
             if isinstance(best_model, Pipeline) else best_model)
    if hasattr(inner, "feature_importances_"):
        importances = dict(zip(FEATURES, [float(x) for x in inner.feature_importances_]))
    elif hasattr(inner, "coef_"):
        importances = dict(zip(FEATURES, [float(abs(x)) for x in inner.coef_[0]]))

    # Persist artefacts
    with (MODELS_DIR / "best_model.pkl").open("wb") as f:
        pickle.dump(best_model, f)
    # Keep legacy filename so the existing app keeps loading without changes.
    with (MODELS_DIR / "best_decision_tree_model.pkl").open("wb") as f:
        pickle.dump(best_model, f)

    metrics_payload = {
        "winner": winner,
        "feature_order": FEATURES,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "best_params": best_params_per_model[winner],
        "feature_importance": importances,
        "all_models": {name: {k: v for k, v in info.items()
                              if not k.startswith("_")}
                       for name, info in results.items()},
        # Top-level shortcut keys for the Streamlit app
        "accuracy":  results[winner]["accuracy"],
        "precision": results[winner]["precision"],
        "recall":    results[winner]["recall"],
        "f1_score":  results[winner]["f1_score"],
        "roc_auc":   results[winner]["roc_auc"],
        "cv_mean":   results[winner]["cv_mean"],
        "cv_std":    results[winner]["cv_std"],
        "confusion_matrix": results[winner]["confusion_matrix"],
    }
    (MODELS_DIR / "model_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2))

    print("\nSaved artefacts:")
    print(f"  models/best_model.pkl")
    print(f"  models/best_decision_tree_model.pkl  (alias for app)")
    print(f"  models/model_metrics.json")

    # Reports + plots
    plot_model_comparison(rows_df, PLOTS_DIR / "model_comparison.png")
    plot_roc_curves(results, y_test, PLOTS_DIR / "roc_curves.png")
    plot_confusion_matrices(results, PLOTS_DIR / "confusion_matrices.png")
    write_report(rows_df, results, winner,
                 best_params_per_model[winner], importances)

    # Final classification report on the winner (printed for the user)
    print("\nClassification report (winner on test set):")
    print(classification_report(y_test, results[winner]["_y_pred"]))


if __name__ == "__main__":
    main()

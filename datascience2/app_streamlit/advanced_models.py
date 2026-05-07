"""
advanced_models.py

High-performance ML models: XGBoost and LightGBM with comprehensive comparison.

Features:
- Trains XGBoost and LightGBM with tuned hyperparameters
- Compares with LogisticRegression, DecisionTree, RandomForest
- Comprehensive evaluation: accuracy, precision, recall, F1, ROC-AUC
- Confusion matrices and ROC curves
- Feature importance plots (XGBoost vs LightGBM)
- Model comparison visualizations
- Professional summary report

Usage:
    python advanced_models.py --data PATH/TO/csv [--output results/]

Output:
    - plots/ directory with all visualizations
    - results_summary.csv with metrics comparison
    - model_comparison.txt with detailed insights
"""

import argparse
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Install via: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not installed. Install via: pip install lightgbm")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib/seaborn not installed. Plots will be skipped.")


def find_data(path: str = None) -> pd.DataFrame:
    """Locate and load dataset from various standard locations."""
    candidates = []
    if path:
        candidates.append(path)
    base = os.path.dirname(__file__)
    candidates += [
        os.path.join(base, "data", "bank_personal_loan.csv"),
        os.path.join(base, "data", "Bank_Personal_Loan_Modelling.csv"),
        os.path.join(base, "..", "data", "bank_personal_loan.csv"),
        os.path.join(base, "..", "data", "Bank_Personal_Loan_Modelling.csv"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            print(f"✓ Loading data from: {p}")
            return pd.read_csv(p)
    raise FileNotFoundError("No dataset found. Provide --data PATH or place csv in data/ folder.")


def standardize_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Standardize target column name."""
    target_candidates = ["Loan", "PersonalLoan", "Personal Loan", "loan"]
    for c in target_candidates:
        if c in df.columns:
            df = df.copy()
            df.rename(columns={c: "Loan"}, inplace=True)
            df["Loan"] = df["Loan"].astype(int)
            return df, "Loan"
    # Fallback: find binary column
    for c in df.columns:
        vals = df[c].dropna().unique()
        if set(vals).issubset({0, 1}):
            df = df.copy()
            df.rename(columns={c: "Loan"}, inplace=True)
            df["Loan"] = df["Loan"].astype(int)
            return df, "Loan"
    raise ValueError("Could not find binary target column.")


def prepare_data(
    df: pd.DataFrame, target: str = "Loan", test_size=0.2, random_state=42
):
    """Prepare train/test splits with stratification."""
    X = df.drop(columns=[target])
    y = df[target]
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, numeric_cols


def train_baseline_models(X_train, X_test, y_train, y_test) -> Dict:
    """Train baseline models: LogisticRegression, DecisionTree, RandomForest."""
    results = {}

    # Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

    results["LogisticRegression"] = {
        "model": lr,
        "y_pred": y_pred_lr,
        "y_proba": y_proba_lr,
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
    }

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    y_proba_dt = dt.predict_proba(X_test)[:, 1]

    results["DecisionTree"] = {
        "model": dt,
        "y_pred": y_pred_dt,
        "y_proba": y_proba_dt,
        "X_train": X_train,
        "X_test": X_test,
    }

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]

    results["RandomForest"] = {
        "model": rf,
        "y_pred": y_pred_rf,
        "y_proba": y_proba_rf,
        "X_train": X_train,
        "X_test": X_test,
    }

    return results


def train_xgboost(X_train, X_test, y_train, y_test) -> Dict:
    """Train XGBoost with tuned hyperparameters."""
    if not HAS_XGBOOST:
        print("⚠ XGBoost not available. Skipping.")
        return {}

    params = {
        "n_estimators": 300,
        "max_depth": 7,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.5,
        "objective": "binary:logistic",
        "random_state": 42,
        "eval_metric": "logloss",
    }

    xgb_model = xgb.XGBClassifier(**params)
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = xgb_model.predict(X_test)
    y_proba = xgb_model.predict_proba(X_test)[:, 1]

    return {
        "XGBoost": {
            "model": xgb_model,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "X_train": X_train,
            "X_test": X_test,
            "params": params,
        }
    }


def train_lightgbm(X_train, X_test, y_train, y_test) -> Dict:
    """Train LightGBM with tuned hyperparameters."""
    if not HAS_LIGHTGBM:
        print("⚠ LightGBM not available. Skipping.")
        return {}

    params = {
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.5,
        "objective": "binary",
        "random_state": 42,
    }

    lgb_model = lgb.LGBMClassifier(**params, verbose=-1)
    lgb_model.fit(X_train, y_train)

    y_pred = lgb_model.predict(X_test)
    y_proba = lgb_model.predict_proba(X_test)[:, 1]

    return {
        "LightGBM": {
            "model": lgb_model,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "X_train": X_train,
            "X_test": X_test,
            "params": params,
        }
    }


def evaluate_model(y_test, y_pred, y_proba) -> Dict:
    """Compute comprehensive metrics."""
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "roc_curve": roc_curve(y_test, y_proba),
    }


def create_output_dir(output_dir: str):
    """Create output directories."""
    for d in [output_dir, os.path.join(output_dir, "plots")]:
        os.makedirs(d, exist_ok=True)


def plot_feature_importance(models: Dict, output_dir: str):
    """Plot feature importance for tree-based and boosting models."""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Feature Importance Comparison", fontsize=16, fontweight="bold")

    models_to_plot = ["DecisionTree", "RandomForest", "XGBoost", "LightGBM"]
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for (model_name, pos), ax in zip(zip(models_to_plot, positions), axes.flat):
        if model_name not in models:
            ax.text(0.5, 0.5, f"{model_name} not available", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        model = models[model_name]["model"]
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            ax.text(0.5, 0.5, "No feature importance", ha="center", va="center")
            continue

        # Top 15 features
        indices = np.argsort(importances)[-15:]
        ax.barh(range(len(indices)), importances[indices], color="#4F8FFF")
        ax.set_yticks(range(len(indices)))
        ax.set_title(f"{model_name} (Top 15)", fontweight="bold")
        ax.set_xlabel("Importance")
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "feature_importance.png"), dpi=300, bbox_inches="tight")
    print(f"✓ Saved: feature_importance.png")
    plt.close()


def plot_roc_curves(models: Dict, y_test, output_dir: str):
    """Plot ROC curves for all models."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#FF5252", "#00D084", "#4F8FFF", "#FFB800", "#9C27B0"]

    for i, (name, res) in enumerate(models.items()):
        fpr, tpr, _ = res.get("roc_curve", ([], [], []))
        if len(fpr) > 0:
            auc_score = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.4f})", color=colors[i % len(colors)], lw=2)

    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "roc_curves.png"), dpi=300, bbox_inches="tight")
    print(f"✓ Saved: roc_curves.png")
    plt.close()


def plot_confusion_matrices(models: Dict, output_dir: str):
    """Plot confusion matrices for all models."""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Confusion Matrices", fontsize=16, fontweight="bold")

    model_names = list(models.keys())[:6]
    for ax, name in zip(axes.flat, model_names):
        cm = models[name]["confusion_matrix"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
        ax.set_title(name, fontweight="bold")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

    # Hide unused subplots
    for i in range(len(model_names), len(axes.flat)):
        axes.flat[i].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "confusion_matrices.png"), dpi=300, bbox_inches="tight")
    print(f"✓ Saved: confusion_matrices.png")
    plt.close()


def plot_model_comparison(metrics_df: pd.DataFrame, output_dir: str):
    """Plot model performance comparison."""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold")

    metrics = ["accuracy", "precision", "recall", "f1"]
    colors_list = ["#FF5252", "#00D084", "#4F8FFF", "#FFB800", "#9C27B0", "#00BCD4"]

    for ax, metric in zip(axes.flat, metrics):
        data = metrics_df[["model", metric]].sort_values(metric, ascending=False)
        ax.barh(data["model"], data[metric], color=colors_list[:len(data)])
        ax.set_xlabel(metric.capitalize(), fontsize=11, fontweight="bold")
        ax.set_title(f"{metric.upper()}", fontweight="bold")
        ax.set_xlim(0, 1)
        for i, v in enumerate(data[metric]):
            ax.text(v + 0.01, i, f"{v:.4f}", va="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "model_comparison.png"), dpi=300, bbox_inches="tight")
    print(f"✓ Saved: model_comparison.png")
    plt.close()


def run_full_pipeline(data_path: str = None, output_dir: str = "results"):
    """Run complete advanced modeling pipeline."""
    create_output_dir(output_dir)

    # Load and prepare
    df = find_data(data_path)
    df, target = standardize_target(df)
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(df, target)

    print(f"\n📊 Dataset: {len(df)} records, {len(feature_cols)} features")
    print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")

    # Train models
    print("\n🤖 Training baseline models...")
    baseline_models = train_baseline_models(X_train, X_test, y_train, y_test)

    print("🚀 Training advanced models...")
    xgb_models = train_xgboost(X_train, X_test, y_train, y_test)
    lgb_models = train_lightgbm(X_train, X_test, y_train, y_test)

    all_models = {**baseline_models, **xgb_models, **lgb_models}

    # Evaluate
    print("\n📈 Evaluating models...")
    metrics_data = []
    for name, res in all_models.items():
        metrics = evaluate_model(y_test, res["y_pred"], res["y_proba"])
        all_models[name].update(metrics)
        metrics_data.append({
            "model": name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
        })

    metrics_df = pd.DataFrame(metrics_data)

    # Save results
    metrics_df.to_csv(os.path.join(output_dir, "metrics_comparison.csv"), index=False)
    print(f"✓ Saved: metrics_comparison.csv")

    # Visualizations
    print("\n📊 Generating visualizations...")
    plot_feature_importance(all_models, output_dir)
    plot_roc_curves(all_models, y_test, output_dir)
    plot_confusion_matrices(all_models, output_dir)
    plot_model_comparison(metrics_df, output_dir)

    # Summary report
    summary = _generate_summary(metrics_df, all_models)
    with open(os.path.join(output_dir, "model_comparison_summary.txt"), "w") as f:
        f.write(summary)
    print(f"✓ Saved: model_comparison_summary.txt")

    print("\n" + "=" * 70)
    print("METRICS SUMMARY")
    print("=" * 70)
    print(metrics_df.to_string(index=False))
    print("\n" + "=" * 70)
    print(summary)

    return metrics_df, all_models


def _generate_summary(metrics_df: pd.DataFrame, models: Dict) -> str:
    """Generate detailed summary report."""
    lines = []
    lines.append("=" * 70)
    lines.append("ADVANCED ML MODEL COMPARISON REPORT")
    lines.append("=" * 70)

    # Best models
    lines.append("\n🏆 BEST PERFORMERS:")
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        best = metrics_df.loc[metrics_df[metric].idxmax()]
        lines.append(f"  • {metric.upper()}: {best['model']} ({best[metric]:.4f})")

    # Insights
    lines.append("\n📊 KEY INSIGHTS:")

    xgb_f1 = metrics_df[metrics_df["model"] == "XGBoost"]["f1"].values
    lgb_f1 = metrics_df[metrics_df["model"] == "LightGBM"]["f1"].values
    rf_f1 = metrics_df[metrics_df["model"] == "RandomForest"]["f1"].values

    if len(xgb_f1) > 0 and len(rf_f1) > 0:
        diff = (xgb_f1[0] - rf_f1[0]) / rf_f1[0] * 100
        lines.append(f"\n  • XGBoost vs RandomForest F1-score: {diff:+.2f}%")
        if diff > 0:
            lines.append(f"    ✓ XGBoost outperforms due to: gradient boosting, \
regularization")
        else:
            lines.append(f"    • RandomForest is simpler, less prone to overfitting here")

    if len(lgb_f1) > 0 and len(rf_f1) > 0:
        diff = (lgb_f1[0] - rf_f1[0]) / rf_f1[0] * 100
        lines.append(f"\n  • LightGBM vs RandomForest F1-score: {diff:+.2f}%")
        if diff > 0:
            lines.append(f"    ✓ LightGBM is faster and uses leaf-wise growth strategy")

    lines.append("\n  • Why Boosting Models Win:")
    lines.append("    1. Sequential error correction: each tree focuses on mistakes")
    lines.append("    2. Regularization: L1/L2 prevents overfitting")
    lines.append("    3. Feature interactions: captures complex patterns")
    lines.append("    4. Adaptive learning: lower learning rate = better generalization")

    lines.append("\n  • Model Characteristics:")
    lines.append("    • Logistic Regression: Fast, interpretable, linear relationships")
    lines.append("    • Decision Tree: Single tree, high variance, quick to train")
    lines.append("    • Random Forest: Ensemble, parallelizable, robust to outliers")
    lines.append("    • XGBoost: Sequential boosting, high accuracy, GPU support")
    lines.append("    • LightGBM: Fast training, memory efficient, leaf-wise growth")

    lines.append("\n" + "=" * 70)
    lines.append("RECOMMENDATION:")
    lines.append("=" * 70)
    best_model = metrics_df.loc[metrics_df["f1"].idxmax(), "model"]
    lines.append(f"✓ Use {best_model} for production deployment")
    lines.append("  - Highest F1-score ensures balanced precision-recall")
    lines.append("  - Hyperparameters tuned to avoid overfitting")
    lines.append("  - Feature importance analysis available for interpretability")

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced ML models (XGBoost, LightGBM) comparison")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV dataset")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    metrics_df, models = run_full_pipeline(data_path=args.data, output_dir=args.output)

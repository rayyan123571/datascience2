"""
shap_explainability.py

SHAP (SHapley Additive exPlanations) for model interpretability and explainability.

Features:
- Train Random Forest and XGBoost models
- Generate SHAP explanations
- Create summary plots (mean |SHAP| values)
- Create force plots (individual predictions)
- Create decision plots (prediction paths)
- BEESWARM plots (feature value vs SHAP effect)
- Interpret results for business stakeholders

This module explains which features drive loan acceptance and by how much.

Usage:
    python shap_explainability.py --data PATH/TO/csv [--output results/]

Output:
    - plots/ directory with visualizations
    - interpretations.txt with business insights
"""

import argparse
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠ SHAP not installed. Install via: pip install shap")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def find_data(path: str = None) -> pd.DataFrame:
    """Locate and load dataset."""
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
    raise FileNotFoundError("No dataset found.")


def standardize_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Standardize target column."""
    for c in ["Loan", "PersonalLoan", "Personal Loan", "loan"]:
        if c in df.columns:
            df = df.copy()
            df.rename(columns={c: "Loan"}, inplace=True)
            df["Loan"] = df["Loan"].astype(int)
            return df, "Loan"
    for c in df.columns:
        if set(df[c].dropna().unique()).issubset({0, 1}):
            df = df.copy()
            df.rename(columns={c: "Loan"}, inplace=True)
            df["Loan"] = df["Loan"].astype(int)
            return df, "Loan"
    raise ValueError("No binary target found.")


def prepare_data(df: pd.DataFrame, target: str = "Loan", test_size=0.2, random_state=42):
    """Prepare train/test splits."""
    X = df.drop(columns=[target])
    y = df[target]
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, numeric_cols


def train_random_forest(X_train, X_test, y_train, y_test) -> Dict:
    """Train Random Forest with tuned hyperparameters."""
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(y_test, rf.predict(X_test))
    f1 = f1_score(y_test, rf.predict(X_test))
    
    return {
        "model": rf,
        "name": "Random Forest",
        "X_train": X_train,
        "X_test": X_test,
        "accuracy": acc,
        "f1": f1,
    }


def train_xgboost(X_train, X_test, y_train, y_test) -> Dict:
    """Train XGBoost with tuned hyperparameters."""
    if not HAS_XGBOOST:
        return None

    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.5,
        random_state=42,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(y_test, xgb_model.predict(X_test))
    f1 = f1_score(y_test, xgb_model.predict(X_test))
    
    return {
        "model": xgb_model,
        "name": "XGBoost",
        "X_train": X_train,
        "X_test": X_test,
        "accuracy": acc,
        "f1": f1,
    }


def create_shap_explainer(model, X_train, model_name: str):
    """Create SHAP explainer based on model type."""
    if not HAS_SHAP:
        return None

    if "XGBoost" in model_name or "xgb" in str(type(model)).lower():
        explainer = shap.TreeExplainer(model)
    elif "RandomForest" in model_name or "random" in str(type(model)).lower():
        # Use TreeExplainer for tree ensemble
        explainer = shap.TreeExplainer(model)
    else:
        # Fallback to KernelExplainer for other models
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 100))
    
    return explainer


def plot_shap_summary(explainer, X_test, model_name: str, output_dir: str):
    """Plot SHAP summary plot (mean absolute SHAP values)."""
    if not HAS_SHAP or not HAS_MATPLOTLIB:
        return

    shap_values = explainer.shap_values(X_test)
    
    # Handle multi-class output (take class 1 for binary)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title(f"{model_name} - Feature Importance (Mean |SHAP|)", fontsize=14, fontweight="bold", pad=20)
    plt.xlabel("Mean |SHAP value|", fontsize=12)
    plt.tight_layout()
    filepath = os.path.join(output_dir, f"shap_summary_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {os.path.basename(filepath)}")
    plt.close()


def plot_shap_beeswarm(explainer, X_test, model_name: str, output_dir: str):
    """Plot SHAP beeswarm (feature values vs SHAP effects)."""
    if not HAS_SHAP or not HAS_MATPLOTLIB:
        return

    shap_values = explainer.shap_values(X_test)
    
    # Handle multi-class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, plot_type="violin", show=False)
    plt.title(f"{model_name} - Feature Impact Distribution", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    filepath = os.path.join(output_dir, f"shap_beeswarm_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {os.path.basename(filepath)}")
    plt.close()


def plot_shap_force(explainer, X_test, y_test, model_name: str, output_dir: str, sample_idx: int = 0):
    """Plot SHAP force plot for individual prediction."""
    if not HAS_SHAP or not HAS_MATPLOTLIB:
        return

    shap_values = explainer.shap_values(X_test)
    
    # Handle multi-class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    base_value = explainer.expected_value
    if isinstance(base_value, list):
        base_value = base_value[1]

    # Create force plot
    plt.figure(figsize=(14, 4))
    shap.force_plot(
        base_value,
        shap_values[sample_idx],
        X_test.iloc[sample_idx],
        matplotlib=True,
        show=False,
    )
    plt.title(
        f"{model_name} - Individual Prediction Explanation\n"
        f"Sample {sample_idx} (Actual: {'Accepted' if y_test.iloc[sample_idx] == 1 else 'Rejected'})",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    plt.tight_layout()
    filepath = os.path.join(output_dir, f"shap_force_{model_name.lower().replace(' ', '_')}_sample_{sample_idx}.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {os.path.basename(filepath)}")
    plt.close()


def plot_shap_waterfall(explainer, X_test, model_name: str, output_dir: str, sample_idx: int = 0):
    """Plot SHAP waterfall plot (decision path)."""
    if not HAS_SHAP or not HAS_MATPLOTLIB:
        return

    shap_values = explainer.shap_values(X_test)
    
    # Handle multi-class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    base_value = explainer.expected_value
    if isinstance(base_value, list):
        base_value = base_value[1]

    # Create explanation object
    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=base_value,
        data=X_test.iloc[sample_idx],
        feature_names=X_test.columns.tolist(),
    )

    plt.figure(figsize=(12, 8))
    shap.waterfall_plot(explanation, show=False)
    plt.title(f"{model_name} - Prediction Decision Path (Sample {sample_idx})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    filepath = os.path.join(output_dir, f"shap_waterfall_{model_name.lower().replace(' ', '_')}_sample_{sample_idx}.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {os.path.basename(filepath)}")
    plt.close()


def generate_interpretations(models_dict: Dict, output_dir: str) -> str:
    """Generate business-friendly interpretations."""
    lines = []
    lines.append("=" * 80)
    lines.append("SHAP EXPLAINABILITY REPORT - MODEL INTERPRETATION GUIDE")
    lines.append("=" * 80)

    lines.append("\n📊 WHAT IS SHAP?")
    lines.append("-" * 80)
    lines.append("SHAP values explain how each feature contributes to a prediction.")
    lines.append("Think of it as: 'What would change if we removed this feature?'")
    lines.append("\nRed color → Feature INCREASES loan acceptance probability")
    lines.append("Blue color → Feature DECREASES loan acceptance probability")
    lines.append("Size → How much the feature impacts the decision")

    lines.append("\n\n🎯 KEY FINDINGS BY MODEL")
    lines.append("-" * 80)

    for name, model_info in models_dict.items():
        if model_info is None:
            continue

        lines.append(f"\n{name}:")
        lines.append(f"  • Accuracy: {model_info['accuracy']:.4f}")
        lines.append(f"  • F1-Score: {model_info['f1']:.4f}")

        # Feature importance from model
        model = model_info["model"]
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_names = model_info["X_train"].columns.tolist()
            top_indices = np.argsort(importances)[-5:]
            lines.append(f"  • Top 5 Most Important Features:")
            for idx in reversed(top_indices):
                lines.append(f"    - {feature_names[idx]}: {importances[idx]:.4f}")

    lines.append("\n\n💡 BUSINESS INTERPRETATIONS")
    lines.append("-" * 80)

    lines.append("\n1. INCOME (Usually Highest Impact)")
    lines.append("   • Higher income → STRONG positive signal for loan acceptance")
    lines.append("   • Why: Banks trust high-income customers to repay")
    lines.append("   • Action: Target high-income segments for loan offers")

    lines.append("\n2. CREDIT CARD AVERAGE (CCAvg) - Second Highest")
    lines.append("   • High credit card spending → Strong positive signal")
    lines.append("   • Why: Shows customer handles credit responsibly")
    lines.append("   • Action: Cross-sell loans to existing high-spend credit card customers")

    lines.append("\n3. EXPERIENCE & AGE")
    lines.append("   • More experience/age → Moderate positive effect")
    lines.append("   • Why: Stability & established financial history")
    lines.append("   • Action: Target professionals with 10+ years experience")

    lines.append("\n4. FAMILY SIZE & MORTGAGE")
    lines.append("   • Family size → Moderate impact (mixed effects)")
    lines.append("   • Mortgage → Shows financial responsibility")
    lines.append("   • Action: Use as secondary filters alongside income")

    lines.append("\n5. EDUCATION LEVEL")
    lines.append("   • Higher education → Slight positive effect")
    lines.append("   • Why: Correlates with income and job stability")
    lines.append("   • Action: Mention it in targeting but not primary criterion")

    lines.append("\n\n🔍 HOW TO READ THE PLOTS")
    lines.append("-" * 80)

    lines.append("\n1. SUMMARY PLOT (Bar Chart)")
    lines.append("   • Shows average impact of each feature")
    lines.append("   • Longer bars = more important features")
    lines.append("   • Colors show direction (red=positive, blue=negative)")

    lines.append("\n2. BEESWARM PLOT (Scatter)")
    lines.append("   • Each dot = one prediction")
    lines.append("   • Horizontal position = SHAP value (impact on prediction)")
    lines.append("   • Color = actual feature value (red=high, blue=low)")
    lines.append("   • Pattern shows if higher values help/hurt acceptance")

    lines.append("\n3. FORCE PLOT (Individual)")
    lines.append("   • Explains ONE specific prediction")
    lines.append("   • Red arrows push toward 'Accepted'")
    lines.append("   • Blue arrows push toward 'Rejected'")
    lines.append("   • Starting point = baseline probability")
    lines.append("   • Ending point = final prediction")

    lines.append("\n4. WATERFALL PLOT (Decision Path)")
    lines.append("   • Shows step-by-step how features pushed decision")
    lines.append("   • Top = most influential features")
    lines.append("   • Height = magnitude of impact")
    lines.append("   • Direction = positive (right) or negative (left) impact")

    lines.append("\n\n📋 PRACTICAL RECOMMENDATIONS")
    lines.append("-" * 80)
    lines.append("\n✓ DO:")
    lines.append("  1. Prioritize income and credit card spending in targeting")
    lines.append("  2. Use SHAP for customer-facing explanations")
    lines.append("  3. Monitor if SHAP values align with business logic")
    lines.append("  4. Retrain model if SHAP patterns change unexpectedly")

    lines.append("\n✗ DON'T:")
    lines.append("  1. Over-rely on single features (use all together)")
    lines.append("  2. Assume SHAP values = causality (correlation, not causation)")
    lines.append("  3. Ignore fairness implications of top features")
    lines.append("  4. Show raw SHAP values to customers (explain in simple terms)")

    lines.append("\n\n🎓 FOR NON-TECHNICAL STAKEHOLDERS")
    lines.append("-" * 80)
    lines.append("\nSimple Explanation:")
    lines.append("'Our loan prediction model looks at customer income, spending habits,")
    lines.append("and experience. SHAP tells us HOW MUCH each factor matters in the decision.'")
    lines.append("\nExample:")
    lines.append("'If a customer has high income and high credit card spending,")
    lines.append("those two factors strongly support approving their loan.'")

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)


def create_output_dir(output_dir: str):
    """Create output directories."""
    for d in [output_dir, os.path.join(output_dir, "plots")]:
        os.makedirs(d, exist_ok=True)


def run_shap_pipeline(data_path: str = None, output_dir: str = "results"):
    """Run complete SHAP explainability pipeline."""
    if not HAS_SHAP:
        print("❌ SHAP not installed. Install via: pip install shap")
        return

    create_output_dir(output_dir)

    # Load and prepare
    df = find_data(data_path)
    df, target = standardize_target(df)
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(df, target)

    print(f"\n📊 Dataset: {len(df)} records, {len(feature_cols)} features")
    print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")

    # Train models
    models_dict = {}

    print("\n🤖 Training Random Forest...")
    rf_result = train_random_forest(X_train, X_test, y_train, y_test)
    models_dict["Random Forest"] = rf_result

    print("🤖 Training XGBoost...")
    if HAS_XGBOOST:
        xgb_result = train_xgboost(X_train, X_test, y_train, y_test)
        if xgb_result:
            models_dict["XGBoost"] = xgb_result
    else:
        print("⚠ XGBoost not available")

    # Generate SHAP explanations
    print("\n🔍 Generating SHAP explanations...")
    for model_name, model_info in models_dict.items():
        print(f"\n  {model_name}:")
        model = model_info["model"]
        X_train_m = model_info["X_train"]
        X_test_m = model_info["X_test"]

        # Create explainer
        explainer = create_shap_explainer(model, X_train_m, model_name)
        if explainer is None:
            print(f"    ⚠ Could not create explainer for {model_name}")
            continue

        # Generate plots
        plot_dir = os.path.join(output_dir, "plots")
        print(f"    • Generating summary plot...")
        plot_shap_summary(explainer, X_test_m, model_name, plot_dir)

        print(f"    • Generating beeswarm plot...")
        plot_shap_beeswarm(explainer, X_test_m, model_name, plot_dir)

        print(f"    • Generating force plot (sample 0)...")
        plot_shap_force(explainer, X_test_m, y_test, model_name, plot_dir, sample_idx=0)

        print(f"    • Generating waterfall plot (sample 0)...")
        plot_shap_waterfall(explainer, X_test_m, model_name, plot_dir, sample_idx=0)

        # Find interesting samples (high and low acceptance probability)
        proba = model.predict_proba(X_test_m)[:, 1]
        high_idx = np.argmax(proba)
        low_idx = np.argmin(proba)

        print(f"    • Generating force plot (high acceptance sample {high_idx})...")
        plot_shap_force(explainer, X_test_m, y_test, model_name, plot_dir, sample_idx=high_idx)

        print(f"    • Generating waterfall plot (low acceptance sample {low_idx})...")
        plot_shap_waterfall(explainer, X_test_m, model_name, plot_dir, sample_idx=low_idx)

    # Generate interpretations
    print("\n📝 Generating business interpretations...")
    interp = generate_interpretations(models_dict, output_dir)
    interp_file = os.path.join(output_dir, "shap_interpretations.txt")
    with open(interp_file, "w") as f:
        f.write(interp)
    print(f"✓ Saved: shap_interpretations.txt")

    # Print summary
    print("\n" + "=" * 80)
    print(interp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP explainability analysis")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV dataset")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    run_shap_pipeline(data_path=args.data, output_dir=args.output)

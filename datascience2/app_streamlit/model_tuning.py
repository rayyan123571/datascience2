"""
model_tuning.py

Industrial-style ML hyperparameter tuning pipeline using GridSearchCV.

Features:
- Loads dataset (path or default candidates)
- Prepares train/test splits with stratification
- Builds sklearn Pipelines for LogisticRegression, DecisionTree, RandomForest
- Runs baseline (untuned) training and evaluation
- Runs GridSearchCV with meaningful parameter grids (cv=5)
- Compares before/after performance and prints summary table

Usage:
    python model_tuning.py --data PATH/TO/csv

"""

import argparse
import os
import pickle
from typing import Dict, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def find_data(path: str = None) -> pd.DataFrame:
    candidates = []
    if path:
        candidates.append(path)
    base = os.path.dirname(__file__)
    candidates += [
        os.path.join(base, "data", "Bank_Personal_Loan_Modelling.csv"),
        os.path.join(base, "data", "bank_personal_loan.csv"),
        os.path.join(base, "data", "bank.csv"),
        os.path.join(base, "..", "data", "Bank_Personal_Loan_Modelling.csv"),
        os.path.join(base, "..", "data", "bank_personal_loan.csv"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            print(f"Loading data from: {p}")
            return pd.read_csv(p)
    raise FileNotFoundError("No dataset found. Provide --data PATH or place csv in data/ folder.")


def standardize_target(df: pd.DataFrame, target_candidates=None) -> Tuple[pd.DataFrame, str]:
    if target_candidates is None:
        target_candidates = ["Loan", "PersonalLoan", "Personal Loan", "loan"]
    for c in target_candidates:
        if c in df.columns:
            df = df.copy()
            df.rename(columns={c: "Loan"}, inplace=True)
            # Ensure binary (0/1)
            df["Loan"] = df["Loan"].astype(int)
            return df, "Loan"
    # fallback: if there is a column with only 0/1 values
    for c in df.columns:
        vals = df[c].dropna().unique()
        if set(vals).issubset({0, 1}):
            df = df.copy()
            df.rename(columns={c: "Loan"}, inplace=True)
            df["Loan"] = df["Loan"].astype(int)
            return df, "Loan"
    raise ValueError("Could not find binary target column in dataset.")


def prepare_data(df: pd.DataFrame, target: str = "Loan", test_size=0.2, random_state=42):
    # Choose features: numeric columns except the target
    X = df.drop(columns=[target])
    y = df[target]
    # keep numeric features only for this pipeline
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, numeric_cols


def build_pipelines(feature_cols):
    # Use scaler for numeric features
    scaler = StandardScaler()

    # Pipelines place the scaler first, classifier under key 'clf'
    pipe_log = Pipeline([("scaler", scaler), ("clf", LogisticRegression(max_iter=20000))])
    pipe_tree = Pipeline([("scaler", scaler), ("clf", DecisionTreeClassifier(random_state=42))])
    pipe_rf = Pipeline([("scaler", scaler), ("clf", RandomForestClassifier(random_state=42))])

    return {"LogisticRegression": pipe_log, "DecisionTree": pipe_tree, "RandomForest": pipe_rf}


def baseline_evaluate(pipelines: Dict[str, Pipeline], X_train, X_test, y_train, y_test):
    results = {}
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "model": pipe,
        }
    return results


def get_param_grids():
    # Meaningful, not trivial grids; structured as list/dicts to avoid incompatible LR params
    lr_grid = [
        {
            "clf__solver": ["saga"],
            "clf__penalty": ["l1", "l2"],
            "clf__C": [0.01, 0.1, 1.0, 10.0],
        },
        {
            "clf__solver": ["saga"],
            "clf__penalty": ["elasticnet"],
            "clf__C": [0.01, 0.1, 1.0],
            "clf__l1_ratio": [0.2, 0.5],
        },
        {
            "clf__solver": ["lbfgs", "newton-cg"],
            "clf__penalty": ["l2"],
            "clf__C": [0.1, 1.0, 10.0],
        },
    ]

    tree_grid = {
        "clf__criterion": ["gini", "entropy"],
        "clf__max_depth": [3, 5, 7, 10, None],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": [None, "sqrt", "log2"],
    }

    rf_grid = {
        "clf__n_estimators": [100, 300, 500],
        "clf__max_depth": [None, 10, 20, 30],
        "clf__max_features": ["sqrt", "log2", 0.3, 0.5],
        "clf__min_samples_split": [2, 5, 10],
    }

    return {"LogisticRegression": lr_grid, "DecisionTree": tree_grid, "RandomForest": rf_grid}


def tune_model(pipe: Pipeline, param_grid, X_train, y_train, cv=5, scoring="f1", n_jobs=-1):
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, param_grid, cv=cv_strategy, scoring=scoring, n_jobs=n_jobs, verbose=1)
    gs.fit(X_train, y_train)
    return gs


def evaluate_pipeline(pipe, X_test, y_test) -> Dict[str, float]:
    preds = pipe.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
    }


def save_model_artifact(model, output_path: str) -> str:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb") as f:
        pickle.dump(model, f)
    return str(output_file)


def run_tuning(data_path: str = None, test_size=0.2, cv=5):
    df = find_data(data_path)
    df, target = standardize_target(df)
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(df, target, test_size=test_size)

    pipelines = build_pipelines(feature_cols)
    print("Running baseline evaluation (untuned models)...")
    baseline = baseline_evaluate(pipelines, X_train, X_test, y_train, y_test)

    param_grids = get_param_grids()

    tuned_results = {}
    for name, pipe in pipelines.items():
        print(f"\nTuning {name} ...")
        grid = param_grids[name]
        gs = tune_model(pipe, grid, X_train, y_train, cv=cv)
        best = gs.best_estimator_
        tuned_metrics = evaluate_pipeline(best, X_test, y_test)
        tuned_results[name] = {
            "best_params": gs.best_params_,
            "metrics": tuned_metrics,
            "best_estimator": best,
            "cv_results": gs.cv_results_,
        }

    # Build comparison table
    rows = []
    for name in pipelines.keys():
        base = baseline[name]
        tuned = tuned_results[name]
        rows.append(
            {
                "model": name,
                "stage": "baseline",
                "accuracy": base["accuracy"],
                "precision": base["precision"],
                "recall": base["recall"],
                "f1": base["f1"],
            }
        )
        rows.append(
            {
                "model": name,
                "stage": "tuned",
                "accuracy": tuned["metrics"]["accuracy"],
                "precision": tuned["metrics"]["precision"],
                "recall": tuned["metrics"]["recall"],
                "f1": tuned["metrics"]["f1"],
            }
        )

    cmp_df = pd.DataFrame(rows)
    print("\nTuning complete. Summary table:\n")
    print(cmp_df.pivot_table(index=["model"], columns="stage", values=["accuracy", "precision", "recall", "f1"]))

    # Print best params
    for name, res in tuned_results.items():
        print(f"\n{name} best params:")
        print(res["best_params"])

    best_tree_model = tuned_results["DecisionTree"]["best_estimator"]
    model_path = save_model_artifact(
        best_tree_model,
        os.path.join(os.path.dirname(__file__), "models", "best_decision_tree_model.pkl"),
    )
    print(f"\nSaved best DecisionTree model to: {model_path}")

    return {
        "baseline": baseline,
        "tuned": tuned_results,
        "comparison": cmp_df,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for loan models")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV dataset")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split proportion")
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds")
    args = parser.parse_args()

    results = run_tuning(data_path=args.data, test_size=args.test_size, cv=args.cv)

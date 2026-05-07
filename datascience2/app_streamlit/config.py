"""Configuration constants for the loan-prediction Streamlit app."""

import json
from pathlib import Path

APP_TITLE = "Bank Personal Loan Prediction"
APP_SUBTITLE = "AI-Powered Loan Acceptance Prediction System"
APP_DESCRIPTION = (
    "Predict whether a customer will accept a personal loan offer using a "
    "Decision Tree model trained on the Bank Personal Loan dataset."
)

_HERE = Path(__file__).parent
MODEL_PATH = str(_HERE / "models" / "best_decision_tree_model.pkl")
METRICS_PATH = str(_HERE / "models" / "model_metrics.json")

# Order MUST match training. See train_model.py: FEATURES.
FEATURE_ORDER = [
    "Age", "Experience", "Income", "Family", "CCAvg", "Education",
    "Mortgage", "Securities Account", "CD Account", "Online", "CreditCard",
]

FEATURE_DESCRIPTIONS = {
    "Age":          {"description": "Customer's age in years",
                     "min": 23, "max": 67, "default": 45, "unit": "years"},
    "Experience":   {"description": "Years of professional experience",
                     "min": 0, "max": 43, "default": 20, "unit": "years"},
    "Income":       {"description": "Annual income in thousands of dollars",
                     "min": 25, "max": 224, "default": 75, "unit": "$1000s"},
    "Family":       {"description": "Number of family members",
                     "min": 1, "max": 4, "default": 2, "unit": "members"},
    "CCAvg":        {"description": "Average credit card spending in $1000s",
                     "min": 0, "max": 10, "default": 2, "unit": "$1000s"},
    "Education":    {"description": "Education level (1=Undergrad, 2=Graduate, 3=Advanced)",
                     "min": 1, "max": 3, "default": 1, "unit": "level"},
    "Mortgage":     {"description": "Mortgage amount in thousands of dollars",
                     "min": 0, "max": 635, "default": 0, "unit": "$1000s"},
}

DEMOGRAPHIC_FEATURES = {k: FEATURE_DESCRIPTIONS[k] for k in ("Age", "Family", "Education")}
FINANCIAL_FEATURES   = {k: FEATURE_DESCRIPTIONS[k] for k in ("Income", "CCAvg", "Mortgage")}
PROFESSIONAL_FEATURES = {k: FEATURE_DESCRIPTIONS[k] for k in ("Experience",)}

EDUCATION_MAPPING = {1: "Undergraduate", 2: "Graduate", 3: "Advanced Degree"}

COLORS = {
    "accept":         "#00D084",
    "reject":         "#FF5252",
    "neutral":        "#4F8FFF",
    "warning":        "#FFB800",
    "background":     "#F5F7FA",
    "text_primary":   "#1E293B",
    "text_secondary": "#64748B",
}

# Defaults shown until the real metrics file is loaded by app.py.
# These get overridden at runtime — see app.load_model_resource().
MODEL_METRICS = {
    "accuracy":  0.0, "precision": 0.0, "recall": 0.0,
    "f1_score":  0.0, "cv_mean":   0.0, "cv_std":  0.0,
}
FEATURE_IMPORTANCE: dict = {}


def load_runtime_metrics() -> dict:
    """Read the metrics JSON written by train_model.py."""
    p = Path(METRICS_PATH)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}


ACCEPTANCE_THRESHOLD = 0.5
SHOW_FEATURE_IMPORTANCE = True

SAMPLE_PROFILES = {
    "Conservative Profile": {
        "Age": 35, "Experience": 10, "Income": 50, "Family": 2, "CCAvg": 1.0,
        "Education": 1, "Mortgage": 0,
        "Securities Account": 0, "CD Account": 0, "Online": 0, "CreditCard": 0,
    },
    "Moderate Profile": {
        "Age": 45, "Experience": 20, "Income": 100, "Family": 3, "CCAvg": 3.0,
        "Education": 2, "Mortgage": 100,
        "Securities Account": 0, "CD Account": 0, "Online": 1, "CreditCard": 1,
    },
    "Premium Profile": {
        "Age": 55, "Experience": 30, "Income": 180, "Family": 4, "CCAvg": 8.0,
        "Education": 3, "Mortgage": 300,
        "Securities Account": 1, "CD Account": 1, "Online": 1, "CreditCard": 1,
    },
}

"""Utility helpers for the loan-prediction Streamlit app."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)


def _is_int(x: Any) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


class ModelLoader:
    """Load the trained model + companion metrics from disk."""

    def __init__(self, model_path: str, metrics_path: Optional[str] = None):
        self.model_path = Path(model_path)
        self.metrics_path = Path(metrics_path) if metrics_path else self.model_path.with_name("model_metrics.json")
        self.model = None
        self.metrics: Dict[str, Any] = {}

    def load(self) -> bool:
        if not self.model_path.exists():
            return False
        with self.model_path.open("rb") as f:
            self.model = pickle.load(f)
        if self.metrics_path.exists():
            try:
                self.metrics = json.loads(self.metrics_path.read_text())
            except json.JSONDecodeError:
                self.metrics = {}
        return True

    def is_loaded(self) -> bool:
        return self.model is not None

    def get_model(self):
        return self.model

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics


class PredictionEngine:
    """Make predictions while keeping feature order consistent with training."""

    def __init__(self, model, feature_order: List[str]):
        self.model = model
        self.feature_order = feature_order

    def prepare_features(self, features_dict: Dict[str, float]) -> pd.DataFrame:
        # DataFrame (not ndarray) so the model sees the same column names it
        # was fit with — silences sklearn's "valid feature names" warning.
        row = {f: [features_dict[f]] for f in self.feature_order}
        return pd.DataFrame(row, columns=self.feature_order)

    def predict(self, features_df: pd.DataFrame) -> Tuple[int, float]:
        prediction = int(self.model.predict(features_df)[0])
        if hasattr(self.model, "predict_proba"):
            classes = list(getattr(self.model, "classes_", [0, 1]))
            proba_row = self.model.predict_proba(features_df)[0]
            # Probability of the POSITIVE class (loan accepted), not max
            pos_idx = classes.index(1) if 1 in classes else len(classes) - 1
            probability = float(proba_row[pos_idx])
        else:
            probability = 1.0 if prediction == 1 else 0.0
        return prediction, probability

    def get_feature_importance(self) -> Dict[str, float]:
        if hasattr(self.model, "feature_importances_"):
            return dict(zip(self.feature_order, self.model.feature_importances_))
        return {}


class DataValidator:
    """Lightweight range checks for the form inputs."""

    @staticmethod
    def _check(name: str, value: Any, lo: float, hi: float, integer: bool = False) -> Tuple[bool, str]:
        if integer:
            if not _is_int(value):
                return False, f"{name} must be a whole number"
        else:
            if not _is_number(value):
                return False, f"{name} must be a number"
        if value < lo or value > hi:
            unit = "" if integer else ""
            return False, f"{name} must be between {lo} and {hi}{unit}"
        return True, ""

    @staticmethod
    def validate_all(features_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
        rules = [
            ("Age",          features_dict.get("Age"),         18,  100, True),
            ("Experience",   features_dict.get("Experience"),   0,   50, True),
            ("Income",       features_dict.get("Income"),       0,  500, False),
            ("Family",       features_dict.get("Family"),       1,   10, True),
            ("CCAvg",        features_dict.get("CCAvg"),        0,   20, False),
            ("Education",    features_dict.get("Education"),    1,    3, True),
            ("Mortgage",     features_dict.get("Mortgage"),     0, 1000, False),
        ]
        errors: List[str] = []
        for name, value, lo, hi, is_int in rules:
            ok, msg = DataValidator._check(name, value, lo, hi, is_int)
            if not ok:
                errors.append(msg)

        # Cross-field check: experience cannot exceed (age - 16)
        age = features_dict.get("Age")
        exp = features_dict.get("Experience")
        if _is_number(age) and _is_number(exp) and exp > age - 16:
            errors.append("Experience cannot exceed Age - 16")

        return len(errors) == 0, errors


class ResultFormatter:
    @staticmethod
    def format_probability(probability: float, decimals: int = 2) -> str:
        return f"{probability * 100:.{decimals}f}%"

    @staticmethod
    def get_prediction_text(prediction: int) -> Tuple[str, str]:
        if prediction == 1:
            return "LOAN LIKELY TO BE ACCEPTED", "#00D084"
        return "LOAN LIKELY TO BE REJECTED", "#FF5252"


class RiskAnalysis:
    """Heuristic positive / risk factor breakdown for the result page."""

    @staticmethod
    def analyze_risk_factors(features_dict: Dict[str, float]) -> Dict[str, Any]:
        positives: List[str] = []
        risks: List[str] = []
        score = 0

        income = features_dict["Income"]
        if income > 100:
            positives.append("High income (>$100K)")
            score += 30
        elif income > 50:
            positives.append("Moderate income ($50K-$100K)")
            score += 15
        else:
            risks.append("Low income (<$50K)")
            score -= 10

        ccavg = features_dict["CCAvg"]
        if ccavg > 5:
            positives.append("High credit card usage")
            score += 20
        elif ccavg > 2:
            positives.append("Moderate credit card usage")
            score += 10
        else:
            risks.append("Low credit card activity")
            score -= 5

        exp = features_dict["Experience"]
        if exp > 15:
            positives.append("Significant professional experience")
            score += 15
        elif exp < 5:
            risks.append("Limited professional experience")
            score -= 10

        if features_dict.get("Mortgage", 0) > 100:
            positives.append("Significant financial commitment (mortgage)")
            score += 10

        if features_dict.get("CD Account") == 1:
            positives.append("Has CD account (engaged customer)")
            score += 10
        if features_dict.get("Securities Account") == 1:
            positives.append("Has securities account")
            score += 5

        return {"positive_factors": positives, "risk_factors": risks, "score": score}


def create_summary_stats(features_dict: Dict[str, float]) -> Dict[str, str]:
    return {
        "Annual Income": f"${features_dict['Income']:.0f}K",
        "Credit Card Avg": f"${features_dict['CCAvg']:.2f}K/month",
        "Mortgage": f"${features_dict['Mortgage']:.0f}K",
        "Professional Experience": f"{int(features_dict['Experience'])} years",
        "Family Size": f"{int(features_dict['Family'])} members",
        "Age": f"{int(features_dict['Age'])} years",
    }

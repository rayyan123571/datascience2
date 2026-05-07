# Machine Learning Pipeline Diagram

## Complete ML Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA ACQUISITION PHASE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Bank Personal Loan Modelling Dataset                                  │
│  ├─ Format: CSV/Excel                                                   │
│  ├─ Size: 5,001 records                                                 │
│  ├─ Features: 14                                                        │
│  └─ Target: Personal Loan (Binary: 0/1)                                 │
│                                                                          │
└──────────────────────────┬──────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    EXPLORATORY DATA ANALYSIS (EDA)                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. Data Inspection                                                     │
│     ├─ Shape & Info: 5001 × 14                                          │
│     ├─ Data Types: Integer, Float                                       │
│     ├─ Missing Values: 0                                                │
│     └─ Duplicates: 0                                                    │
│                                                                          │
│  2. Statistical Analysis                                                │
│     ├─ Descriptive statistics (mean, std, min, max)                     │
│     ├─ Target distribution: 9.2% (loan), 90.8% (no loan)               │
│     └─ Feature correlations                                             │
│                                                                          │
│  3. Visualizations (20+ graphs)                                         │
│     ├─ Histograms (5 features)                                          │
│     ├─ Boxplots (5 features)                                            │
│     ├─ Feature vs Target (5 boxplots)                                   │
│     ├─ Correlation Heatmap                                              │
│     ├─ Validation Curves                                                │
│     ├─ Confusion Matrices (3)                                           │
│     ├─ Feature Importance (2)                                           │
│     └─ Decision Tree Visualization                                      │
│                                                                          │
│  4. Outlier Detection                                                   │
│     ├─ Method: IQR (Interquartile Range)                               │
│     ├─ Outliers identified per feature                                  │
│     └─ Impact analysis: with/without outliers                           │
│                                                                          │
└──────────────────────────┬──────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    DATA PREPROCESSING & CLEANING                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. Data Type Conversion                                                │
│     └─ Ensure numeric fields are properly formatted                     │
│                                                                          │
│  2. Handling Missing Values                                             │
│     └─ Verified: No missing values detected                             │
│                                                                          │
│  3. Duplicate Removal                                                   │
│     └─ Verified: No duplicates found                                    │
│                                                                          │
│  4. Feature-Target Separation                                           │
│     ├─ X: All features except ID and target                             │
│     ├─ y: Personal Loan (target variable)                               │
│     └─ Dimensions: X (5001 × 12), y (5001,)                             │
│                                                                          │
│  5. Initial Standardization                                             │
│     └─ StandardScaler: Transform features to mean=0, std=1              │
│                                                                          │
└──────────────────────────┬──────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────────┐
│              FEATURE SELECTION & DIMENSIONALITY REDUCTION               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Method 1: SelectKBest (Statistical)                                    │
│  ├─ Algorithm: f_classif (ANOVA F-statistic)                           │
│  ├─ Configuration: k=5 features                                         │
│  ├─ Selection: Top 5 statistically significant features                 │
│  └─ Performance: CV Score = 94.23%                                      │
│                                                                          │
│  Method 2: Variance Threshold (Variance-based)                          │
│  ├─ Algorithm: Remove low-variance features                             │
│  ├─ Configuration: threshold=0.1                                        │
│  ├─ Selection: Features with variance > 0.1                             │
│  └─ Performance: CV Score = 95.14% ✓ BEST FOR THIS PROJECT             │
│                                                                          │
│  Method 3: Sequential Feature Selection (Greedy)                        │
│  ├─ Algorithm: Backward elimination with CV                             │
│  ├─ Configuration: k=5, CV=5-fold, Logistic Regression                 │
│  ├─ Selection: 5 features via iterative removal                         │
│  └─ Performance: CV Score = 93.67%                                      │
│                                                                          │
│  ✓ SELECTED: Variance Threshold (Best performer)                        │
│                                                                          │
└──────────────────────────┬──────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    TRAIN-TEST DATA SPLITTING                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Configuration:                                                         │
│  ├─ Test Size: 30%                                                      │
│  ├─ Train Size: 70%                                                     │
│  ├─ Stratification: Yes (maintains class distribution)                  │
│  ├─ Random State: 42 (reproducibility)                                  │
│  └─ Method: Stratified K-Fold                                           │
│                                                                          │
│  Data Splits:                                                           │
│  ├─ X_train: (3500 × 12) - Training features                            │
│  ├─ X_test: (1500 × 12) - Testing features                              │
│  ├─ y_train: (3500,) - Training labels                                  │
│  └─ y_test: (1500,) - Testing labels                                    │
│                                                                          │
│  Final Preprocessing:                                                   │
│  └─ StandardScaler: Fit on training, transform both sets                │
│                                                                          │
└──────────────────────────┬──────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────────┐
│              MODEL TRAINING & HYPERPARAMETER TUNING                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  MODEL 1: LOGISTIC REGRESSION                                           │
│  ├─ Algorithm: Logistic Regression (Linear Classification)              │
│  ├─ Hyperparameter Tuning: GridSearchCV                                 │
│  ├─ Parameter Grid:                                                     │
│  │  ├─ C: [0.01, 0.1, 1, 10, 100]                                       │
│  │  ├─ penalty: ['l1', 'l2']                                            │
│  │  └─ solver: ['liblinear']                                            │
│  ├─ CV Strategy: 5-fold cross-validation                                │
│  └─ Training Data: X_train_scaled, y_train                              │
│                                                                          │
│  MODEL 2: DECISION TREE ⭐ BEST PERFORMER                               │
│  ├─ Algorithm: Decision Tree Classifier                                 │
│  ├─ Hyperparameter Tuning: RandomizedSearchCV                           │
│  ├─ Parameter Grid:                                                     │
│  │  ├─ criterion: ['gini', 'entropy']                                   │
│  │  ├─ max_depth: [1-15]                                                │
│  │  ├─ min_samples_split: [2, 5, 10]                                    │
│  │  └─ min_samples_leaf: [1, 2, 4]                                      │
│  ├─ CV Strategy: 5-fold, 50 iterations                                  │
│  └─ Training Data: X_train, y_train (No scaling needed)                 │
│                                                                          │
│  MODEL 3: RANDOM FOREST                                                 │
│  ├─ Algorithm: Random Forest Classifier (Ensemble)                      │
│  ├─ Hyperparameter Tuning: RandomizedSearchCV                           │
│  ├─ Parameter Grid:                                                     │
│  │  ├─ n_estimators: [100, 200, 300]                                    │
│  │  ├─ max_depth: [None, 5, 10, 15]                                     │
│  │  ├─ min_samples_split: [2, 5, 10]                                    │
│  │  ├─ min_samples_leaf: [1, 2, 4]                                      │
│  │  └─ max_features: ['sqrt', 'log2']                                   │
│  ├─ CV Strategy: 5-fold, 50 iterations                                  │
│  └─ Training Data: X_train, y_train                                     │
│                                                                          │
└──────────────────────────┬──────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      MODEL EVALUATION & VALIDATION                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  METRICS CALCULATED (Per Model):                                        │
│  ├─ Accuracy: (TP + TN) / (TP + TN + FP + FN)                          │
│  ├─ Precision: TP / (TP + FP)                                           │
│  ├─ Recall: TP / (TP + FN)                                              │
│  ├─ F1-Score: 2 × (Precision × Recall) / (Precision + Recall)          │
│  └─ Cross-Validation Scores: 5-fold mean ± std                         │
│                                                                          │
│  RESULTS SUMMARY:                                                       │
│                                                                          │
│  ┌─ LOGISTIC REGRESSION                                                │
│  │  Accuracy:  95.13%                                                   │
│  │  Precision: 81.42%                                                   │
│  │  Recall:    63.89%                                                   │
│  │  F1-Score:  71.60%                                                   │
│  │  CV Score:  95.20% ± 0.86%                                           │
│  │                                                                       │
│  ├─ DECISION TREE ⭐                                                    │
│  │  Accuracy:  98.87% 🏆 BEST                                           │
│  │  Precision: 93.20%                                                   │
│  │  Recall:    95.14%                                                   │
│  │  F1-Score:  94.16%                                                   │
│  │  CV Score:  98.73% ± 0.98%                                           │
│  │                                                                       │
│  └─ RANDOM FOREST                                                      │
│     Accuracy:  98.87%                                                   │
│     Precision: 97.74%                                                   │
│     Recall:    90.28%                                                   │
│     F1-Score:  93.86%                                                   │
│     CV Score:  98.47% ± 0.45%                                           │
│                                                                          │
│  VISUALIZATIONS GENERATED:                                              │
│  ├─ Confusion Matrices (3) - Classification breakdown                   │
│  ├─ Feature Importance (2) - Top predictive features                    │
│  ├─ Validation Curves - Decision Tree performance vs depth              │
│  └─ Classification Reports - Per-class metrics                          │
│                                                                          │
└──────────────────────────┬──────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    MODEL COMPARISON & SELECTION                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  COMPARISON TABLE:                                                      │
│  ┌────────────────┬──────────┬───────────┬────────┬──────────┐          │
│  │ Model          │ Accuracy │ Precision │ Recall │ F1-Score │          │
│  ├────────────────┼──────────┼───────────┼────────┼──────────┤          │
│  │ Logistic Reg   │  95.13%  │  81.42%   │ 63.89% │  71.60%  │          │
│  │ Decision Tree⭐│  98.87%  │  93.20%   │ 95.14% │  94.16%  │          │
│  │ Random Forest  │  98.87%  │  97.74%   │ 90.28% │  93.86%  │          │
│  └────────────────┴──────────┴───────────┴────────┴──────────┘          │
│                                                                          │
│  BEST MODEL: DECISION TREE                                              │
│  ├─ Reason 1: Best precision-recall balance (F1: 94.16%)                │
│  ├─ Reason 2: Excellent overall accuracy (98.87%)                       │
│  ├─ Reason 3: High recall (95.14%) - catches most loan applicants      │
│  ├─ Reason 4: High precision (93.20%) - minimizes false positives       │
│  └─ Reason 5: High CV score with stable variance (98.73% ± 0.98%)      │
│                                                                          │
└──────────────────────────┬──────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         MODEL DEPLOYMENT PHASE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  STEP 1: MODEL SERIALIZATION                                            │
│  ├─ Best Model: Decision Tree                                           │
│  ├─ Format: Pickle (.pkl)                                               │
│  ├─ File: models/best_decision_tree_model.pkl                           │
│  ├─ Size: Compact for storage and loading                               │
│  └─ Method: joblib.dump()                                               │
│                                                                          │
│  STEP 2: FEATURE SCALER SERIALIZATION                                   │
│  ├─ Scaler: StandardScaler (fitted on training data)                    │
│  ├─ Format: Pickle (.pkl)                                               │
│  ├─ File: models/feature_scaler.pkl                                     │
│  ├─ Purpose: Transform new data consistently                            │
│  └─ Method: joblib.dump()                                               │
│                                                                          │
│  STEP 3: VALIDATION OF LOADED MODEL                                     │
│  ├─ Load both model and scaler from files                               │
│  ├─ Test on held-out test set                                           │
│  ├─ Verify accuracy matches training accuracy                           │
│  └─ Result: ✓ Validation passed (98.87% accuracy)                       │
│                                                                          │
│  STEP 4: PREDICTION PIPELINE                                            │
│  ├─ Input: New customer banking data                                    │
│  ├─ Preprocessing: Apply StandardScaler transform                       │
│  ├─ Prediction: Load model and predict                                  │
│  └─ Output: Loan acceptance probability (0 or 1)                        │
│                                                                          │
└──────────────────────────┬──────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION DEPLOYMENT & MONITORING                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  DEPLOYMENT CHECKLIST:                                                  │
│  ✅ Model accuracy verified (98.87%)                                    │
│  ✅ Cross-validation consistent (CV score within 1% of test)            │
│  ✅ Model serialized and loaded successfully                            │
│  ✅ Prediction pipeline functional                                      │
│  ✅ All dependencies documented (pandas, sklearn, joblib)               │
│  ✅ Code reproducible with fixed random_state                           │
│                                                                          │
│  MONITORING & MAINTENANCE:                                              │
│  ├─ Track model performance over time                                   │
│  ├─ Monitor for data drift in input features                            │
│  ├─ Retrain periodically with new data                                  │
│  ├─ Maintain version history of models                                  │
│  └─ Log predictions for audit and analysis                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Summary Statistics

| Stage | Data Size | Features | Time Complexity |
|-------|-----------|----------|-----------------|
| Raw Data | 5,001 × 14 | 14 | O(1) |
| After Selection | 5,001 × 12 | 12 | O(1) |
| Training Set | 3,500 × 12 | 12 | O(n) |
| Test Set | 1,500 × 12 | 12 | O(n) |

## Key Performance Indicators

- **Best Model Accuracy:** 98.87%
- **Training Time:** Minutes
- **Inference Time:** Milliseconds per prediction
- **Model Size:** ~100 KB (compressed)
- **Cross-Validation Score:** 98.73% ± 0.98%

## Technology Stack

```
Data Processing:
  └─ pandas, numpy

ML Framework:
  └─ scikit-learn

Feature Selection:
  └─ mlxtend, sklearn

Visualization:
  └─ matplotlib, seaborn

Model Serialization:
  └─ joblib
```

---

*Pipeline Overview Generated: May 6, 2026*

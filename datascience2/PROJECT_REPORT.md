# Project Report — Bank Personal Loan Prediction

A complete end-to-end Data Science Pipeline built for the semester project: data collection → preprocessing → EDA → visualisation → model building → evaluation → reporting → deployment as a web app.

---

## 1. Problem Statement

Banks running marketing campaigns for personal loans face a low conversion rate (~9.6%). Targeting every customer wastes budget. The task is to build a **binary classifier** that, given a customer's demographics, financial profile and banking-relationship features, predicts whether they will accept a personal-loan offer (`Personal Loan = 1`).

Success criteria:

| Goal | Target |
|------|--------|
| Accuracy on hold-out set | ≥ 95% |
| F1-score (positive class) | ≥ 0.85 |
| ROC-AUC | ≥ 0.95 |
| Reproducible pipeline (one command) | yes |
| Interactive web app for predictions | yes |

All four criteria are met (see §6).

---

## 2. Dataset

* **Source file**: `data/Bank_Personal_Loan_Modelling.csv`
* **Records**: 5,000 customers
* **Original columns**: 14
* **Missing values**: 0
* **Duplicate rows**: 0
* **Class balance**: 9.6% positive (imbalanced)

### 2.1 Columns

| Column | Type | Used? | Reason |
|--------|------|------|--------|
| ID | int | **dropped** | Identifier — non-predictive, would leak per-row identity. |
| Age | int | yes | Customer age in years. |
| Experience | int | yes | Years of professional experience. |
| Income | int | yes | Annual income in $1000s. |
| ZIP Code | int | **dropped** | Geographic noise; high cardinality, not predictive without engineering. |
| Family | int | yes | Family members (1–4). |
| CCAvg | float | yes | Average monthly credit-card spend in $1000s. |
| Education | int | yes | 1=Undergrad, 2=Graduate, 3=Advanced. |
| Mortgage | int | yes | Mortgage value in $1000s. |
| Securities Account | int | yes | Has a securities account (binary). |
| CD Account | int | yes | Has a certificate of deposit (binary). |
| Online | int | yes | Uses online banking (binary). |
| CreditCard | int | yes | Has a credit card with the bank (binary). |
| Personal Loan | int | **target** | Target — accepted the loan offer. |

11 features go into the model.

---

## 3. Pipeline Architecture

```
data/Bank_Personal_Loan_Modelling.csv
            │
            ▼
   ┌───────────────────┐         ┌─────────────────────────┐
   │   eda.py          │────────▶│ eda_images/*.png (12)   │
   │   (visual EDA)    │         │ reports/eda_summary.txt │
   └───────────────────┘         └─────────────────────────┘
            │
            ▼
   ┌───────────────────────────┐
   │   train_model.py          │
   │   • LogisticRegression    │   ┌──────────────────────────────────┐
   │   • DecisionTree          │──▶│ models/best_model.pkl            │
   │   • RandomForest          │   │ models/model_metrics.json        │
   │   • XGBoost (optional)    │   │ reports/plots/*.png              │
   │   pick best by F1         │   │ reports/model_comparison_*.txt   │
   └───────────────────────────┘   └──────────────────────────────────┘
            │
            ▼
   ┌───────────────────────────┐
   │   app_streamlit/app.py    │ ◄── interactive web app
   │   (Streamlit dashboard)   │      reads pkl + metrics.json
   └───────────────────────────┘
```

A single command runs everything:

```bash
python run_pipeline.py
```

---

## 4. Preprocessing

Handled inside [train_model.py](app_streamlit/train_model.py):

1. **Load** the CSV with pandas.
2. **Drop** `ID` and `ZIP Code` (see §2.1).
3. **Type-cast** target to `int`.
4. **Stratified split** 70/30 with `random_state=42` so positive-rate stays at ~9.6% in both sets.
5. **Scaling**: only Logistic Regression uses `StandardScaler`; the tree-based models do not need scaling.
6. **Class imbalance**: candidate models are searched both with and without `class_weight="balanced"` so the grid can pick the better setting.

---

## 5. Exploratory Data Analysis (Matplotlib + Seaborn)

`eda.py` produces 12 figures under `eda_images/` plus a one-page text summary in `reports/eda_summary.txt`.

| # | File | What it shows |
|---|------|---------------|
| 01 | `01_class_balance.png` | Bar chart of `Personal Loan` (~9.6% positive). |
| 02 | `02_numeric_distributions.png` | Histograms + KDE of 5 numeric features. |
| 03 | `03_numeric_boxplots.png` | Boxplots — outlier check via IQR. |
| 04 | `04_features_by_target.png` | Numeric features grouped by loan outcome. |
| 05 | `05_correlation_heatmap.png` | Pearson correlations across all numeric features. |
| 06 | `06_acceptance_by_categorical.png` | Acceptance rate broken down by Education / CD Account / etc. |
| 07 | `07_income_vs_ccavg.png` | Scatter of Income vs CCAvg, coloured by outcome. |
| 08 | `08_income_by_outcome.png` | Income distribution split by outcome. |
| 09 | `09_age_by_outcome.png` | Age distribution split by outcome. |
| 10 | `10_pairplot.png` | Pairplot of top numeric features. |
| 11 | `11_education_income_violin.png` | Violin plots of Income vs Education, split by outcome. |
| 12 | `12_family_stack.png` | Stacked-share bar chart of outcome by Family Size. |

### 5.1 Key findings

* **Income** is the strongest single signal — the median income of acceptors is roughly 2× that of non-acceptors.
* **CCAvg** has a near-linear relationship with acceptance rate; customers spending > $5K/month accept at >70%.
* **Education** matters: advanced-degree holders accept ~14%, undergrads ~4%.
* **CD Account holders** accept at ~46% vs ~7% for non-holders — a near-textbook example of "engaged customer".
* `Age` and `Experience` have correlation ≈ 1.0 — redundant by themselves, but cheap to keep.
* The dataset has *no* missing values and *no* duplicates, so no imputation is required.

---

## 6. Model Building & Evaluation

`train_model.py` trains four candidate models with cross-validated hyperparameter search and picks the best by F1 on the test set.

### 6.1 Search strategy

| Model | Search | CV folds | Notes |
|-------|--------|----------|-------|
| Logistic Regression | `GridSearchCV` over `C` | 5 (stratified) | Wrapped in `Pipeline` with `StandardScaler`. |
| Decision Tree | `GridSearchCV` over `criterion`, `max_depth`, `min_samples_*`, `class_weight` | 5 | 270 candidates. |
| Random Forest | `RandomizedSearchCV` over `n_estimators`, `max_depth`, `min_samples_*`, `max_features`, `class_weight` | 5 | 25 iterations. |
| XGBoost (optional) | `RandomizedSearchCV` | 5 | Skipped if package not installed. |

Scoring metric for selection: **F1** (because of the class imbalance — accuracy alone would be misleading).

### 6.2 Test-set metrics (this run)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV mean |
|-------|---------:|----------:|-------:|---:|--------:|--------:|
| Logistic Regression | 0.8953 | 0.4762 | 0.9028 | 0.6235 | 0.9614 | 0.8946 |
| Decision Tree | 0.9820 | 0.8926 | 0.9236 | 0.9078 | 0.9726 | 0.9817 |
| **Random Forest** ⭐ | **0.9887** | **0.9704** | 0.9097 | **0.9391** | **0.9983** | **0.9851** |

**Winner: Random Forest** — best F1 *and* best ROC-AUC, with very strong precision (97.04%) and minimal cross-validation variance.

### 6.3 Best hyperparameters

```
n_estimators       = 200
max_depth          = 12
max_features       = sqrt
min_samples_split  = 2
min_samples_leaf   = 1
class_weight       = None
```

### 6.4 Confusion matrix (Random Forest, test set)

|                    | Predicted: No | Predicted: Yes |
|--------------------|---------------:|---------------:|
| **Actual: No**     | 1352           | 4              |
| **Actual: Yes**    | 13             | 131            |

Only **17** wrong predictions out of 1,500 — and just **4** false positives (the costlier error from a marketing-budget perspective).

### 6.5 Feature importance (winner)

| Rank | Feature | Importance |
|------|---------|-----------:|
| 1 | Income | 0.352 |
| 2 | Education | 0.193 |
| 3 | CCAvg | 0.156 |
| 4 | Family | 0.108 |
| 5 | CD Account | 0.058 |
| 6 | Age | 0.039 |
| 7 | Experience | 0.038 |
| 8 | Mortgage | 0.035 |
| 9 | CreditCard | 0.008 |
| 10 | Online | 0.008 |
| 11 | Securities Account | 0.006 |

Top 3 features (Income + Education + CCAvg) account for **70%** of the model's decisions — consistent with the EDA findings.

### 6.6 Visual artefacts produced

* `reports/plots/model_comparison.png` — bar chart of all 5 metrics across all models
* `reports/plots/roc_curves.png` — ROC curve overlay
* `reports/plots/confusion_matrices.png` — heatmap CM per model

---

## 7. Web Application

A two-page Streamlit app under `app_streamlit/`:

| File | Purpose |
|------|---------|
| `app.py` | Interactive prediction form + model-info & analytics tabs. Loads `models/best_model.pkl` and reads metrics from `models/model_metrics.json` at runtime — **no hardcoded numbers**. |
| `dashboard_app.py` | Filterable BI dashboard over the dataset (income/age distributions, correlation heatmap, acceptance breakdowns). |

Run them with:

```bash
streamlit run app_streamlit/app.py
streamlit run app_streamlit/dashboard_app.py
```

The prediction app:

* Takes the same 11 features the model was trained on.
* Validates each field (range + cross-field rule `Experience ≤ Age − 16`).
* Shows the predicted class, calibrated probability `P(Accept)`, the model's feature-importance bar chart, and a heuristic "positive vs risk factors" panel.
* Includes three sample profiles (Conservative / Moderate / Premium) loadable from the sidebar.

A `Dockerfile` + `docker-compose.yml` are included for one-command deployment:

```bash
cd app_streamlit && docker compose up --build
```

---

## 8. Reproducibility

| Aspect | How it is enforced |
|--------|-------------------|
| Random seed | `random_state=42` in every split / search / model. |
| Single dependency file | `requirements.txt` at project root. |
| Single entry-point | `python run_pipeline.py` runs EDA + training. |
| No leakage | `ID` dropped before split; no test-set info touches training. |
| Stratified split | Preserves the 9.6% positive rate in both sets. |
| Persisted artefacts | Model + metrics + plots are written to disk so the report is self-contained. |

---

## 9. How to Run

```bash
# 0. clone / copy the project, then from the project root:
pip install -r requirements.txt

# 1. (optional but recommended) regenerate everything from scratch
python run_pipeline.py

# 2. launch the web app
streamlit run app_streamlit/app.py
```

If you only want to retrain:

```bash
python app_streamlit/train_model.py
```

If you only want EDA:

```bash
python app_streamlit/eda.py
```

---

## 10. Limitations & Future Work

* The dataset is small (5,000 rows) and synthetic — production deployment would need more data and drift monitoring.
* `ZIP Code` was dropped; with more time it could be feature-engineered (state, region, urban/rural).
* Add SHAP explanations to the Streamlit app (`shap_explainability.py` already exists as a CLI script).
* Add an MLflow run-tracker for hyperparameter sweeps.
* Bayesian optimisation (`Optuna`) over the candidate space.

---

## 11. Conclusion

The pipeline meets every requirement of the project brief:

| Brief item | Where |
|------------|-------|
| Data collection | `data/Bank_Personal_Loan_Modelling.csv` + `archive/*.xlsx` |
| Preprocessing | `app_streamlit/train_model.py` (drop, split, scale) |
| Visualisation (Matplotlib + Seaborn) | `app_streamlit/eda.py` → 12 PNGs |
| EDA | `eda.py` plots + `reports/eda_summary.txt` |
| Model building (multiple algos) | `train_model.py` (LR, DT, RF, optional XGB) |
| Evaluation metrics | Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, CV |
| Reporting | `PROJECT_REPORT.md` (this file) + `reports/*.txt` |
| Web app | `app_streamlit/app.py` (Streamlit) + Docker setup |

Final winning model — **Random Forest, 98.87% accuracy, 0.94 F1, 0.998 ROC-AUC** — comfortably exceeds the success criteria stated in §1.

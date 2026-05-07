# Bank Personal Loan Prediction

Machine-learning project that predicts whether a bank customer will accept a personal-loan offer. Includes a trained Decision Tree, hyperparameter-tuning and explainability scripts, a Streamlit web app, and a Docker setup.

---

## Quick start

```bash
# 1. Install dependencies (single requirements file)
pip install -r requirements.txt

# 2. Run the full pipeline (EDA + train all candidate models + pick best)
python run_pipeline.py

# 3. Launch the Streamlit web app
streamlit run app_streamlit/app.py
```

Open http://localhost:8501 once Streamlit prints its URL. See [PROJECT_REPORT.md](PROJECT_REPORT.md) for the full write-up.

---

## Problem statement

Predict the binary `Personal Loan` target — whether a customer will accept a personal-loan marketing offer — from demographic and financial features.

| Attribute | Value |
|-----------|-------|
| Records | 5,000 customers |
| Features used | 11 (ID and ZIP Code dropped) |
| Target | `Personal Loan` (0 = no, 1 = yes) |
| Class balance | 9.6% positive |
| Train / test split | 70 / 30 (stratified) |

---

## Features used by the model

| Feature | Type | Description |
|---|---|---|
| Age | numeric | Customer's age in years |
| Experience | numeric | Years of professional experience |
| Income | numeric | Annual income in $1000s |
| Family | ordinal | Number of family members (1–4) |
| CCAvg | numeric | Average monthly credit-card spend in $1000s |
| Education | ordinal | 1=Undergrad, 2=Graduate, 3=Advanced |
| Mortgage | numeric | Mortgage amount in $1000s |
| Securities Account | binary | Has investment securities account |
| CD Account | binary | Has certificate of deposit |
| Online | binary | Uses online banking |
| CreditCard | binary | Has a credit card with the bank |

`ID` and `ZIP Code` are intentionally excluded — they are not predictive and `ID` would leak per-row identity.

---

## Models

`train_model.py` trains **four** candidate algorithms (Logistic Regression, Decision Tree, Random Forest, and XGBoost if installed), each with cross-validated hyperparameter search, and persists the best one (chosen by F1 on the held-out test set).

Last run picked **Random Forest**: accuracy 98.87%, F1 0.94, ROC-AUC 0.998.

All metrics are saved to `app_streamlit/models/model_metrics.json` and shown live in the Streamlit app — no hardcoded numbers in the UI.

---

## Repository layout

```
datascience2/
├── README.md
├── PROJECT_REPORT.md                       # full report (problem → results)
├── requirements.txt                        # single source of truth
├── .gitignore
├── run_pipeline.py                         # one-command runner: EDA + train
├── personal-loan-prediction.ipynb          # original exploratory notebook
├── data/
│   └── Bank_Personal_Loan_Modelling.csv
├── archive/
│   └── Bank_Personal_Loan_Modelling.xlsx   # original Excel source
├── eda_images/                             # 12 PNGs produced by eda.py
├── reports/
│   ├── PIPELINE_DIAGRAM.md
│   ├── eda_summary.txt
│   ├── model_comparison_summary.txt
│   └── plots/
│       ├── model_comparison.png
│       ├── roc_curves.png
│       └── confusion_matrices.png
└── app_streamlit/
    ├── app.py                  # Streamlit prediction UI
    ├── dashboard_app.py        # Streamlit BI dashboard (separate app)
    ├── eda.py                  # generates eda_images/*.png + summary
    ├── train_model.py          # trains LR + DT + RF + XGBoost, picks best
    ├── config.py               # constants, feature order, paths
    ├── utils.py                # ModelLoader, PredictionEngine, validators
    ├── advanced_models.py      # extra XGBoost / LightGBM CLI (optional)
    ├── model_tuning.py         # GridSearchCV CLI on 3 baseline models
    ├── shap_explainability.py  # SHAP plots CLI (optional)
    ├── Dockerfile
    ├── docker-compose.yml
    └── models/
        ├── best_model.pkl
        ├── best_decision_tree_model.pkl    # alias of best_model.pkl
        └── model_metrics.json
```

---

## Running the BI dashboard

```bash
cd app_streamlit
streamlit run dashboard_app.py
```

Filterable charts over the dataset (income/age distributions, correlation heatmap, acceptance by education / family size).

---

## Running with Docker

```bash
cd app_streamlit
docker compose up --build
```

The compose file uses the project root as build context so that `requirements.txt` and `data/` are visible to the image.

---

## Optional: advanced models and SHAP

```bash
cd app_streamlit
python advanced_models.py --data ../data/Bank_Personal_Loan_Modelling.csv
python shap_explainability.py --data ../data/Bank_Personal_Loan_Modelling.csv
```

These produce comparison plots and SHAP explanation plots under `results/`.

---

## License

Educational and research use.

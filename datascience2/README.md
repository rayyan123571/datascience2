# Bank Personal Loan Prediction

Production-ready machine learning project for predicting whether a customer will accept a personal loan offer. The repository includes model training, evaluation, explainability, and an interactive Streamlit application.

## Highlights

- End-to-end ML pipeline: EDA, preprocessing, model training, evaluation, and artifact persistence.
- Multiple candidate models with model selection based on robust evaluation metrics.
- Interactive Streamlit UI for prediction, model comparison, analytics, and business insights.
- Docker support for reproducible local deployment.

## Project Overview

This project solves a binary classification problem:

- Target: `Personal Loan` (0 = not accepted, 1 = accepted)
- Dataset size: 5,000 customer records
- Feature count: 11 model input features
- Typical split: stratified train/test split for class balance preservation

The objective is to help a bank prioritize high-probability customers for personal loan campaigns.

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost / LightGBM (optional, if installed)
- SHAP (explainability)
- Streamlit + Plotly

## Repository Structure

```text
datascience2/
├── README.md
├── PROJECT_REPORT.md
├── requirements.txt
├── run_pipeline.py
├── personal-loan-prediction.ipynb
├── data/
│   └── Bank_Personal_Loan_Modelling.csv
├── reports/
│   ├── eda_summary.txt
│   ├── model_comparison_summary.txt
│   └── plots/
└── app_streamlit/
    ├── app.py
    ├── dashboard_app.py
    ├── config.py
    ├── utils.py
    ├── train_model.py
    ├── model_tuning.py
    ├── advanced_models.py
    ├── shap_explainability.py
    └── models/
        └── model_metrics.json
```

## Installation

Use a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

Run the full pipeline:

```bash
python run_pipeline.py
```

Launch the Streamlit app:

```bash
python -m streamlit run app_streamlit/app.py
```

Open `http://localhost:8501` in your browser.

## Streamlit App Modules

The main app includes:

- Prediction workflow for customer-level scoring
- Model comparison dashboard
- Model details and feature-importance view
- Data analytics page
- Insights and recommendations page

## Model Artifacts

Trained model metadata and evaluation summaries are saved under:

- `app_streamlit/models/model_metrics.json`

Depending on training flow, serialized model files are also stored in `app_streamlit/models/`.

## Docker (Optional)

From the `app_streamlit/` directory:

```bash
docker compose up --build
```

## Report

Detailed methodology, experiments, and outcomes are documented in:

- `PROJECT_REPORT.md`

## License

This project is intended for educational and research use.

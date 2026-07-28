# ChurnPulse — Customer Churn Prediction

ChurnPulse is an end-to-end customer-retention decision-support application built with the IBM Telco Customer Churn dataset. It combines a reproducible machine-learning workflow with an interactive Streamlit experience for understanding churn risk, prioritising a customer portfolio, and reviewing the drivers behind an individual prediction.

> This is a portfolio and decision-support project. Predictions should inform retention work, not make fully automated customer decisions.

## Highlights

- **Executive dashboard** — model performance, data-quality indicators, customer-risk portfolio, and platform-health status in one view.
- **Individual churn assessment** — enter a customer profile to receive a predicted churn probability, class, and confidence.
- **Feature-level explanations** — shows the factors that increase and reduce the risk for the submitted customer, along with business-focused recommendations.
- **Reliable explanation fallback** — uses SHAP when it is available; otherwise, the app performs scenario analysis against a typical customer profile so explanations remain available.
- **Portfolio risk summary** — calculates high- and low-risk customer counts from the production model’s row-level predictions.
- **Export-ready outputs** — download individual predictions as CSV and export supported analysis tables to Excel (via `openpyxl`).
- **Reproducible ML workflow** — cleaning, feature engineering, imbalance handling with SMOTE, model evaluation, threshold analysis, and persisted model artifacts.

## Current model snapshot

The shipped model artifact is a **Gradient Boosting** classifier trained with **SMOTE**. Its recorded validation results are:

| Metric | Score |
| --- | ---: |
| Accuracy | 79.43% |
| ROC-AUC | 0.8390 |
| PR-AUC | 0.6472 |
| Precision | 60.20% |
| Recall | 65.86% |
| F1 score | 0.6290 |

Metrics are specific to the included validation run; retraining can produce different results.

## Run locally

The Docker image and continuous-integration workflow use Python 3.12. Start from the repository root.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Open the local address printed by Streamlit (normally `http://localhost:8501`).

### Quick verification

1. Open **Overview** and confirm that Customer Portfolio shows high- and low-risk counts.
2. Open **Churn assessment**, submit a customer profile, and confirm the feature-level drivers and recommendations appear below the prediction metrics.
3. Use **Download Prediction** to verify the CSV export.

## Explanation methods

ChurnPulse uses the strongest explanation method available in the current environment:

| Method | When it is used | What it shows |
| --- | --- | --- |
| SHAP attribution | The `shap` package is installed and initialises successfully | Per-feature model contributions in the transformed model space. |
| Scenario analysis | SHAP is unavailable | The change in predicted churn risk when each submitted feature replaces its typical reference value. |

Scenario analysis is clearly labelled in the interface. It is a local, comparative explanation—not a causal claim.

## Train a model

The training CLI is `app.py`.

```powershell
python app.py --help
python app.py --data data\raw\telco_customer_churn.csv
```

Training outputs are written to `models/` and `reports/`, including the selected model, evaluation metrics, threshold analysis, and generated reports.

## Test and quality checks

Install developer tools and run the test suite:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Useful checks:

```powershell
python -m ruff check app src configs tests
python -m black --check app src configs tests
```

GitHub Actions runs tests with coverage on Python 3.12 for pushes and pull requests to the configured branches.

## Docker

```powershell
docker build -t churnpulse .
docker run --rm -p 8501:8501 churnpulse
```

Then open `http://localhost:8501`.

## Project structure

```text
app/                 Streamlit views, UI components, and application services
src/                 Data, preprocessing, models, evaluation, explainability, thresholds
configs/             Centralised settings, paths, and logging configuration
data/raw/            IBM Telco source dataset
models/              Saved production model and threshold artifacts
reports/             Metrics, reports, and model-comparison outputs
tests/               Automated unit and service tests
.github/workflows/   Continuous-integration workflow
streamlit_app.py     Streamlit application entry point
app.py                Training CLI entry point
```

## Technology stack

Python, Streamlit, pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost, SHAP, Plotly, Matplotlib, seaborn, openpyxl, pytest, Docker, and GitHub Actions.

## Scope

The repository includes the source data and model artifacts needed to run the demonstration locally. Authentication, external model storage, production monitoring, and managed databases are intentionally outside the scope of this project.

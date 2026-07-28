"""About page.

Static, descriptive content about the project — no dynamic ML logic.
"""

import streamlit as st

from configs.config import settings


def render_about_page() -> None:
    """Render the About page."""
    st.header(f"ℹ️ About {settings.project_name}")

    st.subheader("Project Overview")
    st.write(
        "This application predicts customer churn for a telecommunications provider and "
        "translates each prediction into an explainable, business-ready recommendation. It "
        "is the presentation layer for an end-to-end machine learning system: data "
        "ingestion and validation, exploratory analysis, feature engineering, class-imbalance "
        "handling, model training with hyperparameter optimization, evaluation, SHAP "
        "explainability, and business-cost-driven threshold optimization."
    )

    st.subheader("Machine Learning Pipeline")
    pipeline_stages = (
        "Data ingestion & validation",
        "Exploratory data analysis & business insights",
        "Feature engineering & preprocessing",
        "Class imbalance handling & sampling experiments",
        "Model training, hyperparameter optimization & experiment tracking",
        "Model evaluation & performance visualization",
        "SHAP explainability & feature attribution",
        "Threshold optimization & business decision engine",
        "This Streamlit application",
    )
    st.markdown("\n".join(f"{i}. {stage}" for i, stage in enumerate(pipeline_stages, start=1)))

    st.subheader("Technologies")
    st.write(
        "Python 3.12, pandas, scikit-learn, XGBoost, imbalanced-learn, SHAP, Matplotlib/"
        "Seaborn, and Streamlit."
    )

    st.subheader("Dataset")
    st.write(
        "The IBM/Kaggle Telco Customer Churn dataset: one row per customer, with "
        "demographic, account, and subscribed-service attributes, and a binary `Churn` label."
    )

    st.subheader("Repository Structure")
    st.code(
        "src/            # All ML logic (data, analysis, preprocessing, sampling,\n"
        "                #  models, evaluation, explainability, threshold)\n"
        "configs/        # Centralized configuration, paths, logging\n"
        "app/            # This Streamlit application (presentation layer only)\n"
        "  pages/        # One module per page\n"
        "  components/   # Reusable UI widgets\n"
        "  services/     # Bridges pages to src/ (no ML logic of its own)\n"
        "  utils/        # Session state, caching, input validation\n"
        "reports/        # Generated reports and figures\n"
        "models/         # Serialized model, preprocessor, and threshold config\n",
        language="text",
    )

    st.subheader("Future Improvements")
    st.markdown(
        "- Automated retraining pipeline triggered by data drift detection\n"
        "- A/B testing framework for comparing retention strategies\n"
        "- Real-time prediction API alongside this batch/interactive UI\n"
        "- Expanded model registry (LightGBM, CatBoost, ensembling)\n"
    )

    st.subheader("Author")
    st.write(
        "Built as a portfolio project demonstrating production-grade ML engineering practices."
    )

    st.caption(f"{settings.project_name} · v{settings.version}")

"""
===============================================================================
Customer Churn Prediction Platform
Reusable Forms
===============================================================================

Shared Streamlit forms used throughout the application.

Responsibilities
----------------
• Collect user input
• Validate basic form data
• Standardize form layout

No preprocessing or ML logic belongs here.
===============================================================================
"""

from __future__ import annotations

from typing import Any

import streamlit as st

DEFAULT_FORM_BORDER = True

def form_submit(
    label: str = "Submit",
) -> bool:
    """
    Render the standard submit button.
    """

    return st.form_submit_button(
        label,
        width="stretch",
    )


def reset_form(
    keys: list[str],
) -> None:
    """
    Reset selected form fields.
    """

    for key in keys:

        st.session_state.pop(
            key,
            None,
        )
        

def render_prediction_form() -> tuple[dict[str, Any] | None, bool]:
    """Collect the IBM Telco features used by the fitted pipeline."""

    st.markdown("### Build a customer profile")
    st.caption("A few inputs go a long way—add account, service and billing details to reveal the retention signal.")
    st.progress(0.33, text="Profile checklist  •  Account  →  Services  →  Billing")

    with st.form("prediction_form", border=True):
        st.markdown("#### 01 — Customer & account")
        left, right = st.columns(2, gap="large")

        with left:
            gender = st.selectbox("Gender", ["Female", "Male"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)

        with right:
            senior = st.selectbox(
                "Senior citizen",
                [0, 1],
                format_func=lambda value: "Yes" if value else "No",
            )
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
            paperless = st.selectbox("Paperless billing", ["No", "Yes"])
            payment = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        st.markdown("#### 02 — Services")
        service_left, service_right = st.columns(2, gap="large")
        internet_options = ["No", "Yes", "No internet service"]

        with service_left:
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multiple = st.selectbox(
                "Multiple lines", ["No", "Yes", "No phone service"]
            )
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online security", internet_options)
            online_backup = st.selectbox("Online backup", internet_options)

        with service_right:
            device_protection = st.selectbox("Device protection", internet_options)
            tech_support = st.selectbox("Tech support", internet_options)
            streaming_tv = st.selectbox("Streaming TV", internet_options)
            streaming_movies = st.selectbox("Streaming movies", internet_options)

        st.markdown("#### 03 — Billing")
        billing_left, billing_right = st.columns(2, gap="large")
        with billing_left:
            monthly = st.number_input(
                "Monthly charges", min_value=0.0, max_value=200.0, value=70.0, step=1.0
            )
        with billing_right:
            total = st.number_input(
                "Total charges", min_value=0.0, max_value=10000.0, value=1500.0, step=10.0
            )

        submitted = st.form_submit_button(
            "Reveal retention signal  ✦", type="primary", width="stretch"
        )

    if not submitted:
        return None, False

    customer_data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }
    return customer_data, True

def render_batch_prediction_form():
    """
    Upload CSV for batch prediction.
    """

    with st.form(
        "batch_prediction_form",
        border=True,
    ):

        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=["csv"],
        )

        submitted = form_submit(
            "Run Batch Prediction"
        )

    if submitted:

        return uploaded_file

    return None

def render_model_selector(
    models: list[str],
) -> str | None:
    """
    Render model selector.
    """

    with st.form(
        "model_selector",
        border=True,
    ):

        selected = st.selectbox(
            "Model",
            models,
        )

        submitted = form_submit(
            "Load Model"
        )

    if submitted:

        return selected

    return None

def render_export_form():
    """
    Export options.
    """

    with st.form(
        "export_form",
        border=True,
    ):

        export_format = st.selectbox(
            "Format",
            [
                "CSV",
                "Excel",
                "JSON",
            ],
        )

        submitted = form_submit(
            "Export"
        )

    if submitted:

        return export_format

    return None

__all__ = [
    "form_submit",
    "reset_form",
    "render_prediction_form",
    "render_batch_prediction_form",
    "render_model_selector",
    "render_export_form",
]

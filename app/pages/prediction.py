"""
===============================================================================
Customer Churn Prediction Platform
Prediction Page
===============================================================================

Purpose
-------
Single-customer churn prediction interface.

Responsibilities
----------------
• Collect customer information
• Validate user input
• Load cached production resources
• Generate churn prediction
• Display business recommendations
• Generate SHAP explanations
• Export prediction results

Architecture
------------
UI Layer Only

This module contains no machine learning logic.
All prediction, recommendation, explanation, and export functionality is
delegated to the service layer.

Workflow
--------
Customer Form
        │
        ▼
Validation
        │
        ▼
Prediction Service
        │
        ▼
Recommendation Service
        │
        ▼
Explanation Service
        │
        ▼
Export Service
===============================================================================
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app.services.prediction_service import PredictionResult
from app.components.cards import (
    render_business_insight_card,
    render_prediction_card,
)
from app.components.forms import (
    render_prediction_form as render_customer_form,
)

from app.services.explanation_service import (
    explain_customer_prediction,
    explain_customer_prediction_with_scenarios,
)
from app.services.prediction_service import (
    predict_customer,
)
from app.services.recommendation_service import (
    generate_recommendation,
)
from app.services.export_service import (
    prepare_prediction_export,
)

from app.utils.cache import (
    get_cached_explainer,
    get_cached_background_sample,
    get_cached_model,
    get_cached_threshold_config,
)

from app.utils.session import (
    add_prediction_to_history,
    set_selected_customer,
)

from app.utils.validators import (
    validate_customer_input,
)

from configs.logging_config import (
    get_logger,
)

logger = get_logger(__name__)

PAGE_TITLE = "Customer Churn Prediction"

PAGE_DESCRIPTION = (
    "Predict the likelihood that an individual customer will churn "
    "using the trained production machine learning model."
)

EXPLANATION_TITLE = "Why did the model make this prediction?"

EXPORT_TITLE = "Export Prediction"

def render_page_header() -> None:
    """
    Render the page heading.
    """

    st.markdown(
        f"""
        <section class="page-hero compact">
            <div class="eyebrow">INDIVIDUAL RISK ASSESSMENT</div>
            <h1>{PAGE_TITLE}</h1>
            <p>{PAGE_DESCRIPTION}</p>
            <div class="hero-status"><span></span> Ready when you are</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

def validate_submission(
    customer_data: dict[str, Any],
) -> bool:
    """
    Validate submitted customer data.

    Parameters
    ----------
    customer_data
        Customer attributes submitted through the form.

    Returns
    -------
    bool
        True if validation succeeds.
    """

    validation_errors = validate_customer_input(
        customer_data,
    )

    if not validation_errors:
        return True

    for error in validation_errors:
        st.error(error)

    return False

def load_prediction_resources() -> tuple[Any, Any, Any]:
    """
    Load all cached production resources.

    Returns
    -------
    tuple
        (
            trained_model,
            threshold_configuration,
            shap_explainer,
        )
    """

    logger.info(
        "Loading cached prediction resources."
    )

    model = get_cached_model()

    threshold_configuration = (
        get_cached_threshold_config()
    )

    try:
        explainer = get_cached_explainer(
            model,
            background_key="default",
        )
    except Exception as exception:
        logger.warning("Explainability resources are unavailable: %s", exception)
        explainer = None

    return (
        model,
        threshold_configuration,
        explainer,
    )

def generate_prediction(
    customer_data: dict[str, Any],
):
    """
    Execute the prediction pipeline.

    Returns
    -------
    tuple
        (
            prediction,
            recommendation,
            model,
            explainer,
        )
    """

    (
        model,
        threshold_configuration,
        explainer,
    ) = load_prediction_resources()

    prediction = predict_customer(
        customer_data,
        model,
    )

    recommendation = generate_recommendation(
        prediction.predicted_probability,
        threshold_configuration,
    )

    return (
        prediction,
        recommendation,
        model,
        explainer,
    )

def persist_prediction(
    customer_data: dict[str, Any],
    recommendation: dict[str, Any],
) -> None:
    """
    Store prediction in session state.
    """

    add_prediction_to_history(
        recommendation,
    )

    set_selected_customer(
        customer_data,
    )

def handle_prediction_exception(
    exception: Exception,
) -> None:
    """
    Display a friendly error message while logging
    the underlying exception.
    """

    logger.exception(
        "Prediction workflow failed.",
        exc_info=exception,
    )

    st.error(
        "Unable to generate a prediction. "
        "Please try again."
    )

    with st.expander(
        "Technical Details",
        expanded=False,
    ):
        st.code(str(exception))

def render_prediction_result(
    recommendation: dict[str, Any],
) -> None:
    """
    Display the prediction summary.
    """

    predicted_class = recommendation.get("predicted_class", "Unknown")
    prediction_label = "Churn" if str(predicted_class).lower() in {"1", "yes", "true", "churn"} else "No Churn"
    probability = float(recommendation.get("predicted_probability", recommendation.get("churn_probability", recommendation.get("probability", 0.0))))
    confidence = float(recommendation.get("confidence", max(probability, 1.0 - probability)))

    render_prediction_card(
        prediction=prediction_label,
        probability=probability,
        confidence=confidence,
    )

def render_prediction_metrics(
    prediction: PredictionResult,
) -> None:
    """
    Display probability metrics.
    """

    st.subheader("Prediction Metrics")

    probability = prediction.predicted_probability
    confidence = prediction.confidence
    predicted_class = prediction.predicted_class

    metric_1, metric_2, metric_3 = st.columns(3)

    with metric_1:

        st.metric(
            "Prediction",
            str(predicted_class),
        )

    with metric_2:

        st.metric(
            "Churn Probability",
            f"{probability:.2%}",
        )

    with metric_3:

        st.metric(
            "Confidence",
            f"{confidence:.2%}",
        )


def render_prediction_section(
    customer_data: dict[str, Any],
) -> tuple[
    PredictionResult,
    dict[str, Any],
    Any,
    Any,
]:
    """
    Execute prediction workflow.

    Returns
    -------
    tuple
        (
            prediction,
            recommendation,
            model,
            explainer,
        )
    """

    with st.spinner(
        "Generating prediction..."
    ):

        (
            prediction,
            recommendation,
            model,
            explainer,
        ) = generate_prediction(
            customer_data,
        )

    persist_prediction(
        customer_data,
        recommendation,
    )

    render_prediction_result(
        recommendation,
    )

    render_prediction_metrics(
        prediction,
    )

    return (
        prediction,
        recommendation,
        model,
        explainer,
    )

def render_prediction_form():

    customer_data, submitted = render_customer_form()

    if not submitted:

        return None

    if not validate_submission(
        customer_data,
    ):

        return None

    return customer_data

def run_prediction_workflow():
    """
    Complete prediction workflow.

    Returns
    -------
    tuple | None
    """

    customer_data = render_prediction_form()

    if customer_data is None:

        return None

    return (
        customer_data,
        *render_prediction_section(customer_data),
    )

def generate_prediction_explanation(
    model: Any,
    explainer: Any,
    customer_data: dict[str, Any],
) -> Any | None:
    """
    Generate a feature-level explanation for a prediction.

    Parameters
    ----------
    model
        Loaded production model.

    explainer
        Cached SHAP explainer.

    customer_data
        Customer record.

    Returns
    -------
    dict | None
        Explanation payload.
    """

    try:

        if explainer is not None:
            return explain_customer_prediction(
                model=model,
                explainer=explainer,
                customer_data=customer_data,
            )
        return explain_customer_prediction_with_scenarios(
            model=model,
            customer_data=customer_data,
            reference_features=get_cached_background_sample(),
        )

    except Exception as exception:

        logger.exception(
            "Unable to generate SHAP explanation.",
            exc_info=exception,
        )

        return None

def render_explanation_section(
    explanation: Any | None,
) -> None:
    """
    Render explanation UI.
    """

    st.subheader(EXPLANATION_TITLE)

    if explanation is None:

        st.info(
            "No explanation is available for this prediction."
        )

        return

    # The service returns an ExplanationResult dataclass. Normalize it here
    # into the small presentation payload this page needs.
    business_insights = getattr(explanation, "business_insights", None)
    prediction_details = getattr(explanation, "prediction", {})

    if isinstance(explanation, dict):
        business_insights = explanation.get("business_insights")
        prediction_details = explanation

    if business_insights:

        render_business_insight_card(
            {
                "title": f"{business_insights.get('risk_level', 'Customer')} churn risk",
                "narrative": business_insights.get("narrative", "Risk drivers were identified for this customer."),
                "recommendations": business_insights.get("recommendations", []),
            },
        )

    contributors = (
        prediction_details.get("top_positive_contributors", [])
        + prediction_details.get("top_negative_contributors", [])
        if isinstance(prediction_details, dict)
        else []
    )

    if contributors:

        st.markdown("#### Feature-level drivers")
        method = prediction_details.get("explanation_method") if isinstance(prediction_details, dict) else None
        if method:
            st.caption(method)
        contribution_frame = pd.DataFrame(contributors).copy()
        contribution_frame["Direction"] = contribution_frame["shap_value"].map(
            lambda value: "Increases churn risk" if value > 0 else "Reduces churn risk"
        )
        contribution_frame["Impact on churn risk"] = contribution_frame["shap_value"].map(
            lambda value: f"{value:+.4f}"
        )
        st.dataframe(
            contribution_frame[["feature", "Direction", "Impact on churn risk"]].rename(
                columns={"feature": "Feature"}
            ),
            width="stretch",
            hide_index=True,
        )

    summary = prediction_details.get("explanation_summary") if isinstance(prediction_details, dict) else None
    if summary:
        st.caption(summary)

    if not business_insights and not contributors:
        st.info("The prediction completed, but no feature drivers were returned.")

def render_export_section(
    prediction: PredictionResult,
) -> None:
    """
    Render export controls.
    """

    st.divider()

    st.subheader(EXPORT_TITLE)

    package = prepare_prediction_export(
        prediction,
    )

    st.download_button(
        label="Download Prediction",
        data=package["data"],
        file_name=package["filename"],
        mime=package["mime_type"],
        width="stretch",
    )

def render_prediction_output(
    customer_data: dict[str, Any],
    prediction: PredictionResult,
    recommendation: dict[str, Any],
    model: Any,
    explainer: Any,
) -> None:
    """
    Render all prediction outputs.
    """

    explanation = generate_prediction_explanation(
        model=model,
        explainer=explainer,
        customer_data=customer_data,
    )
    render_explanation_section(explanation)

    render_export_section(
        prediction,
    )
    
def render_prediction_page() -> None:
    """
    Render the single-customer prediction page.

    Workflow
    --------
    1. Render page header
    2. Collect customer information
    3. Validate input
    4. Generate prediction
    5. Display prediction summary
    6. Generate SHAP explanation
    7. Display business insights
    8. Export prediction
    """

    render_page_header()

    try:

        workflow = run_prediction_workflow()

        if workflow is None:
            return

        (
            customer_data,
            prediction,
            recommendation,
            model,
            explainer,
        ) = workflow

        render_prediction_output(
            customer_data=customer_data,
            prediction=prediction,
            recommendation=recommendation,
            model=model,
            explainer=explainer,
        )

        logger.info(
            "Prediction workflow completed successfully."
        )

    except Exception as exception:

        handle_prediction_exception(
            exception,
        )

__all__ = [
    "render_prediction_page",
]

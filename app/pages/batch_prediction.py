"""
===============================================================================
Customer Churn Prediction Platform
Batch Prediction Page
===============================================================================

Purpose
-------
Predict churn probability for multiple customers simultaneously.

Responsibilities
----------------
• Upload customer dataset
• Validate uploaded data
• Load cached production resources
• Generate batch predictions
• Generate business recommendations
• Display prediction statistics
• Display high-risk customers
• Export prediction results

Architecture
------------
UI Layer Only

No machine learning logic is implemented in this module.
All prediction logic is delegated to the service layer.

Workflow
--------
Upload CSV
      │
      ▼
Dataset Validation
      │
      ▼
Prediction Service
      │
      ▼
Recommendation Service
      │
      ▼
Business Summary
      │
      ▼
Export Service
===============================================================================
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app.components.cards import (
    render_business_insight_card,
)

from app.services.dataset_service import (
    load_uploaded_dataset,
)

from app.services.prediction_service import (
    predict_batch,
)

from app.services.recommendation_service import (
    generate_batch_recommendations,
)

from app.services.export_service import (
    prepare_dataframe_export,
)

from app.utils.cache import (
    get_cached_model,
    get_cached_threshold_config,
)

from configs.logging_config import (
    get_logger,
)

logger = get_logger(__name__)

PAGE_TITLE = " Batch Prediction"

PAGE_DESCRIPTION = (
    "Upload a customer dataset and generate churn predictions "
    "for every customer using the production model."
)

SUPPORTED_FILE_TYPES = [
    "csv",
]

HIGH_RISK_THRESHOLD = 0.75

def render_page_header() -> None:
    """
    Render page heading.
    """

    st.header(PAGE_TITLE)

    st.write(PAGE_DESCRIPTION)

def load_prediction_resources() -> tuple[Any, Any]:
    """
    Load cached prediction resources.

    Returns
    -------
    tuple
        (
            production_model,
            threshold_configuration,
        )
    """

    logger.info(
        "Loading cached batch prediction resources."
    )

    return (
        get_cached_model(),
        get_cached_threshold_config(),
    )

def render_file_uploader():
    """
    Render dataset uploader.

    Returns
    -------
    UploadedFile | None
    """

    return st.file_uploader(
        label="Upload Customer Dataset",
        type=SUPPORTED_FILE_TYPES,
        help=(
            "Upload a CSV file containing customer records."
        ),
    )

def validate_uploaded_file(
    uploaded_file,
) -> bool:
    """
    Validate uploaded file.
    """

    if uploaded_file is None:
        return False

    if uploaded_file.size == 0:

        st.error(
            "The uploaded file is empty."
        )

        return False

    return True

def read_uploaded_dataset(
    uploaded_file,
) -> pd.DataFrame:
    """
    Load uploaded dataset.
    """

    with st.spinner(
        "Reading uploaded dataset..."
    ):

        dataframe = load_uploaded_dataset(
            uploaded_file,
        )

    logger.info(
        "Loaded uploaded dataset containing %d rows.",
        len(dataframe),
    )

    return dataframe

def preview_dataset(
    dataframe: pd.DataFrame,
) -> None:
    """
    Display dataset preview.
    """

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        dataframe.head(10),
        width="stretch",
    )

    st.caption(
        f"{len(dataframe):,} customers loaded."
    )

def handle_batch_exception(
    exception: Exception,
) -> None:
    """
    Handle workflow exceptions.
    """

    logger.exception(
        "Batch prediction failed.",
        exc_info=exception,
    )

    st.error(
        "Batch prediction could not be completed."
    )

    with st.expander(
        "Technical Details",
        expanded=False,
    ):
        st.code(
            str(exception),
        )

def generate_batch_predictions(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Generate predictions for every customer in the uploaded dataset.

    Parameters
    ----------
    dataframe
        Uploaded customer dataset.

    Returns
    -------
    tuple
        (
            prediction_results,
            summary_statistics,
        )
    """

    model, threshold_configuration = (
        load_prediction_resources()
    )

    with st.spinner(
        "Running predictions..."
    ):

        prediction_results = predict_batch(
            dataframe=dataframe,
            model=model,
        )

        prediction_results = (
            generate_batch_recommendations(
                prediction_results,
                threshold_configuration,
            )
        )

    logger.info(
        "Generated predictions for %d customers.",
        len(prediction_results),
    )

    total_customers = len(
        prediction_results
    )

    positive_label = threshold_configuration.get(
        "positive_label",
        1,
    )
    churn_predictions = int(
        (prediction_results["prediction"] == positive_label).sum()
    )

    non_churn_predictions = (
        total_customers
        - churn_predictions
    )

    average_probability = float(
        prediction_results[
            "probability"
        ].mean()
    )

    summary = {
        "total_customers": total_customers,
        "predicted_churn": churn_predictions,
        "predicted_non_churn": non_churn_predictions,
        "average_probability": average_probability,
    }

    return (
        prediction_results,
        summary,
    )

def render_prediction_summary(
    summary: dict[str, Any],
) -> None:
    """
    Display overall prediction summary.
    """

    st.subheader(
        "Prediction Summary"
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:

        st.metric(
            "Customers",
            f"{summary['total_customers']:,}",
        )

    with col2:

        st.metric(
            "Likely to Churn",
            f"{summary['predicted_churn']:,}",
        )

    with col3:

        st.metric(
            "Likely to Stay",
            f"{summary['predicted_non_churn']:,}",
        )

    with col4:

        st.metric(
            "Average Risk",
            f"{summary['average_probability']:.2%}",
        )

def render_high_risk_customers(
    prediction_results: pd.DataFrame,
) -> None:
    """
    Display customers with high churn probability.
    """

    st.subheader(
        "High-Risk Customers"
    )

    high_risk = prediction_results[
        prediction_results[
            "probability"
        ]
        >= HIGH_RISK_THRESHOLD
    ].copy()

    if high_risk.empty:

        st.success(
            "No customers exceed the high-risk threshold."
        )

        return

    high_risk = high_risk.sort_values(
        by="probability",
        ascending=False,
    )

    st.dataframe(
        high_risk,
        width="stretch",
        hide_index=True,
    )

    render_business_insight_card(
        {
            "title": "Retention Opportunity",
            "narrative": (
                f"{len(high_risk)} customers have a churn "
                "probability above the configured "
                "high-risk threshold."
            ),
            "recommendations": [
                "Prioritize proactive customer outreach.",
                "Review account activity.",
                "Offer personalized retention incentives.",
            ],
        }
    )

def render_prediction_results(
    prediction_results: pd.DataFrame,
) -> None:
    """
    Display complete prediction table.
    """

    st.subheader(
        "Prediction Results"
    )

    st.dataframe(
        prediction_results,
        width="stretch",
        hide_index=True,
    )

def run_batch_prediction_workflow(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Execute the complete batch prediction workflow.

    Parameters
    ----------
    dataframe
        Uploaded customer dataset.

    Returns
    -------
    pd.DataFrame
        Prediction results.
    """

    (
        prediction_results,
        summary,
    ) = generate_batch_predictions(
        dataframe,
    )

    st.divider()

    render_prediction_summary(
        summary,
    )

    st.divider()

    render_high_risk_customers(
        prediction_results,
    )

    st.divider()

    render_prediction_results(
        prediction_results,
    )

    return prediction_results

def render_prediction_distribution(
    prediction_results: pd.DataFrame,
) -> None:
    """
    Display prediction distribution.
    """

    st.subheader(
        "Prediction Distribution"
    )

    prediction_counts = (
        prediction_results["prediction"]
        .value_counts()
        .rename_axis("Prediction")
        .reset_index(name="Customers")
    )

    st.bar_chart(
        prediction_counts.set_index(
            "Prediction"
        )
    )


def render_probability_distribution(
    prediction_results: pd.DataFrame,
) -> None:
    """
    Display churn probability distribution.
    """

    st.subheader(
        "Churn Probability Distribution"
    )

    st.line_chart(
        prediction_results[
            "probability"
        ].sort_values(
            ascending=False
        ).reset_index(
            drop=True
        )
    )

def render_batch_business_summary(
    prediction_results: pd.DataFrame,
) -> None:
    """
    Display business summary.
    """

    total_customers = len(
        prediction_results
    )

    high_risk = prediction_results[
        prediction_results[
            "probability"
        ]
        >= HIGH_RISK_THRESHOLD
    ]

    risk_percentage = (
        len(high_risk)
        / total_customers
        * 100
    )

    if risk_percentage >= 40:

        risk_level = "High"

    elif risk_percentage >= 20:

        risk_level = "Moderate"

    else:

        risk_level = "Low"

    render_business_insight_card(
        {
            "title":
                "Portfolio Risk Assessment",

            "narrative":
                (
                    f"{risk_percentage:.1f}% of customers "
                    f"are classified as high risk. "
                    f"Overall portfolio risk is "
                    f"considered {risk_level.lower()}."
                ),

            "recommendations":
                [
                    "Review high-risk accounts.",
                    "Launch customer retention campaign.",
                    "Monitor churn trends weekly.",
                ],
        }
    )

def render_export_section(
    prediction_results: pd.DataFrame,
) -> None:
    """
    Render export controls.
    """

    st.divider()

    st.subheader(
        "Export Results"
    )

    csv_package = (
        prepare_dataframe_export(
            dataframe=prediction_results,
            filename="batch_predictions",
            export_format="csv",
        )
    )

    excel_package = (
        prepare_dataframe_export(
            dataframe=prediction_results,
            filename="batch_predictions",
            export_format="xlsx",
        )
    )

    json_package = (
        prepare_dataframe_export(
            dataframe=prediction_results,
            filename="batch_predictions",
            export_format="json",
        )
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.download_button(
            label="Download CSV",
            data=csv_package.data,
            file_name=csv_package.filename,
            mime=csv_package.mime_type,
            width="stretch",
        )

    with col2:

        st.download_button(
            label="Download Excel",
            data=excel_package.data,
            file_name=excel_package.filename,
            mime=excel_package.mime_type,
            width="stretch",
        )

    with col3:

        st.download_button(
            label="Download JSON",
            data=json_package.data,
            file_name=json_package.filename,
            mime=json_package.mime_type,
            width="stretch",
        )

def render_batch_dashboard(
    prediction_results: pd.DataFrame,
) -> None:
    """
    Render analytical dashboard.
    """

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        render_prediction_distribution(
            prediction_results,
        )

    with col2:

        render_probability_distribution(
            prediction_results,
        )

    st.divider()

    render_batch_business_summary(
        prediction_results,
    )

    render_export_section(
        prediction_results,
    )

def run_batch_prediction() -> None:
    """
    Execute the complete batch prediction workflow.
    """

    uploaded_file = render_file_uploader()

    if not validate_uploaded_file(
        uploaded_file,
    ):
        return

    dataframe = read_uploaded_dataset(
        uploaded_file,
    )

    if dataframe.empty:

        st.warning(
            "The uploaded dataset does not contain any records."
        )

        return

    preview_dataset(
        dataframe,
    )

    st.divider()

    prediction_results = (
        run_batch_prediction_workflow(
            dataframe,
        )
    )

    render_batch_dashboard(
        prediction_results,
    )

def render_batch_prediction_page() -> None:
    """
    Render the batch prediction page.

    Workflow
    --------
    1. Render page header
    2. Upload customer dataset
    3. Validate uploaded dataset
    4. Generate batch predictions
    5. Display prediction statistics
    6. Display high-risk customers
    7. Display analytical dashboard
    8. Export results
    """

    render_page_header()

    try:

        run_batch_prediction()

        logger.info(
            "Batch prediction workflow completed successfully."
        )

    except Exception as exception:

        handle_batch_exception(
            exception,
        )

__all__ = [
    "render_batch_prediction_page",
]
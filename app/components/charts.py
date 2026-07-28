"""
===============================================================================
Customer Churn Prediction Platform
Reusable Chart Components
===============================================================================

Shared visualization components used throughout the application.

Responsibilities
----------------
• Standardize Plotly charts
• Validate chart inputs
• Apply consistent styling
• Encapsulate visualization logic

Business logic and model training are intentionally excluded.
===============================================================================
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DEFAULT_HEIGHT = 450

DEFAULT_TEMPLATE = "plotly_white"

def validate_dataframe(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate dataframe input.
    """

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise TypeError(
            "Expected a pandas DataFrame."
        )

def validate_columns(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
) -> None:
    """
    Ensure required columns exist.
    """

    missing = [
        column
        for column in columns
        if column not in dataframe.columns
    ]

    if missing:

        raise ValueError(
            "Missing columns: "
            + ", ".join(missing)
        )

def apply_layout_defaults(
    figure: go.Figure,
    *,
    title: str | None = None,
    height: int = DEFAULT_HEIGHT,
) -> go.Figure:
    """
    Apply application-wide layout defaults.
    """

    figure.update_layout(
        title=title,
        template=DEFAULT_TEMPLATE,
        height=height,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    return figure

def render_bar_chart(
    dataframe: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str | None = None,
    color: str | None = None,
    horizontal: bool = False,
) -> None:
    """
    Render a reusable bar chart.
    """

    validate_dataframe(dataframe)
    validate_columns(dataframe, [x, y])

    if horizontal:

        figure = px.bar(
            dataframe,
            x=y,
            y=x,
            color=color,
            orientation="h",
        )

    else:

        figure = px.bar(
            dataframe,
            x=x,
            y=y,
            color=color,
        )

    st.plotly_chart(
        apply_layout_defaults(
            figure,
            title=title,
        ),
        width="stretch",
    )

def render_line_chart(
    dataframe: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str | None = None,
    color: str | None = None,
) -> None:
    """
    Render a line chart.
    """

    validate_dataframe(dataframe)
    validate_columns(dataframe, [x, y])

    figure = px.line(
        dataframe,
        x=x,
        y=y,
        color=color,
    )

    st.plotly_chart(
        apply_layout_defaults(
            figure,
            title=title,
        ),
        width="stretch",
    )

def render_scatter_chart(
    dataframe: pd.DataFrame,
    *,
    x: str,
    y: str,
    color: str | None = None,
    size: str | None = None,
    title: str | None = None,
) -> None:
    """
    Render a scatter plot.
    """

    validate_dataframe(dataframe)
    validate_columns(dataframe, [x, y])

    figure = px.scatter(
        dataframe,
        x=x,
        y=y,
        color=color,
        size=size,
    )

    st.plotly_chart(
        apply_layout_defaults(
            figure,
            title=title,
        ),
        width="stretch",
    )

def render_histogram(
    dataframe: pd.DataFrame,
    *,
    column: str,
    color: str | None = None,
    bins: int = 30,
    title: str | None = None,
) -> None:
    """
    Render a histogram.
    """

    validate_dataframe(dataframe)
    validate_columns(dataframe, [column])

    figure = px.histogram(
        dataframe,
        x=column,
        color=color,
        nbins=bins,
    )

    st.plotly_chart(
        apply_layout_defaults(
            figure,
            title=title,
        ),
        width="stretch",
    )

def render_box_plot(
    dataframe: pd.DataFrame,
    *,
    x: str | None = None,
    y: str,
    color: str | None = None,
    title: str | None = None,
) -> None:
    """
    Render a box plot.
    """

    validate_dataframe(dataframe)

    required = [y]

    if x:
        required.append(x)

    validate_columns(
        dataframe,
        required,
    )

    figure = px.box(
        dataframe,
        x=x,
        y=y,
        color=color,
    )

    st.plotly_chart(
        apply_layout_defaults(
            figure,
            title=title,
        ),
        width="stretch",
    )

def render_pie_chart(
    dataframe: pd.DataFrame,
    *,
    names: str,
    values: str,
    title: str | None = None,
) -> None:
    """
    Render a pie chart.
    """

    validate_dataframe(dataframe)
    validate_columns(
        dataframe,
        [names, values],
    )

    figure = px.pie(
        dataframe,
        names=names,
        values=values,
    )

    st.plotly_chart(
        apply_layout_defaults(
            figure,
            title=title,
        ),
        width="stretch",
    )

def render_area_chart(
    dataframe: pd.DataFrame,
    *,
    x: str,
    y: str,
    color: str | None = None,
    title: str | None = None,
) -> None:
    """
    Render an area chart.
    """

    validate_dataframe(dataframe)
    validate_columns(dataframe, [x, y])

    figure = px.area(
        dataframe,
        x=x,
        y=y,
        color=color,
    )

    st.plotly_chart(
        apply_layout_defaults(
            figure,
            title=title,
        ),
        width="stretch",
    )

def render_confusion_matrix(
    confusion_matrix: pd.DataFrame,
    *,
    title: str = "Confusion Matrix",
) -> None:
    """
    Render a confusion matrix heatmap.
    """

    validate_dataframe(confusion_matrix)

    figure = px.imshow(
        confusion_matrix,
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto",
    )

    figure.update_xaxes(
        title="Predicted Label",
    )

    figure.update_yaxes(
        title="True Label",
    )

    st.plotly_chart(
        apply_layout_defaults(
            figure,
            title=title,
        ),
        width="stretch",
    )

def render_roc_curve(
    roc_dataframe: pd.DataFrame,
    *,
    title: str = "ROC Curve",
) -> None:
    """
    Render Receiver Operating Characteristic curve.

    Expected columns
    ----------------
    false_positive_rate
    true_positive_rate
    """

    validate_dataframe(roc_dataframe)

    validate_columns(
        roc_dataframe,
        [
            "false_positive_rate",
            "true_positive_rate",
        ],
    )

    figure = px.line(
        roc_dataframe,
        x="false_positive_rate",
        y="true_positive_rate",
    )

    figure.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(
            dash="dash",
        ),
    )

    figure.update_xaxes(
        title="False Positive Rate",
    )

    figure.update_yaxes(
        title="True Positive Rate",
    )

    st.plotly_chart(
        apply_layout_defaults(
            figure,
            title=title,
        ),
        width="stretch",
    )

def render_precision_recall_curve(
    pr_dataframe: pd.DataFrame,
    *,
    title: str = "Precision–Recall Curve",
) -> None:
    """
    Render Precision–Recall curve.

    Expected columns
    ----------------
    recall
    precision
    """

    validate_dataframe(
        pr_dataframe,
    )

    validate_columns(
        pr_dataframe,
        [
            "recall",
            "precision",
        ],
    )

    figure = px.line(
        pr_dataframe,
        x="recall",
        y="precision",
    )

    st.plotly_chart(
        apply_layout_defaults(
            figure,
            title=title,
        ),
        width="stretch",
    )

def render_feature_importance(
    feature_dataframe: pd.DataFrame,
    *,
    feature_column: str = "feature",
    importance_column: str = "importance",
    top_n: int = 20,
    title: str = "Feature Importance",
) -> None:
    """
    Render feature importance rankings.
    """

    validate_dataframe(
        feature_dataframe,
    )

    validate_columns(
        feature_dataframe,
        [
            feature_column,
            importance_column,
        ],
    )

    ranked = (
        feature_dataframe
        .sort_values(
            importance_column,
            ascending=False,
        )
        .head(top_n)
    )

    figure = px.bar(
        ranked,
        x=importance_column,
        y=feature_column,
        orientation="h",
    )

    st.plotly_chart(
        apply_layout_defaults(
            figure,
            title=title,
        ),
        width="stretch",
    )

def render_probability_distribution(
    probabilities: pd.DataFrame,
    *,
    probability_column: str = "probability",
    bins: int = 20,
    title: str = "Prediction Probability Distribution",
) -> None:
    """
    Render probability distribution.
    """

    validate_dataframe(
        probabilities,
    )

    validate_columns(
        probabilities,
        [
            probability_column,
        ],
    )

    figure = px.histogram(
        probabilities,
        x=probability_column,
        nbins=bins,
    )

    figure.update_xaxes(
        title="Predicted Probability",
    )

    figure.update_yaxes(
        title="Customer Count",
    )

    st.plotly_chart(
        apply_layout_defaults(
            figure,
            title=title,
        ),
        width="stretch",
    )

def render_calibration_curve(
    calibration_dataframe: pd.DataFrame,
    *,
    title: str = "Calibration Curve",
) -> None:
    """
    Render probability calibration curve.

    Expected columns
    ----------------
    mean_predicted_probability
    fraction_of_positives
    """

    validate_dataframe(
        calibration_dataframe,
    )

    validate_columns(
        calibration_dataframe,
        [
            "mean_predicted_probability",
            "fraction_of_positives",
        ],
    )

    figure = px.line(
        calibration_dataframe,
        x="mean_predicted_probability",
        y="fraction_of_positives",
        markers=True,
    )

    figure.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(
            dash="dash",
        ),
    )

    st.plotly_chart(
        apply_layout_defaults(
            figure,
            title=title,
        ),
        width="stretch",
    )

def render_churn_distribution(
    dataframe: pd.DataFrame,
    *,
    target_column: str = "churn",
    title: str = "Customer Churn Distribution",
) -> None:
    """
    Render churn class distribution.
    """

    validate_dataframe(dataframe)
    validate_columns(dataframe, [target_column])

    distribution = (
        dataframe[target_column]
        .value_counts(dropna=False)
        .rename_axis("Class")
        .reset_index(name="Count")
    )

    figure = px.pie(
        distribution,
        names="Class",
        values="Count",
    )

    st.plotly_chart(
        apply_layout_defaults(
            figure,
            title=title,
        ),
        width="stretch",
    )

def render_customer_segments(
    dataframe: pd.DataFrame,
    *,
    segment_column: str,
    target_column: str = "churn",
    title: str = "Customer Segments",
) -> None:
    """
    Render churn across customer segments.
    """

    validate_dataframe(dataframe)
    validate_columns(
        dataframe,
        [
            segment_column,
            target_column,
        ],
    )

    grouped = (
        dataframe
        .groupby([segment_column, target_column])
        .size()
        .reset_index(name="Count")
    )

    figure = px.bar(
        grouped,
        x=segment_column,
        y="Count",
        color=target_column,
        barmode="group",
    )

    st.plotly_chart(
        apply_layout_defaults(
            figure,
            title=title,
        ),
        width="stretch",
    )

def render_monthly_churn(
    dataframe: pd.DataFrame,
    *,
    month_column: str,
    churn_rate_column: str,
    title: str = "Monthly Churn Rate",
) -> None:
    """
    Render monthly churn trend.
    """

    validate_dataframe(dataframe)

    validate_columns(
        dataframe,
        [
            month_column,
            churn_rate_column,
        ],
    )

    figure = px.line(
        dataframe,
        x=month_column,
        y=churn_rate_column,
        markers=True,
    )

    st.plotly_chart(
        apply_layout_defaults(
            figure,
            title=title,
        ),
        width="stretch",
    )

def render_revenue_loss(
    dataframe: pd.DataFrame,
    *,
    period_column: str,
    revenue_column: str,
    title: str = "Revenue Lost Due to Churn",
) -> None:
    """
    Render estimated revenue loss.
    """

    validate_dataframe(dataframe)

    validate_columns(
        dataframe,
        [
            period_column,
            revenue_column,
        ],
    )

    figure = px.bar(
        dataframe,
        x=period_column,
        y=revenue_column,
    )

    st.plotly_chart(
        apply_layout_defaults(
            figure,
            title=title,
        ),
        width="stretch",
    )

def render_prediction_breakdown(
    dataframe: pd.DataFrame,
    *,
    prediction_column: str = "prediction",
    title: str = "Prediction Breakdown",
) -> None:
    """
    Render prediction counts.
    """

    validate_dataframe(dataframe)
    validate_columns(dataframe, [prediction_column])

    summary = (
        dataframe[prediction_column]
        .value_counts(dropna=False)
        .rename_axis("Prediction")
        .reset_index(name="Count")
    )

    figure = px.bar(
        summary,
        x="Prediction",
        y="Count",
        color="Prediction",
    )

    st.plotly_chart(
        apply_layout_defaults(
            figure,
            title=title,
        ),
        width="stretch",
    )

__all__ = [
    # Core charts
    "render_bar_chart",
    "render_line_chart",
    "render_scatter_chart",
    "render_histogram",
    "render_box_plot",
    "render_pie_chart",
    "render_area_chart",

    # ML charts
    "render_confusion_matrix",
    "render_roc_curve",
    "render_precision_recall_curve",
    "render_feature_importance",
    "render_probability_distribution",
    "render_calibration_curve",

    # Business charts
    "render_churn_distribution",
    "render_customer_segments",
    "render_monthly_churn",
    "render_revenue_loss",
    "render_prediction_breakdown",
]
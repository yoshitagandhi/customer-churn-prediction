"""
===============================================================================
Customer Churn Prediction Platform
Reusable UI Components

File        : components.py
Version     : 1.0

Purpose
-------
Reusable UI components for the Streamlit application.

Responsibilities
----------------
• Metric Cards
• KPI Cards
• Prediction Cards
• Status Badges
• Alerts
• Progress Components
• Profile Cards
• Feature Cards
• Empty States
• Timeline Components
===============================================================================
"""

from __future__ import annotations

from typing import Literal

import streamlit as st

def metric_card(
    *,
    title: str,
    value: str | int | float,
    delta: str | None = None,
    help_text: str | None = None,
) -> None:
    """
    Render a reusable metric card.
    """

    delta_html = (
        f'<div class="kpi-change">{delta}</div>'
        if delta
        else ""
    )

    help_html = (
        f'<div class="card-subtitle">{help_text}</div>'
        if help_text
        else ""
    )

    st.markdown(
        f"""
<div class="kpi-card">

<div class="kpi-label">
{title}
</div>

<div class="kpi-value">
{value}
</div>

{delta_html}

{help_html}

</div>
""",
        unsafe_allow_html=True,
    )

def status_badge(
    label: str,
    status: Literal[
        "success",
        "warning",
        "danger",
        "info",
    ] = "info",
) -> None:
    """
    Display a colored status badge.
    """

    css = f"badge badge-{status}"

    st.markdown(
        f"""
<span class="{css}">
{label}
</span>
""",
        unsafe_allow_html=True,
    )

def alert(
    *,
    title: str,
    message: str,
    kind: Literal[
        "success",
        "warning",
        "danger",
        "info",
    ] = "info",
) -> None:
    """
    Render a reusable alert component.
    """

    st.markdown(
        f"""
<div class="alert alert-{kind}">

<div class="alert-content">

<div class="alert-title">
{title}
</div>

<div class="alert-message">
{message}
</div>

</div>

</div>
""",
        unsafe_allow_html=True,
    )

def progress_bar(
    value: float,
) -> None:
    """
    Render a styled progress bar.

    Parameters
    ----------
    value
        Percentage between 0 and 100.
    """

    value = max(0.0, min(100.0, value))

    st.markdown(
        f"""
<div class="progress">

<div
class="progress-fill"
style="width:{value:.1f}%">
</div>

</div>
""",
        unsafe_allow_html=True,
    )

def feature_card(
    *,
    icon: str,
    title: str,
    description: str,
) -> None:
    """
    Display a feature description card.
    """

    st.markdown(
        f"""
<div class="feature-card">

<div class="feature-icon">
{icon}
</div>

<h3>
{title}
</h3>

<p>
{description}
</p>

</div>
""",
        unsafe_allow_html=True,
    )

__all__ = [
    "metric_card",
    "status_badge",
    "alert",
    "progress_bar",
    "feature_card",
]

def kpi_card(
    *,
    title: str,
    value: str | int | float,
    change: str | None = None,
    trend: Literal["up", "down", "neutral"] = "neutral",
) -> None:
    """
    Display a KPI card with an optional trend indicator.
    """

    trend_class = f"kpi-{trend}"

    change_html = (
        f'<span class="kpi-change {trend_class}">{change}</span>'
        if change
        else ""
    )

    st.markdown(
        f"""
<div class="kpi-card">

<div class="kpi-label">
{title}
</div>

<div class="kpi-value">
{value}
</div>

{change_html}

</div>
""",
        unsafe_allow_html=True,
    )

def statistic_card(
    *,
    title: str,
    value: str | int | float,
    icon: str,
    description: str | None = None,
) -> None:
    """
    Display a statistic card with an icon.
    """

    description_html = (
        f'<div class="card-subtitle">{description}</div>'
        if description
        else ""
    )

    st.markdown(
        f"""
<div class="metric-card">

<div class="metric-icon">
{icon}
</div>

<div class="metric-content">

<div class="metric-value">
{value}
</div>

<div class="metric-label">
{title}
</div>

{description_html}

</div>

</div>
""",
        unsafe_allow_html=True,
    )

def prediction_card(
    *,
    prediction: str,
    confidence: float,
    title: str = "Prediction",
) -> None:
    """
    Display a prediction result with confidence.
    """

    confidence = max(0.0, min(100.0, confidence))

    st.markdown(
        f"""
<div class="prediction-card">

<div class="prediction-title">
{title}
</div>

<div class="prediction-value">
{prediction}
</div>

<div class="prediction-label">

Confidence

</div>

<div class="confidence-bar">

<div
class="confidence-fill"
style="width:{confidence:.1f}%">
</div>

</div>

<br>

<strong>{confidence:.1f}%</strong>

</div>
""",
        unsafe_allow_html=True,
    )

def probability_card(
    *,
    positive_probability: float,
    negative_probability: float,
) -> None:
    """
    Display positive and negative class probabilities.
    """

    positive_probability = max(0.0, min(100.0, positive_probability))
    negative_probability = max(0.0, min(100.0, negative_probability))

    col1, col2 = st.columns(2)

    with col1:

        metric_card(
            title="Positive Class",
            value=f"{positive_probability:.2f}%",
        )

    with col2:

        metric_card(
            title="Negative Class",
            value=f"{negative_probability:.2f}%",
        )


def model_summary_card(
    *,
    model_name: str,
    accuracy: float,
    roc_auc: float,
    f1_score: float,
) -> None:
    """
    Display a quick summary of the selected model.
    """

    st.markdown(
        f"""
<div class="dashboard-card">
<div class="card-header">
<div class="card-title">
{model_name}
</div>
</div>
<table style="width:100%">

<tr>
<td><strong>Accuracy</strong></td>
<td>{accuracy:.4f}</td>
</tr>

<tr>
<td><strong>ROC-AUC</strong></td>
<td>{roc_auc:.4f}</td>
</tr>

<tr>
<td><strong>F1 Score</strong></td>
<td>{f1_score:.4f}</td>
</tr>

</table>

</div>
""",
        unsafe_allow_html=True,
    )


def score_card(
    *,
    label: str,
    score: float,
) -> None:
    """
    Display a normalized score card.
    """

    score = max(0.0, min(1.0, score))

    progress = score * 100

    st.markdown(
        f"""
<div class="card">

<div class="card-title">
{label}
</div>

<h2>
{score:.3f}
</h2>

<div class="progress">

<div
class="progress-fill"
style="width:{progress:.1f}%">
</div>

</div>

</div>
""",
        unsafe_allow_html=True,
    )

__all__.extend(
    [
        "kpi_card",
        "statistic_card",
        "prediction_card",
        "probability_card",
        "model_summary_card",
        "score_card",
    ]
)

def customer_profile_card(
    *,
    customer_id: str,
    name: str | None = None,
    tenure: int | None = None,
    contract: str | None = None,
    monthly_charges: float | None = None,
) -> None:
    """
    Display customer profile information.
    """

    customer_name = name or "Unknown Customer"

    st.markdown(
        f"""
<div class="profile-card">
<div class="profile-avatar">



</div>

<div class="profile-details">

<h3>{customer_name}</h3>

<p><strong>ID:</strong> {customer_id}</p>
<p><strong>Tenure:</strong> {tenure if tenure is not None else "-"}</p>
<p><strong>Contract:</strong> {contract or "-"}</p>
<p><strong>Monthly Charges:</strong> ${
monthly_charges:.2f if monthly_charges is not None else "-"}</p>

</div>

</div>
""",
        unsafe_allow_html=True,
    )

def dataset_summary_card(
    *,
    rows: int,
    columns: int,
    missing_values: int,
    duplicates: int,
) -> None:
    """
    Display dataset summary statistics.
    """

    col1, col2 = st.columns(2)

    with col1:

        metric_card(
            title="Rows",
            value=f"{rows:,}",
        )

        metric_card(
            title="Missing Values",
            value=f"{missing_values:,}",
        )

    with col2:

        metric_card(
            title="Columns",
            value=columns,
        )

        metric_card(
            title="Duplicates",
            value=f"{duplicates:,}",
        )

def experiment_card(
    *,
    model_name: str,
    sampling_strategy: str,
    validation_score: float,
    training_time: float,
) -> None:
    """
    Display experiment summary.
    """

    st.markdown(
        f"""
<div class="dashboard-card">
<div class="card-header">

<div class="card-title">
Experiment Summary
</div>
</div>

<table style="width:100%">

<tr>

<td>Model</td>
<td>{model_name}</td>

</tr>

<tr>

<td>Sampling</td>
<td>{sampling_strategy}</td>

</tr>

<tr>

<td>Validation Score</td>
<td>{validation_score:.4f}</td>

</tr>

<tr>
<td>Training Time</td>
<td>{training_time:.2f} sec</td>
</tr>

</table>

</div>
""",
        unsafe_allow_html=True,
    )

def recommendation_card(
    *,
    title: str,
    recommendations: list[str],
) -> None:
    """
    Display a recommendation panel.
    """

    items = "\n".join(
        f"<li>{item}</li>"
        for item in recommendations
    )

    st.markdown(
        f"""
<div class="card">

<h3>

 {title}

</h3>

<ul>
{items}
</ul>

</div>
""",
        unsafe_allow_html=True,
    )

def report_summary_card(
    *,
    generated_on: str,
    best_model: str,
    report_size: str,
) -> None:
    """
    Display generated report information.
    """

    st.markdown(
        f"""
<div class="dashboard-card">
<div class="card-title">

Generated Report

</div>

<hr>

<p>
<strong>Generated:</strong>
{generated_on}
</p>

<p>
<strong>Best Model:</strong>
{best_model}
</p>

<p>
<strong>Report Size:</strong>
{report_size}
</p>

</div>
""",
        unsafe_allow_html=True,
    )

def information_list(
    *,
    title: str,
    items: dict[str, str | int | float],
) -> None:
    """
    Display key-value information.
    """

    rows = "\n".join(
        f"""
<tr>
<td><strong>{key}</strong></td>
<td>{value}</td>
</tr>
"""
        for key, value in items.items()
    )

    st.markdown(
        f"""
<div class="card">
<div class="card-title">

{title}

</div>
<table style="width:100%">
{rows}
</table>
</div>
""",
        unsafe_allow_html=True,
    )

__all__.extend(
    [
        "customer_profile_card",
        "dataset_summary_card",
        "experiment_card",
        "recommendation_card",
        "report_summary_card",
        "information_list",
    ]
)

def timeline_item(
    *,
    title: str,
    description: str,
    timestamp: str | None = None,
) -> None:
    """
    Display a timeline entry.
    """
    timestamp_html = (
        f"<small>{timestamp}</small>"
        if timestamp
        else ""
    )

    st.markdown(
        f"""
<div class="timeline-item">
<h4>{title}</h4>
<p>{description}</p>
{timestamp_html}

</div>
""",
        unsafe_allow_html=True,
    )

def activity_feed(
    activities: list[dict[str, str]],
) -> None:
    """
    Render an activity feed.

    Expected keys:
        title
        description
        timestamp
    """

    st.markdown('<div class="timeline">', unsafe_allow_html=True)

    for activity in activities:

        timeline_item(
            title=activity.get("title", ""),
            description=activity.get("description", ""),
            timestamp=activity.get("timestamp"),
        )

    st.markdown("</div>", unsafe_allow_html=True)

def loading_card(
    *,
    message: str = "Loading...",
) -> None:
    """
    Display a loading card.
    """

    st.markdown(
        f"""
<div class="card text-center">
<div class="spinner"></div>
<br>
<h3>{message}</h3>

</div>
""",
        unsafe_allow_html=True,
    )

def skeleton_card() -> None:
    """
    Display a skeleton placeholder.
    """

    st.markdown(
        """
<div class="card">
<div class="skeleton skeleton-title"></div>
<div class="skeleton skeleton-text"></div>
<div class="skeleton skeleton-text"></div>
<div class="skeleton skeleton-text"></div>

</div>
""",
        unsafe_allow_html=True,
    )

def empty_state_card(
    *,
    title: str,
    description: str,
    icon: str = "",
) -> None:
    """
    Display an empty state.
    """

    st.markdown(
        f"""
<div class="empty-state">
<h1>{icon}</h1>
<h3>{title}</h3>
<p>{description}</p>

</div>
""",
        unsafe_allow_html=True,
    )

def status_panel(
    *,
    title: str,
    status: Literal[
        "online",
        "processing",
        "warning",
        "offline",
    ],
) -> None:
    """
    Display a colored status indicator.
    """

    st.markdown(
        f"""
<div class="status-card">

<div>

<strong>{title}</strong>

</div>

<div class="status-pill status-{status}">

<div class="status-dot {status}"></div>

{status.title()}

</div>

</div>
""",
        unsafe_allow_html=True,
    )

def log_panel(
    logs: list[str],
) -> None:
    """
    Display application logs.
    """

    content = "<br>".join(logs)

    st.markdown(
        f"""
<div class="log-panel">

{content}

</div>
""",
        unsafe_allow_html=True,
    )

def step_progress(
    current_step: int,
    total_steps: int,
) -> None:
    """
    Display pipeline progress.
    """

    percentage = 0.0

    if total_steps > 0:
        percentage = current_step / total_steps * 100

    st.markdown(
        f"""
<div class="card">

<h4>

Pipeline Progress

</h4>

<p>

Step {current_step} of {total_steps}

</p>

<div class="progress">

<div
class="progress-fill"
style="width:{percentage:.1f}%">
</div>

</div>

</div>
""",
        unsafe_allow_html=True,
    )

__all__.extend(
    [
        "timeline_item",
        "activity_feed",
        "loading_card",
        "skeleton_card",
        "empty_state_card",
        "status_panel",
        "log_panel",
        "step_progress",
    ]
)

def model_comparison_card(
    *,
    model_name: str,
    metrics: dict[str, float],
    is_best: bool = False,
) -> None:
    """
    Display a model comparison summary.
    """

    rows = "".join(
        f"""
<tr>
<td><strong>{metric}</strong></td>
<td>{value:.4f}</td>
</tr>
"""
        for metric, value in metrics.items()
    )

    badge = (
        '<span class="badge badge-success">Best Model</span>'
        if is_best
        else ""
    )

    st.markdown(
        f"""
<div class="dashboard-card">

<div class="card-header">

<div class="card-title">

{model_name}

</div>

{badge}

</div>

<table style="width:100%">
{rows}
</table>

</div>
""",
        unsafe_allow_html=True,
    )

def feature_importance_card(
    *,
    feature_name: str,
    importance: float,
) -> None:
    """
    Display a single ranked feature.
    """

    percentage = max(0.0, min(100.0, importance * 100))

    st.markdown(
        f"""
<div class="feature-card">

<div>

<strong>{feature_name}</strong>

</div>

<div style="width:45%;">

<div class="progress">

<div
class="progress-fill"
style="width:{percentage:.1f}%">
</div>

</div>

</div>

<div>

{importance:.3f}

</div>

</div>
""",
        unsafe_allow_html=True,
    )

def shap_insight_card(
    *,
    feature: str,
    contribution: float,
    direction: Literal["positive", "negative"],
) -> None:
    """
    Display a SHAP contribution.
    """

    badge = (
        "badge-danger"
        if direction == "positive"
        else "badge-success"
    )

    st.markdown(
        f"""
<div class="card">

<div class="card-header">

<div class="card-title">

{feature}

</div>

<span class="badge {badge}">

{direction.title()}

</span>

</div>

<h2>

{contribution:.4f}

</h2>

</div>
""",
        unsafe_allow_html=True,
    )

def business_insight_card(
    *,
    title: str,
    summary: str,
) -> None:
    """
    Display a business recommendation.
    """

    st.markdown(
        f"""
<div class="card">

<h3>

 {title}

</h3>

<p>

{summary}

</p>

</div>
""",
        unsafe_allow_html=True,
    )

def model_health_card(
    *,
    accuracy: float,
    precision: float,
    recall: float,
    roc_auc: float,
) -> None:
    """
    Display overall model health.
    """

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        score_card(
            label="Accuracy",
            score=accuracy,
        )

    with c2:

        score_card(
            label="Precision",
            score=precision,
        )

    with c3:

        score_card(
            label="Recall",
            score=recall,
        )

    with c4:

        score_card(
            label="ROC-AUC",
            score=roc_auc,
        )

def ai_summary_panel(
    *,
    title: str,
    summary: str,
) -> None:
    """
    Display an AI-generated summary.
    """

    st.markdown(
        f"""
<div class="prediction-card">

<div class="prediction-title">

{title}

</div>

<p>

{summary}

</p>

</div>
""",
        unsafe_allow_html=True,
    )

__all__.extend(
    [
        "model_comparison_card",
        "feature_importance_card",
        "shap_insight_card",
        "business_insight_card",
        "model_health_card",
        "ai_summary_panel",
    ]
)
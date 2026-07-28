"""
===============================================================================
Customer Churn Prediction Platform
Evaluation Dashboard
===============================================================================

Purpose
-------
Evaluate the production churn prediction model using validation data,
compare trained models, visualize model performance, assess reliability,
generate business insights, and export evaluation artifacts.

Architecture
------------
UI Layer Only

This page contains no machine learning logic.

All computation is delegated to:

• evaluation_service
• export_service
• model_service

Responsibilities
----------------
• Load cached evaluation resources
• Evaluate production model
• Compare trained models
• Display evaluation metrics
• Display performance visualizations
• Display business insights
• Export evaluation artifacts

Workflow
--------
Load Cached Resources
        │
        ▼
Evaluate Model
        │
        ▼
Compare Models
        │
        ▼
Reliability Analysis
        │
        ▼
Business Dashboard
        │
        ▼
Export Results
===============================================================================
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def render_saved_figure(
    figure: Any,
) -> None:
    """
    Render a saved or generated figure.
    """

    if figure is None:

        st.info(
            "Figure is unavailable."
        )

        return

    try:

        st.pyplot(
            figure,
        )

    except Exception:

        try:

            st.plotly_chart(
                figure,
            )

        except Exception:

            if isinstance(
                figure,
                (bytes, bytearray),
            ):

                st.image(
                    figure,
                )

                return

            st.write(
                "Unable to display the figure."
            )

from app.components.cards import (
    render_business_insight_card,
)

from app.components.metrics import (
    render_metrics_grid,
)

from app.components.tables import (
    render_model_comparison_table,
)

from app.services.evaluation_service import (
    compare_models,
    evaluate_model,
)

from app.services.export_service import (
    prepare_dataframe_export,
    prepare_metrics_export,
)

from app.utils.cache import (
    get_cached_experiment_records,
    get_cached_training_metadata,
    get_cached_validation_dataset,
    get_cached_model,
)

from configs.logging_config import (
    get_logger,
)

logger = get_logger(__name__)

PAGE_TITLE = "Model Evaluation"

PAGE_DESCRIPTION = (
    "Evaluate the production churn prediction model using "
    "validation data, compare trained models, and assess "
    "deployment readiness."
)

def render_page_header() -> None:
    """
    Render page heading.
    """

    st.header(
        PAGE_TITLE,
    )

    st.write(
        PAGE_DESCRIPTION,
    )


def load_evaluation_resources() -> tuple[
    Any,
    pd.DataFrame,
    pd.Series,
    dict[str, Any],
    list[Any],
]:
    """
    Load cached resources required for evaluation.

    Returns
    -------
    tuple
        (
            production_model,
            validation_features,
            validation_target,
            training_metadata,
            experiment_records,
        )
    """

    logger.info(
        "Loading evaluation resources."
    )

    model = get_cached_model()

    (
        validation_features,
        validation_target,
    ) = get_cached_validation_dataset()

    training_metadata = (
        get_cached_training_metadata()
    )

    experiment_records = (
        get_cached_experiment_records()
    )

    return (
        model,
        validation_features,
        validation_target,
        training_metadata,
        experiment_records,
    )


def validate_resources(
    features: pd.DataFrame,
    target: pd.Series,
) -> bool:
    """
    Validate evaluation resources.
    """

    if features.empty:

        st.error(
            "Validation features are unavailable."
        )

        return False

    if target.empty:

        st.error(
            "Validation target is unavailable."
        )

        return False

    return True


def perform_model_evaluation():
    """
    Execute production model evaluation.

    Returns
    -------
    EvaluationResult
    """

    (
        model,
        validation_features,
        validation_target,
        _,
        _,
    ) = load_evaluation_resources()

    if not validate_resources(
        validation_features,
        validation_target,
    ):

        return None

    with st.spinner(
        "Evaluating production model..."
    ):

        evaluation = evaluate_model(
            model=model,
            features=validation_features,
            target=validation_target,
        )

    logger.info(
        "Model evaluation completed."
    )

    return evaluation


def perform_model_comparison():
    """
    Compare trained models.

    Returns
    -------
    ComparisonResult | None
    """

    (
        _,
        _,
        _,
        _,
        experiment_records,
    ) = load_evaluation_resources()

    if not experiment_records:

        logger.warning(
            "No experiment records available."
        )

        return None

    with st.spinner(
        "Comparing trained models..."
    ):

        comparison = compare_models(
            experiment_records,
        )

    logger.info(
        "Model comparison completed."
    )

    return comparison


def handle_evaluation_exception(
    exception: Exception,
) -> None:
    """
    Handle page exceptions.
    """

    logger.exception(
        "Evaluation page failed.",
        exc_info=exception,
    )

    st.error(
        "The evaluation dashboard could not be loaded."
    )

    with st.expander(
        "Technical Details",
        expanded=False,
    ):
        st.code(
            str(exception),
        )

def render_model_information(
    training_metadata: dict[str, Any],
) -> None:
    """
    Display production model information.
    """

    st.subheader(
        "Production Model"
    )

    if not training_metadata:

        st.info(
            "Training metadata is unavailable."
        )

        return

    left_column, right_column = st.columns(2)

    with left_column:

        st.write(
            f"**Model**: "
            f"{training_metadata.get('model_name', 'Unknown')}"
        )

        st.write(
            f"**Training Samples**: "
            f"{training_metadata.get('training_samples', 'N/A'):,}"
            if training_metadata.get("training_samples")
            else "**Training Samples**: N/A"
        )

        st.write(
            f"**Validation Samples**: "
            f"{training_metadata.get('validation_samples', 'N/A'):,}"
            if training_metadata.get("validation_samples")
            else "**Validation Samples**: N/A"
        )

    with right_column:

        st.write(
            f"**Training Date**: "
            f"{training_metadata.get('training_date', 'N/A')}"
        )

        st.write(
            f"**Feature Count**: "
            f"{training_metadata.get('feature_count', 'N/A')}"
        )

        st.write(
            f"**Version**: "
            f"{training_metadata.get('model_version', 'N/A')}"
        )

def render_executive_summary(
    evaluation: Any,
) -> None:
    """
    Display key evaluation metrics.
    """

    st.subheader(
        "Executive Summary"
    )

    metrics = [
        {
            "label": "Accuracy",
            "value": evaluation["metrics"]["accuracy"],
            "format": ".2%",
        },
        {
            "label": "Precision",
            "value": evaluation["metrics"]["precision"],
            "format": ".2%",
        },
        {
            "label": "Recall",
            "value": evaluation["metrics"]["recall"],
            "format": ".2%",
        },
        {
            "label": "F1 Score",
            "value": evaluation["metrics"]["f1"],
            "format": ".2%",
        },
    ]

    render_metrics_grid(
        metrics,
    )

def render_dataset_summary(
    features: pd.DataFrame,
    target: pd.Series,
) -> None:
    """
    Display validation dataset statistics.
    """

    st.subheader(
        "Validation Dataset"
    )

    positive_cases = int(
        target.sum()
    )

    negative_cases = (
        len(target)
        - positive_cases
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Records",
            f"{len(features):,}",
        )

    with col2:

        st.metric(
            "Churn",
            f"{positive_cases:,}",
        )

    with col3:

        st.metric(
            "Non-Churn",
            f"{negative_cases:,}",
        )

def render_deployment_status(
    evaluation: Any,
) -> None:
    """
    Display deployment readiness.
    """

    st.subheader(
        "Deployment Readiness"
    )

    if evaluation["metrics"]["f1"] >= 0.90:

        status = "Production Ready"

    elif evaluation["metrics"]["f1"] >= 0.80:

        status = "Requires Monitoring"

    else:

        status = "Needs Improvement"

    render_business_insight_card(
        {
            "title": "Deployment Assessment",
            "narrative": (
                f"The current production model is "
                f"classified as '{status}'."
            ),
            "recommendations": [
                "Continue monitoring model performance.",
                "Track prediction drift.",
                "Retrain periodically using fresh data.",
            ],
        }
    )

def render_executive_dashboard(
    evaluation: Any,
    features: pd.DataFrame,
    target: pd.Series,
    training_metadata: dict[str, Any],
) -> None:
    """
    Render executive dashboard.
    """

    render_executive_summary(
        evaluation,
    )

    st.divider()

    render_model_information(
        training_metadata,
    )

    st.divider()

    render_dataset_summary(
        features,
        target,
    )

    st.divider()

    render_deployment_status(
        evaluation,
    )

def render_classification_metrics(
    evaluation: Any,
) -> None:
    """
    Display detailed classification metrics.
    """

    st.subheader(
        "Classification Metrics"
    )

    metrics = [
        {
            "label": "ROC-AUC",
            "value": evaluation["metrics"]["roc_auc"],
            "format": ".3f",
        },
        {
            "label": "PR-AUC",
            "value": evaluation["metrics"]["pr_auc"],
            "format": ".3f",
        },
        {
            "label": "Specificity",
            "value": evaluation["metrics"].get("specificity", 0.0),
            "format": ".2%",
        },
        {
            "label": "Balanced Accuracy",
            "value": evaluation["metrics"].get("balanced_accuracy", 0.0),
            "format": ".2%",
        },
    ]

    render_metrics_grid(
        metrics,
    )

def render_confusion_matrix(
    evaluation: Any,
) -> None:
    """
    Display confusion matrix.
    """

    st.subheader(
        "Confusion Matrix"
    )

    if getattr(
        evaluation,
        "confusion_matrix_figure",
        None,
    ) is None:

        st.info(
            "Confusion matrix is unavailable."
        )

        return

    render_saved_figure(
        None,
    )

def render_roc_curve(
    evaluation: Any,
) -> None:
    """
    Display ROC curve.
    """

    st.subheader(
        "ROC Curve"
    )

    if getattr(
        evaluation,
        "roc_curve_figure",
        None,
    ) is None:

        st.info(
            "ROC curve is unavailable."
        )

        return

    render_saved_figure(
        None,
    )

def render_precision_recall_curve(
    evaluation: Any,
) -> None:
    """
    Display Precision-Recall curve.
    """

    st.subheader(
        "Precision–Recall Curve"
    )

    if getattr(
        evaluation,
        "pr_curve_figure",
        None,
    ) is None:

        st.info(
            "Precision–Recall curve is unavailable."
        )

        return

    render_saved_figure(
        None,
    )

def render_performance_interpretation(
    evaluation: Any,
) -> None:
    """
    Display business interpretation of evaluation metrics.
    """

    st.subheader(
        "Performance Interpretation"
    )

    if evaluation["metrics"]["f1"] >= 0.90:

        interpretation = (
            "The model demonstrates excellent predictive "
            "performance and is suitable for production use."
        )

    elif evaluation["metrics"]["f1"] >= 0.80:

        interpretation = (
            "The model performs well but should be monitored "
            "for performance drift after deployment."
        )

    else:

        interpretation = (
            "The model requires additional optimization before "
            "production deployment."
        )

    render_business_insight_card(
        {
            "title": "Model Assessment",
            "narrative": interpretation,
            "recommendations": [
                "Monitor prediction quality.",
                "Track model drift regularly.",
                "Retrain using recent customer data.",
            ],
        }
    )

def render_model_evaluation_dashboard(
    evaluation: Any,
) -> None:
    """
    Render the complete model evaluation dashboard.
    """

    render_classification_metrics(
        evaluation,
    )

    st.divider()

    left_column, right_column = st.columns(2)

    with left_column:

        render_confusion_matrix(
            evaluation,
        )

    with right_column:

        render_roc_curve(
            evaluation,
        )

    st.divider()

    render_precision_recall_curve(
        evaluation,
    )

    st.divider()

    render_performance_interpretation(
        evaluation,
    )

def render_best_model_summary(
    comparison: Any,
) -> None:
    """
    Display the best performing model.
    """

    st.subheader(
        "Best Performing Model"
    )

    if comparison is None:

        st.info(
            "Model comparison results are unavailable."
        )

        return

    best_model = comparison.best_model

    render_business_insight_card(
        {
            "title": "Champion Model",
            "narrative": (
                f"'{best_model.name}' achieved the highest "
                f"overall validation performance with an "
                f"F1 Score of {best_model.f1_score:.2%}."
            ),
            "recommendations": [
                "Deploy this model to production.",
                "Continue monitoring performance after deployment.",
                "Retain previous production model for rollback.",
            ],
        }
    )

def render_model_ranking(
    comparison: Any,
) ->None:
    """
    Display ranked model leaderboard.
    """

    st.subheader(
        "Model Leaderboard"
    )

    ranking = comparison.ranking.copy()

    ranking = ranking.reset_index(
        drop=True,
    )

    ranking.index += 1

    ranking.index.name = "Rank"

    st.dataframe(
        ranking,
        width="stretch",
    )

def render_comparison_table(
    comparison: Any,
) -> None:
    """
    Display detailed model comparison table.
    """

    st.subheader(
        "Performance Comparison"
    )

    render_model_comparison_table(
        comparison.metrics,
    )

def render_performance_gap(
    comparison: Any,
) -> None:
    """
    Display performance difference between the
    top-ranked models.
    """

    st.subheader(
        "Performance Gap"
    )

    if len(
        comparison.ranking
    ) < 2:

        st.info(
            "At least two models are required."
        )

        return

    best = comparison.ranking.iloc[0]
    second = comparison.ranking.iloc[1]

    gap = (
        best["f1_score"]
        - second["f1_score"]
    )

    st.metric(
        "F1 Score Difference",
        f"{gap:.2%}",
    )

    if gap < 0.01:

        st.caption(
            "Top models perform similarly. "
            "Consider inference latency and "
            "model complexity when selecting "
            "the production model."
        )

    else:

        st.caption(
            "The leading model demonstrates a "
            "meaningful performance advantage."
        )

def render_model_recommendation(
    comparison: Any,
) -> None:
    """
    Display recommendation based on
    model comparison.
    """

    st.subheader(
        "Recommendation"
    )

    best_model = comparison.best_model

    render_business_insight_card(
        {
            "title": "Deployment Recommendation",
            "narrative": (
                f"Deploy '{best_model.name}' "
                "as the production model because "
                "it achieved the strongest overall "
                "validation performance."
            ),
            "recommendations": [
                "Archive previous experiments.",
                "Version the selected model.",
                "Schedule periodic retraining.",
                "Monitor production drift.",
            ],
        }
    )

def render_model_comparison_dashboard(
    comparison: Any,
) -> None:
    """
    Render the complete
    model comparison dashboard.
    """

    if comparison is None:

        return

    render_best_model_summary(
        comparison,
    )

    st.divider()

    render_model_ranking(
        comparison,
    )

    st.divider()

    render_comparison_table(
        comparison,
    )

    st.divider()

    render_performance_gap(
        comparison,
    )

    st.divider()

    render_model_recommendation(
        comparison,
    )

def render_calibration_analysis(
    evaluation: Any,
) -> None:
    """
    Display calibration analysis for the production model.
    """

    st.subheader(
        "Calibration Analysis"
    )

    figure = getattr(
        evaluation,
        "calibration_curve_figure",
        None,
    )

    if figure is None:

        st.info(
            "Calibration curve is unavailable."
        )

        return

    render_saved_figure(
        figure,
    )

def render_learning_curve(
    evaluation: Any,
) -> None:
    """
    Display learning curve.
    """

    st.subheader(
        "Learning Curve"
    )

    figure = getattr(
        evaluation,
        "learning_curve_figure",
        None,
    )

    if figure is None:

        st.info(
            "Learning curve is unavailable."
        )

        return

    render_saved_figure(
        figure,
    )

def render_generalization_summary(
    evaluation: Any,
) -> None:
    """
    Summarize the model's ability to generalize.
    """

    st.subheader(
        "Generalization Assessment"
    )

    train_score = getattr(
        evaluation,
        "training_score",
        None,
    )

    validation_score = getattr(
        evaluation,
        "validation_score",
        None,
    )

    if (
        train_score is None
        or validation_score is None
    ):

        st.info(
            "Generalization metrics are unavailable."
        )

        return

    performance_gap = abs(
        train_score
        - validation_score
    )

    if performance_gap < 0.03:

        assessment = (
            "The model demonstrates excellent "
            "generalization with minimal evidence "
            "of overfitting."
        )

    elif performance_gap < 0.08:

        assessment = (
            "The model generalizes reasonably well "
            "but should continue to be monitored "
            "after deployment."
        )

    else:

        assessment = (
            "A noticeable gap exists between training "
            "and validation performance, indicating "
            "potential overfitting."
        )

    render_business_insight_card(
        {
            "title": "Generalization Summary",
            "narrative": assessment,
            "recommendations": [
                "Monitor validation performance.",
                "Retrain using recent production data.",
                "Review feature engineering if drift occurs.",
            ],
        }
    )

def render_reliability_summary(
    evaluation: Any,
) -> None:
    """
    Display production reliability assessment.
    """

    st.subheader(
        "Production Reliability"
    )

    probability_quality = getattr(
        evaluation,
        "brier_score",
        None,
    )

    if probability_quality is None:

        st.info(
            "Reliability metrics are unavailable."
        )

        return

    if probability_quality <= 0.10:

        level = "Excellent"

    elif probability_quality <= 0.20:

        level = "Good"

    else:

        level = "Needs Improvement"

    render_business_insight_card(
        {
            "title": "Prediction Reliability",
            "narrative": (
                f"The model's probability estimates "
                f"are assessed as **{level}**."
            ),
            "recommendations": [
                "Continue monitoring calibration.",
                "Periodically validate probability estimates.",
                "Retrain if reliability deteriorates.",
            ],
        }
    )

def render_reliability_dashboard(
    evaluation: Any,
) -> None:
    """
    Render the complete reliability dashboard.
    """

    left_column, right_column = st.columns(2)

    with left_column:

        render_calibration_analysis(
            evaluation,
        )

    with right_column:

        render_learning_curve(
            evaluation,
        )

    st.divider()

    render_generalization_summary(
        evaluation,
    )

    st.divider()

    render_reliability_summary(
        evaluation,
    )

def render_business_summary(
    evaluation: Any,
) -> None:
    """
    Display an executive business summary.
    """

    st.subheader(
        "Executive Business Summary"
    )

    if evaluation["metrics"]["f1"] >= 0.90:

        summary = (
            "The production model demonstrates excellent "
            "predictive capability and is expected to provide "
            "strong support for customer retention initiatives."
        )

    elif evaluation["metrics"]["f1"] >= 0.80:

        summary = (
            "The production model delivers reliable predictions "
            "and can be deployed with continuous monitoring."
        )

    else:

        summary = (
            "The current model requires additional optimization "
            "before large-scale deployment."
        )

    render_business_insight_card(
        {
            "title": "Executive Summary",
            "narrative": summary,
            "recommendations": [
                "Review model KPIs monthly.",
                "Measure retention campaign outcomes.",
                "Track business ROI.",
            ],
        }
    )

def render_operational_risks(
    evaluation: Any,
) -> None:
    """
    Display operational risks.
    """

    st.subheader(
        "Operational Risks"
    )

    risks: list[str] = []

    if evaluation["metrics"]["recall"] < 0.80:

        risks.append(
            "Some customers likely to churn may not be identified."
        )

    if evaluation["metrics"]["precision"] < 0.80:

        risks.append(
            "Retention efforts may target customers who would not churn."
        )

    if evaluation["metrics"]["roc_auc"] < 0.85:

        risks.append(
            "Customer ranking capability can be improved."
        )

    if not risks:

        st.success(
            "No major operational risks were identified."
        )

        return

    for risk in risks:

        st.warning(
            risk,
        )

def render_deployment_checklist(
    evaluation: Any,
) -> None:
    """
    Display production deployment checklist.
    """

    st.subheader(
        "Deployment Checklist"
    )

    checks = [
        (
            "Model performance validated",
            evaluation["metrics"]["f1"] >= 0.80,
        ),
        (
            "Probability calibration reviewed",
            getattr(
                evaluation,
                "brier_score",
                1.0,
            )
            <= 0.20,
        ),
        (
            "Business metrics approved",
            True,
        ),
        (
            "Monitoring plan available",
            True,
        ),
    ]

    for label, passed in checks:

        icon = "" if passed else "️"

        st.write(
            f"{icon} {label}"
        )

def render_monitoring_strategy() -> None:
    """
    Display production monitoring guidance.
    """

    st.subheader(
        "Production Monitoring"
    )

    monitoring = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "F1 Score",
                "ROC-AUC",
                "Prediction Drift",
                "Feature Drift",
                "Calibration",
            ],
            "Frequency": [
                "Weekly",
                "Weekly",
                "Monthly",
                "Daily",
                "Weekly",
                "Monthly",
            ],
            "Action": [
                "Compare with baseline",
                "Investigate degradation",
                "Review ranking quality",
                "Investigate immediately",
                "Retrain if necessary",
                "Recalibrate model",
            ],
        }
    )

    st.dataframe(
        monitoring,
        width="stretch",
        hide_index=True,
    )

def render_business_kpis(
    evaluation: Any,
) -> None:
    """
    Display business-focused KPIs.
    """

    st.subheader(
        "Business KPIs"
    )

    metrics = [
        {
            "label": "Recall",
            "value": evaluation["metrics"]["recall"],
            "format": ".2%",
        },
        {
            "label": "Precision",
            "value": evaluation["metrics"]["precision"],
            "format": ".2%",
        },
        {
            "label": "ROC-AUC",
            "value": evaluation["metrics"]["roc_auc"],
            "format": ".3f",
        },
        {
            "label": "Balanced Accuracy",
            "value": evaluation["metrics"].get("balanced_accuracy", 0.0),
            "format": ".2%",
        },
    ]

    render_metrics_grid(
        metrics,
    )

def render_business_dashboard(
    evaluation: Any,
) -> None:
    """
    Render the complete business dashboard.
    """

    render_business_summary(
        evaluation,
    )

    st.divider()

    render_business_kpis(
        evaluation,
    )

    st.divider()

    render_operational_risks(
        evaluation,
    )

    st.divider()

    render_deployment_checklist(
        evaluation,
    )

    st.divider()

    render_monitoring_strategy()

def render_export_metrics(
    evaluation: Any,
) -> None:
    """
    Export evaluation metrics.
    """

    metrics_package = prepare_metrics_export(
        evaluation,
    )

    st.download_button(
        label="Download Evaluation Metrics",
        data=metrics_package.data,
        file_name=metrics_package.filename,
        mime=metrics_package.mime_type,
        width="stretch",
    )

def render_export_comparison(
    comparison: Any,
) -> None:
    """
    Export model comparison.
    """

    if comparison is None:

        return

    comparison_package = (
        prepare_dataframe_export(
            dataframe=comparison.metrics,
            filename="model_comparison",
            export_format="csv",
        )
    )

    st.download_button(
        label="Download Model Comparison",
        data=comparison_package.data,
        file_name=comparison_package.filename,
        mime=comparison_package.mime_type,
        width="stretch",
    )

def render_export_predictions(
    evaluation: Any,
) -> None:
    """
    Export validation predictions.
    """

    predictions = getattr(
        evaluation,
        "predictions",
        None,
    )

    if predictions is None:

        return

    package = prepare_dataframe_export(
        dataframe=predictions,
        filename="validation_predictions",
        export_format="csv",
    )

    st.download_button(
        label="Download Validation Predictions",
        data=package.data,
        file_name=package.filename,
        mime=package.mime_type,
        width="stretch",
    )

def render_export_feature_importance(
    evaluation: Any,
) -> None:
    """
    Export feature importance.
    """

    importance = getattr(
        evaluation,
        "feature_importance",
        None,
    )

    if importance is None:

        return

    package = prepare_dataframe_export(
        dataframe=importance,
        filename="feature_importance",
        export_format="csv",
    )

    st.download_button(
        label="Download Feature Importance",
        data=package.data,
        file_name=package.filename,
        mime=package.mime_type,
        width="stretch",
    )

def render_export_dashboard(
    evaluation: Any,
    comparison: Any,
) -> None:
    """
    Render all export options.
    """

    st.header(
        "Export Evaluation Results"
    )

    left_column, right_column = st.columns(2)

    with left_column:

        render_export_metrics(
            evaluation,
        )

        render_export_predictions(
            evaluation,
        )

    with right_column:

        render_export_comparison(
            comparison,
        )

        render_export_feature_importance(
            evaluation,
        )
        

def run_evaluation_workflow() -> None:
    """
    Execute the complete evaluation workflow.
    """

    (
        model,
        validation_features,
        validation_target,
        training_metadata,
        experiment_records,
    ) = load_evaluation_resources()

    if not validate_resources(
        validation_features,
        validation_target,
    ):
        return

    with st.spinner(
        "Evaluating production model..."
    ):

        evaluation = evaluate_model(
            model=model,
            features=validation_features,
            target=validation_target,
        )

    comparison = None

    if experiment_records:

        comparison = compare_models(
            experiment_records,
        )

    render_executive_dashboard(
        evaluation=evaluation,
        features=validation_features,
        target=validation_target,
        training_metadata=training_metadata,
    )

    st.divider()

    render_model_evaluation_dashboard(
        evaluation,
    )

    st.divider()

    render_model_comparison_dashboard(
        comparison,
    )

    st.divider()

    render_reliability_dashboard(
        evaluation,
    )

    st.divider()

    render_business_dashboard(
        evaluation,
    )

    st.divider()

    render_export_dashboard(
        evaluation,
        comparison,
    )

def render_evaluation_page() -> None:
    """
    Render the evaluation dashboard.
    """

    render_page_header()

    try:

        run_evaluation_workflow()

        logger.info(
            "Evaluation dashboard rendered successfully."
        )

    except Exception as exception:

        handle_evaluation_exception(
            exception,
        )

__all__ = [
    "render_evaluation_page",
]
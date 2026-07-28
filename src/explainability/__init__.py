"""SHAP explainability, feature attribution, and business insights.

Explains the best-performing model (Milestone 6/7) both globally
(which features drive churn predictions overall) and locally (why a
specific customer was predicted to churn), and translates those
explanations into business-friendly language and grounded
recommendations. No model retraining, evaluation, or threshold
optimization happens here.

Typical usage::

    from src.explainability import run_explainability_pipeline

    results = run_explainability_pipeline(features_train, features_eval)
"""

from typing import Any

import pandas as pd

from configs.config import settings
from configs.logging_config import get_logger
from src.explainability.business_insights import generate_business_insights
from src.explainability.explainer import (
    compute_shap_values,
    get_processed_features,
    load_explainer,
    load_explainer_inputs,
)
from src.explainability.feature_importance import (
    get_top_negative_contributors,
    get_top_positive_contributors,
    rank_features,
)
from src.explainability.prediction_explainer import explain_prediction
from src.explainability.report import generate_shap_report
from src.explainability.shap_analysis import (
    extract_positive_class_values,
    generate_global_explanations,
    generate_local_explanation,
    plot_shap_bar,
    plot_shap_beeswarm,
    plot_shap_dependence,
    plot_shap_summary,
    plot_shap_waterfall,
)

logger = get_logger(__name__)

__all__ = [
    "load_explainer_inputs",
    "get_processed_features",
    "load_explainer",
    "compute_shap_values",
    "extract_positive_class_values",
    "generate_global_explanations",
    "generate_local_explanation",
    "plot_shap_bar",
    "plot_shap_beeswarm",
    "plot_shap_summary",
    "plot_shap_dependence",
    "plot_shap_waterfall",
    "rank_features",
    "get_top_positive_contributors",
    "get_top_negative_contributors",
    "explain_prediction",
    "generate_business_insights",
    "generate_shap_report",
    "run_explainability_pipeline",
]


def run_explainability_pipeline(
    features_train: pd.DataFrame,
    features_eval: pd.DataFrame,
    model_path: Any = settings.best_model_path,
    metadata_path: Any = settings.training_metadata_path,
    example_customer_positions: list[int] | None = None,
    positive_label: str = settings.positive_label,
) -> dict[str, Any]:
    """Run the full explainability pipeline end to end.

    Args:
        features_train: Training features (raw, pre-preprocessing),
            used only as the SHAP explainer's background distribution.
        features_eval: Features to compute SHAP values for (typically
            the validation/test split from Milestone 6/7). Example
            customer explanations are drawn from this set.
        model_path: Path to the serialized best model. Defaults to
            ``settings.best_model_path``.
        metadata_path: Path to the training metadata JSON. Defaults to
            ``settings.training_metadata_path``.
        example_customer_positions: Row positions (within
            ``features_eval``) to generate example explanations for.
            Defaults to the first ``settings.shap_example_customers_count``
            rows. Any customer can be explained — this is not limited
            to a fixed example.
        positive_label: The label representing churn. Defaults to
            ``settings.positive_label``.

    Returns:
        A dictionary with the global explanation, ranked features,
        top global positive/negative contributors, example customer
        explanations, every generated figure path, and the generated
        report paths.
    """
    inputs = load_explainer_inputs(model_path, metadata_path)
    pipeline = inputs["pipeline"]

    background_processed = get_processed_features(pipeline, features_train)
    eval_processed = get_processed_features(pipeline, features_eval)

    explainer = load_explainer(pipeline, background_processed)
    shap_values = compute_shap_values(explainer, eval_processed)
    feature_names = eval_processed.columns.tolist()

    global_explanation = generate_global_explanations(shap_values, feature_names)
    ranked_features = rank_features(global_explanation)
    shap_values_array = extract_positive_class_values(shap_values)
    top_positive_global = get_top_positive_contributors(shap_values_array, feature_names)
    top_negative_global = get_top_negative_contributors(shap_values_array, feature_names)

    figure_paths: dict[str, Any] = {
        "shap_bar": plot_shap_bar(shap_values),
        "shap_beeswarm": plot_shap_beeswarm(shap_values),
        "shap_summary": plot_shap_summary(shap_values_array, eval_processed),
        "shap_dependence": plot_shap_dependence(
            ranked_features[0]["feature"], shap_values_array, eval_processed
        ),
    }
    logger.info("Visualizations generated.")

    if example_customer_positions is None:
        example_customer_positions = list(
            range(min(settings.shap_example_customers_count, len(features_eval)))
        )

    example_explanations = []
    for position in example_customer_positions:
        customer_row = features_eval.iloc[[position]]
        prediction = explain_prediction(
            pipeline, explainer, customer_row, positive_label=positive_label
        )
        insights = generate_business_insights(prediction)
        example_explanations.append(
            {
                "customer_index": int(features_eval.index[position]),
                "prediction": prediction,
                "business_insights": insights,
            }
        )
    logger.info("Business insights created.")

    figure_paths["shap_waterfall"] = plot_shap_waterfall(shap_values, example_customer_positions[0])

    report_paths = generate_shap_report(ranked_features, example_explanations, figure_paths)
    logger.info("Reports generated.")
    logger.info("Explainability completed.")

    return {
        "global_explanation": global_explanation,
        "ranked_features": ranked_features,
        "top_positive_global_contributors": top_positive_global,
        "top_negative_global_contributors": top_negative_global,
        "example_explanations": example_explanations,
        "figure_paths": figure_paths,
        "report_paths": report_paths,
    }

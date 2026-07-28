"""Business insight generation from EDA statistics.

Every insight and recommendation produced here is derived directly
from values computed by :mod:`src.analysis.statistics` for the
dataset actually being analyzed. Nothing about a specific dataset
(e.g., "month-to-month contracts churn more") is hardcoded — the
templates only describe whatever pattern the numbers show, and are
skipped when the observed effect is too small to be meaningful.
"""

from typing import Any, Final

import pandas as pd

from configs.config import settings
from configs.logging_config import get_logger
from src.analysis.statistics import (
    CONTINUOUS_NUMERIC_FEATURES,
    KEY_CATEGORICAL_FEATURES,
    coerce_numeric,
    compute_churn_rate_by_category,
)

logger = get_logger(__name__)

# Minimum percentage-point gap between a category's churn rate and the
# overall churn rate before it is considered a meaningful pattern
# worth surfacing as an insight.
_MIN_MEANINGFUL_GAP_POINTS: Final[float] = 5.0


def generate_categorical_insights(
    dataframe: pd.DataFrame,
    categorical_columns: tuple[str, ...] = KEY_CATEGORICAL_FEATURES,
    target_column: str = settings.target_column,
) -> list[dict[str, Any]]:
    """Generate one insight per categorical feature with a meaningful churn gap.

    Args:
        dataframe: The dataset to analyze.
        categorical_columns: Categorical columns to evaluate. Defaults
            to ``KEY_CATEGORICAL_FEATURES``.
        target_column: Name of the target column. Defaults to
            ``settings.target_column``.

    Returns:
        A list of insight dictionaries, each with the feature name,
        the highest/lowest churn-rate groups, the gap between them,
        and a human-readable narrative.
    """
    insights: list[dict[str, Any]] = []
    for column in categorical_columns:
        if column not in dataframe.columns:
            continue

        churn_rates = compute_churn_rate_by_category(dataframe, column, target_column)
        if len(churn_rates) < 2:
            continue

        highest_group, highest_rate = next(iter(churn_rates.items()))
        lowest_group, lowest_rate = list(churn_rates.items())[-1]
        gap = round(highest_rate - lowest_rate, 2)

        if gap < _MIN_MEANINGFUL_GAP_POINTS:
            continue

        insights.append(
            {
                "feature": column,
                "highest_churn_group": highest_group,
                "highest_churn_rate": highest_rate,
                "lowest_churn_group": lowest_group,
                "lowest_churn_rate": lowest_rate,
                "gap_percentage_points": gap,
                "narrative": (
                    f"Customers with {column} = '{highest_group}' churn at {highest_rate}%, "
                    f"compared to {lowest_rate}% for {column} = '{lowest_group}' — a gap of "
                    f"{gap} percentage points, suggesting {column} is associated with churn risk."
                ),
            }
        )

    logger.debug("Generated %d categorical insight(s).", len(insights))
    return insights


def generate_numerical_insights(
    dataframe: pd.DataFrame,
    numeric_columns: tuple[str, ...] = CONTINUOUS_NUMERIC_FEATURES,
    target_column: str = settings.target_column,
    positive_label: str = "Yes",
) -> list[dict[str, Any]]:
    """Compare the mean of each numeric feature between churned and retained customers.

    Args:
        dataframe: The dataset to analyze.
        numeric_columns: Numeric columns to evaluate. Defaults to
            ``CONTINUOUS_NUMERIC_FEATURES``.
        target_column: Name of the target column. Defaults to
            ``settings.target_column``.
        positive_label: The target label representing churn. Defaults
            to "Yes".

    Returns:
        A list of insight dictionaries comparing churned vs. retained
        customer means for each numeric feature.
    """
    insights: list[dict[str, Any]] = []
    for column in numeric_columns:
        if column not in dataframe.columns:
            continue

        numeric_series = coerce_numeric(dataframe[column])
        is_churned = dataframe[target_column] == positive_label

        churned_mean = numeric_series[is_churned].mean()
        retained_mean = numeric_series[~is_churned].mean()
        if pd.isna(churned_mean) or pd.isna(retained_mean):
            continue

        direction = "higher" if churned_mean > retained_mean else "lower"
        insights.append(
            {
                "feature": column,
                "churned_mean": round(float(churned_mean), 2),
                "retained_mean": round(float(retained_mean), 2),
                "narrative": (
                    f"Churned customers have a {direction} average {column} "
                    f"({churned_mean:.2f}) compared to retained customers ({retained_mean:.2f})."
                ),
            }
        )

    logger.debug("Generated %d numerical insight(s).", len(insights))
    return insights


def generate_business_recommendations(
    categorical_insights: list[dict[str, Any]], numerical_insights: list[dict[str, Any]]
) -> list[str]:
    """Derive actionable recommendations from generated insights.

    Recommendations reference only the features and groups that were
    actually flagged as meaningful; no generic or assumed advice is
    included.

    Args:
        categorical_insights: Output of ``generate_categorical_insights``.
        numerical_insights: Output of ``generate_numerical_insights``.

    Returns:
        A list of recommendation strings.
    """
    recommendations: list[str] = []

    for insight in categorical_insights:
        recommendations.append(
            f"Prioritize retention efforts for customers with "
            f"{insight['feature']} = '{insight['highest_churn_group']}', "
            f"which shows a {insight['gap_percentage_points']}-point higher churn rate."
        )

    for insight in numerical_insights:
        if insight["churned_mean"] < insight["retained_mean"]:
            recommendations.append(
                f"Investigate retention programs for customers with low {insight['feature']} "
                f"(churned customers average {insight['churned_mean']:.2f} vs. "
                f"{insight['retained_mean']:.2f} for retained customers)."
            )

    if not recommendations:
        recommendations.append(
            "No strong churn-driving patterns were detected in the analyzed features; "
            "consider expanding the feature set analyzed in future milestones."
        )

    return recommendations


def generate_business_insights(
    dataframe: pd.DataFrame,
    target_column: str = settings.target_column,
    categorical_columns: tuple[str, ...] = KEY_CATEGORICAL_FEATURES,
    numeric_columns: tuple[str, ...] = CONTINUOUS_NUMERIC_FEATURES,
) -> dict[str, Any]:
    """Generate the full set of business insights and recommendations.

    Args:
        dataframe: The dataset to analyze.
        target_column: Name of the target column. Defaults to
            ``settings.target_column``.
        categorical_columns: Categorical columns to evaluate.
        numeric_columns: Numeric columns to evaluate.

    Returns:
        A dictionary with "categorical_insights", "numerical_insights",
        and "recommendations".
    """
    categorical_insights = generate_categorical_insights(
        dataframe, categorical_columns, target_column
    )
    numerical_insights = generate_numerical_insights(dataframe, numeric_columns, target_column)
    recommendations = generate_business_recommendations(categorical_insights, numerical_insights)

    return {
        "categorical_insights": categorical_insights,
        "numerical_insights": numerical_insights,
        "recommendations": recommendations,
    }

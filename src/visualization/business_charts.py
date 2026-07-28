"""Business-oriented churn visualizations.

Unlike :mod:`src.visualization.distributions` (which describes a
single feature in isolation), these charts always relate a feature to
the churn outcome, framed for business interpretation: "how does the
churn rate differ across groups of this feature?"
"""

from pathlib import Path
from typing import Final

import pandas as pd

from configs.config import settings
from configs.logging_config import get_logger
from configs.paths import FIGURES_DIR
from src.utils.helpers import coerce_numeric, compute_churn_rate_by_category
from src.visualization.plots import plot_bar_from_series

logger = get_logger(__name__)

# (category_column, output_filename) pairs for the standard business
# churn-rate-by-category charts this project reports on.
_CATEGORY_CHART_SPECS: Final[tuple[tuple[str, str], ...]] = (
    ("Contract", "contract_vs_churn.png"),
    ("InternetService", "internet_service_vs_churn.png"),
    ("PaymentMethod", "payment_method_vs_churn.png"),
    ("SeniorCitizen", "senior_citizen_vs_churn.png"),
    ("TechSupport", "tech_support_vs_churn.png"),
    ("OnlineSecurity", "online_security_vs_churn.png"),
)

_TENURE_BIN_WIDTH_MONTHS: Final[int] = 12
_MONTHLY_CHARGES_BIN_LABELS: Final[tuple[str, ...]] = ("Low", "Medium", "High", "Very High")


def plot_churn_rate_by_category(
    dataframe: pd.DataFrame,
    category_column: str,
    target_column: str = settings.target_column,
    title: str | None = None,
    filename: str | None = None,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Plot the churn rate (%) for each category of a given column.

    Args:
        dataframe: The dataset to visualize.
        category_column: Categorical column to group by.
        target_column: Name of the target column. Defaults to
            ``settings.target_column``.
        title: Figure title. Defaults to "Churn Rate by {column}".
        filename: Output filename. Defaults to a slugified version of
            the column name.
        output_dir: Directory to save into.

    Returns:
        The path the figure was saved to.
    """
    churn_rate = compute_churn_rate_by_category(dataframe, category_column, target_column)
    series = pd.Series(churn_rate)

    return plot_bar_from_series(
        series,
        title=title or f"Churn Rate by {category_column}",
        filename=filename or f"{category_column.lower()}_vs_churn.png",
        xlabel=category_column,
        ylabel="Churn Rate (%)",
        value_suffix="%",
        output_dir=output_dir,
    )


def plot_churn_rate_by_tenure_group(
    dataframe: pd.DataFrame,
    target_column: str = settings.target_column,
    tenure_column: str = "tenure",
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Plot the churn rate (%) across tenure groups (binned in fixed-width months).

    Args:
        dataframe: The dataset to visualize.
        target_column: Name of the target column. Defaults to
            ``settings.target_column``.
        tenure_column: Name of the tenure column. Defaults to "tenure".
        output_dir: Directory to save into.

    Returns:
        The path the figure was saved to.
    """
    tenure_groups = _bin_tenure(dataframe[tenure_column])
    working_frame = dataframe.assign(**{"_tenure_group": tenure_groups})
    return plot_churn_rate_by_category(
        working_frame,
        category_column="_tenure_group",
        target_column=target_column,
        title="Churn Rate by Tenure Group",
        filename="tenure_group_vs_churn.png",
        output_dir=output_dir,
    )


def plot_churn_rate_by_monthly_charges_group(
    dataframe: pd.DataFrame,
    target_column: str = settings.target_column,
    monthly_charges_column: str = "MonthlyCharges",
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Plot the churn rate (%) across monthly-charges quartile groups.

    Args:
        dataframe: The dataset to visualize.
        target_column: Name of the target column. Defaults to
            ``settings.target_column``.
        monthly_charges_column: Name of the monthly charges column.
            Defaults to "MonthlyCharges".
        output_dir: Directory to save into.

    Returns:
        The path the figure was saved to.
    """
    charges_groups = _bin_into_quartiles(dataframe[monthly_charges_column])
    working_frame = dataframe.assign(**{"_monthly_charges_group": charges_groups})
    return plot_churn_rate_by_category(
        working_frame,
        category_column="_monthly_charges_group",
        target_column=target_column,
        title="Churn Rate by Monthly Charges Group",
        filename="monthly_charges_vs_churn.png",
        output_dir=output_dir,
    )


def generate_business_charts(
    dataframe: pd.DataFrame,
    target_column: str = settings.target_column,
    output_dir: Path = FIGURES_DIR,
) -> dict[str, Path]:
    """Generate the full set of business churn-rate charts.

    Args:
        dataframe: The dataset to visualize.
        target_column: Name of the target column. Defaults to
            ``settings.target_column``.
        output_dir: Directory to save into.

    Returns:
        A mapping of chart identifier to the path of its generated
        figure.
    """
    chart_paths: dict[str, Path] = {}

    for category_column, filename in _CATEGORY_CHART_SPECS:
        if category_column not in dataframe.columns:
            logger.debug("Column '%s' not found; skipping business chart.", category_column)
            continue
        chart_paths[category_column] = plot_churn_rate_by_category(
            dataframe, category_column, target_column, filename=filename, output_dir=output_dir
        )

    if "tenure" in dataframe.columns:
        chart_paths["tenure_group"] = plot_churn_rate_by_tenure_group(
            dataframe, target_column, output_dir=output_dir
        )

    if "MonthlyCharges" in dataframe.columns:
        chart_paths["monthly_charges_group"] = plot_churn_rate_by_monthly_charges_group(
            dataframe, target_column, output_dir=output_dir
        )

    logger.info("Generated %d business chart(s).", len(chart_paths))
    return chart_paths


def _bin_tenure(series: pd.Series, bin_width: int = _TENURE_BIN_WIDTH_MONTHS) -> pd.Series:
    """Bin a tenure (months) column into fixed-width groups.

    Bin edges are derived from the data's own range, so the grouping
    adapts to whatever tenure values are actually present.

    Args:
        series: The tenure column to bin.
        bin_width: Width of each group, in months.

    Returns:
        A categorical Series of tenure group labels (e.g., "0-12").
    """
    numeric_series = coerce_numeric(series)
    max_tenure = int(numeric_series.max())
    bin_edges = list(range(0, max_tenure + bin_width, bin_width))
    labels = [f"{bin_edges[i]}-{bin_edges[i + 1]}" for i in range(len(bin_edges) - 1)]
    return pd.cut(numeric_series, bins=bin_edges, labels=labels, include_lowest=True)


def _bin_into_quartiles(
    series: pd.Series, labels: tuple[str, ...] = _MONTHLY_CHARGES_BIN_LABELS
) -> pd.Series:
    """Bin a numeric column into quartile-based groups.

    Args:
        series: The numeric column to bin.
        labels: Labels to assign to each quartile, from lowest to
            highest.

    Returns:
        A categorical Series of quartile group labels.
    """
    numeric_series = coerce_numeric(series)
    return pd.qcut(numeric_series, q=len(labels), labels=labels, duplicates="drop")

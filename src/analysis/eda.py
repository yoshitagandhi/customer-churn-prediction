"""Main exploratory data analysis (EDA) pipeline.

This module coordinates the EDA workflow — it does not contain
analysis logic itself. It consumes an already-validated DataFrame
(produced by the Milestone 2 data ingestion pipeline) and delegates
to :mod:`src.analysis.statistics`, :mod:`src.visualization.plots`,
:mod:`src.visualization.distributions`,
:mod:`src.visualization.business_charts`, and
:mod:`src.analysis.business_insights`, then hands everything to
:mod:`src.analysis.report_generator`.

This module never loads or validates raw data itself; that
responsibility belongs to :mod:`src.data`.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from configs.config import settings
from configs.logging_config import get_logger
from configs.paths import FIGURES_DIR
from src.analysis import business_insights as business_insights_module
from src.analysis import report_generator
from src.analysis.statistics import (
    CONTINUOUS_NUMERIC_FEATURES,
    KEY_CATEGORICAL_FEATURES,
    compute_categorical_statistics,
    compute_correlation_matrix,
    compute_dataset_statistics,
    compute_numerical_statistics,
    compute_target_statistics,
    get_top_correlations,
)
from src.visualization.business_charts import generate_business_charts
from src.visualization.distributions import (
    analyze_categorical_distributions,
    analyze_numerical_distributions,
)
from src.visualization.plots import (
    configure_plot_style,
    plot_correlation_heatmap,
    plot_target_distribution,
)

logger = get_logger(__name__)


def run_eda_pipeline(
    dataframe: pd.DataFrame,
    target_column: str = settings.target_column,
    numeric_columns: tuple[str, ...] = CONTINUOUS_NUMERIC_FEATURES,
    categorical_columns: tuple[str, ...] = KEY_CATEGORICAL_FEATURES,
    figures_dir: Path = FIGURES_DIR,
) -> dict[str, Any]:
    """Run the full EDA and business intelligence pipeline.

    Args:
        dataframe: The already-validated dataset to analyze, as
            produced by ``src.data.run_data_quality_pipeline``. This
            function does not load or re-validate data itself.
        target_column: Name of the target column. Defaults to
            ``settings.target_column``.
        numeric_columns: Continuous numeric columns to analyze.
        categorical_columns: Categorical columns to analyze.
        figures_dir: Directory figures are saved into. Defaults to
            ``configs.paths.FIGURES_DIR``.

    Returns:
        A dictionary containing every artifact produced: statistics,
        target analysis, correlation summary, business insights,
        generated figure paths, and generated report paths.
    """
    logger.info("EDA started.")
    configure_plot_style()

    dataset_statistics = compute_dataset_statistics(dataframe)
    numerical_statistics = compute_numerical_statistics(dataframe, numeric_columns)
    categorical_statistics = compute_categorical_statistics(dataframe, categorical_columns)
    target_statistics = compute_target_statistics(dataframe, target_column)
    correlation_matrix = compute_correlation_matrix(dataframe, numeric_columns)
    correlation_summary = get_top_correlations(correlation_matrix)
    logger.info("Statistics generated.")

    figure_paths = _generate_all_visualizations(
        dataframe,
        target_column,
        numeric_columns,
        categorical_columns,
        correlation_matrix,
        figures_dir,
    )
    logger.info("Visualizations created.")

    insights = business_insights_module.generate_business_insights(
        dataframe, target_column, categorical_columns, numeric_columns
    )
    logger.info("Business insights generated.")

    report_paths = report_generator.generate_eda_report(
        dataset_statistics,
        numerical_statistics,
        categorical_statistics,
        target_statistics,
        correlation_summary,
        insights,
        figure_paths,
    )
    logger.info("Reports saved.")
    logger.info("EDA completed.")

    return {
        "dataset_statistics": dataset_statistics,
        "numerical_statistics": numerical_statistics,
        "categorical_statistics": categorical_statistics,
        "target_statistics": target_statistics,
        "correlation_summary": correlation_summary,
        "business_insights": insights,
        "figure_paths": figure_paths,
        "report_paths": report_paths,
    }


def _generate_all_visualizations(
    dataframe: pd.DataFrame,
    target_column: str,
    numeric_columns: tuple[str, ...],
    categorical_columns: tuple[str, ...],
    correlation_matrix: pd.DataFrame,
    figures_dir: Path,
) -> dict[str, Any]:
    """Generate every figure required by this milestone and collect their paths.

    Args:
        dataframe: The dataset to visualize.
        target_column: Name of the target column.
        numeric_columns: Continuous numeric columns to analyze.
        categorical_columns: Categorical columns to analyze.
        correlation_matrix: Pre-computed numeric correlation matrix.
        figures_dir: Directory figures are saved into.

    Returns:
        A dictionary grouping figure paths by category: "target",
        "numerical", "categorical", "correlation", and "business".
    """
    target_figure = plot_target_distribution(dataframe, target_column, output_dir=figures_dir)
    numerical_figures = analyze_numerical_distributions(
        dataframe, numeric_columns, output_dir=figures_dir
    )
    categorical_figures = analyze_categorical_distributions(
        dataframe, categorical_columns, output_dir=figures_dir
    )
    correlation_figure = plot_correlation_heatmap(correlation_matrix, output_dir=figures_dir)
    business_figures = generate_business_charts(dataframe, target_column, output_dir=figures_dir)

    return {
        "target": target_figure,
        "numerical": numerical_figures,
        "categorical": categorical_figures,
        "correlation": correlation_figure,
        "business": business_figures,
    }

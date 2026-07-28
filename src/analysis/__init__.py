"""Exploratory data analysis and business intelligence.

This package turns a validated DataFrame (produced by
:mod:`src.data`) into descriptive statistics, business insights, and
automated reports. It performs no data cleaning, feature engineering,
or model training — see the ``preprocessing``, ``features``, and
``models`` packages for those concerns in later milestones.

Typical usage::

    from src.data import run_data_quality_pipeline
    from src.analysis import run_eda_pipeline

    dataframe, _, _ = run_data_quality_pipeline()
    eda_results = run_eda_pipeline(dataframe)
"""

from src.analysis.business_insights import generate_business_insights
from src.analysis.eda import run_eda_pipeline
from src.analysis.report_generator import generate_eda_report
from src.analysis.statistics import (
    compute_categorical_statistics,
    compute_correlation_matrix,
    compute_dataset_statistics,
    compute_numerical_statistics,
    compute_target_statistics,
)

__all__ = [
    "run_eda_pipeline",
    "generate_business_insights",
    "generate_eda_report",
    "compute_dataset_statistics",
    "compute_numerical_statistics",
    "compute_categorical_statistics",
    "compute_target_statistics",
    "compute_correlation_matrix",
]

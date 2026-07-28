"""
===============================================================================
Customer Churn Prediction Platform
Export Service
===============================================================================

Purpose
-------
Centralized export utilities for the application.

Responsibilities
----------------
• Export DataFrames
• Export prediction results
• Export evaluation metrics
• Export model comparison
• Export JSON
• Export Excel
• Export CSV
• Generate safe filenames

Notes
-----
This module contains no UI code.
It should be consumed by Streamlit pages and other services.
===============================================================================
"""

from __future__ import annotations

import io
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from configs.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_DATE_FORMAT = "%Y%m%d_%H%M%S"

CSV_MIME = "text/csv"

JSON_MIME = "application/json"

EXCEL_MIME = (
    "application/vnd.openxmlformats-officedocument."
    "spreadsheetml.sheet"
)

SUPPORTED_EXPORT_FORMATS = {
    "csv",
    "xlsx",
    "json",
}

class ExportServiceError(Exception):
    """Base export exception."""

class EmptyDataError(ExportServiceError):
    """Raised when attempting to export empty data."""

class UnsupportedFormatError(ExportServiceError):
    """Raised for unsupported export formats."""

def validate_dataframe(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate a DataFrame before exporting.

    Raises
    ------
    EmptyDataError
    """

    if dataframe is None:
        raise EmptyDataError(
            "DataFrame cannot be None."
        )

    if dataframe.empty:
        raise EmptyDataError(
            "DataFrame is empty."
        )

def validate_export_format(
    export_format: str,
) -> None:
    """
    Validate export format.
    """

    if export_format.lower() not in SUPPORTED_EXPORT_FORMATS:

        raise UnsupportedFormatError(
            f"Unsupported export format: "
            f"{export_format}"
        )

def sanitize_filename(
    filename: str,
) -> str:
    """
    Remove unsafe filename characters.
    """

    filename = filename.strip()

    filename = re.sub(
        r"[^\w\-_. ]",
        "_",
        filename,
    )

    filename = re.sub(
        r"\s+",
        "_",
        filename,
    )

    return filename


def generate_filename(
    prefix: str,
    extension: str,
    include_timestamp: bool = True,
) -> str:
    """
    Generate standardized filenames.

    Example
    -------
    churn_predictions_20260721_143015.csv
    """

    prefix = sanitize_filename(prefix)

    extension = extension.lower().replace(".", "")

    if include_timestamp:

        timestamp = datetime.now().strftime(
            DEFAULT_DATE_FORMAT
        )

        return (
            f"{prefix}_{timestamp}.{extension}"
        )

    return f"{prefix}.{extension}"

def dataframe_to_csv(
    dataframe: pd.DataFrame,
) -> bytes:
    """
    Convert DataFrame to CSV bytes.
    """

    validate_dataframe(dataframe)

    logger.info(
        "Exporting DataFrame to CSV."
    )

    return dataframe.to_csv(
        index=False,
    ).encode("utf-8")


def dataframe_to_json(
    dataframe: pd.DataFrame,
    indent: int = 4,
) -> bytes:
    """
    Convert DataFrame to JSON bytes.
    """

    validate_dataframe(dataframe)

    logger.info(
        "Exporting DataFrame to JSON."
    )

    records = dataframe.to_dict(
        orient="records"
    )

    return json.dumps(
        records,
        indent=indent,
        default=str,
    ).encode("utf-8")


def dataframe_to_excel(
    dataframe: pd.DataFrame,
) -> bytes:
    """
    Convert DataFrame to Excel bytes.
    """

    validate_dataframe(dataframe)

    logger.info(
        "Exporting DataFrame to Excel."
    )

    buffer = io.BytesIO()

    with pd.ExcelWriter(
        buffer,
        engine="openpyxl",
    ) as writer:

        dataframe.to_excel(
            writer,
            sheet_name="Export",
            index=False,
        )

    buffer.seek(0)

    return buffer.read()

def dictionary_to_json(
    data: dict[str, Any],
    indent: int = 4,
) -> bytes:
    """
    Serialize dictionary to JSON.
    """

    logger.info(
        "Exporting dictionary to JSON."
    )

    return json.dumps(
        data,
        indent=indent,
        default=str,
    ).encode("utf-8")

def export_predictions_csv(
    predictions: pd.DataFrame,
) -> tuple[bytes, str, str]:
    """
    Export prediction results as CSV.

    Returns
    -------
    tuple
        (
            file_bytes,
            filename,
            mime_type,
        )
    """

    logger.info("Preparing prediction CSV export.")

    return (
        dataframe_to_csv(predictions),
        generate_filename(
            prefix="churn_predictions",
            extension="csv",
        ),
        CSV_MIME,
    )

def export_predictions_excel(
    predictions: pd.DataFrame,
) -> tuple[bytes, str, str]:
    """
    Export prediction results as Excel.
    """

    logger.info("Preparing prediction Excel export.")

    return (
        dataframe_to_excel(predictions),
        generate_filename(
            prefix="churn_predictions",
            extension="xlsx",
        ),
        EXCEL_MIME,
    )


def export_predictions_json(
    predictions: pd.DataFrame,
) -> tuple[bytes, str, str]:
    """
    Export prediction results as JSON.
    """

    logger.info("Preparing prediction JSON export.")

    return (
        dataframe_to_json(predictions),
        generate_filename(
            prefix="churn_predictions",
            extension="json",
        ),
        JSON_MIME,
    )

def export_evaluation_metrics(
    metrics: dict[str, Any],
) -> tuple[bytes, str, str]:
    """
    Export evaluation metrics.
    """

    logger.info("Preparing evaluation metrics export.")

    return (
        dictionary_to_json(metrics),
        generate_filename(
            prefix="evaluation_metrics",
            extension="json",
        ),
        JSON_MIME,
    )

def export_model_comparison(
    comparison: pd.DataFrame,
) -> tuple[bytes, str, str]:
    """
    Export model comparison table.
    """

    logger.info(
        "Preparing model comparison export."
    )

    return (
        dataframe_to_csv(comparison),
        generate_filename(
            prefix="model_comparison",
            extension="csv",
        ),
        CSV_MIME,
    )

def export_classification_report(
    report: pd.DataFrame,
) -> tuple[bytes, str, str]:
    """
    Export classification report.
    """

    logger.info(
        "Preparing classification report export."
    )

    return (
        dataframe_to_csv(report),
        generate_filename(
            prefix="classification_report",
            extension="csv",
        ),
        CSV_MIME,
    )

def export_feature_importance(
    importance: pd.DataFrame,
) -> tuple[bytes, str, str]:
    """
    Export feature importance.
    """

    logger.info(
        "Preparing feature importance export."
    )

    return (
        dataframe_to_excel(
            importance,
        ),
        generate_filename(
            prefix="feature_importance",
            extension="xlsx",
        ),
        EXCEL_MIME,
    )

def export_shap_values(
    shap_dataframe: pd.DataFrame,
) -> tuple[bytes, str, str]:
    """
    Export SHAP values.
    """

    logger.info(
        "Preparing SHAP export."
    )

    return (
        dataframe_to_excel(
            shap_dataframe,
        ),
        generate_filename(
            prefix="shap_values",
            extension="xlsx",
        ),
        EXCEL_MIME,
    )

def export_dataframe(
    dataframe: pd.DataFrame,
    filename: str,
    export_format: str = "csv",
) -> tuple[bytes, str, str]:
    """
    Generic DataFrame export.

    Supported formats
    -----------------
    csv
    xlsx
    json
    """

    validate_export_format(
        export_format,
    )

    export_format = export_format.lower()

    if export_format == "csv":

        return (
            dataframe_to_csv(
                dataframe,
            ),
            generate_filename(
                filename,
                "csv",
            ),
            CSV_MIME,
        )

    if export_format == "xlsx":

        return (
            dataframe_to_excel(
                dataframe,
            ),
            generate_filename(
                filename,
                "xlsx",
            ),
            EXCEL_MIME,
        )

    return (
        dataframe_to_json(
            dataframe,
        ),
        generate_filename(
            filename,
            "json",
        ),
        JSON_MIME,
    )

def build_export_package(
    data: bytes,
    filename: str,
    mime_type: str,
) -> dict[str, Any]:
    """
    Build a standardized export package.

    Parameters
    ----------
    data:
        Serialized file bytes.

    filename:
        Output filename.

    mime_type:
        MIME type.

    Returns
    -------
    dict
        Standardized export metadata.
    """

    return {
        "data": data,
        "filename": filename,
        "mime_type": mime_type,
        "size": len(data),
    }

def prepare_dataframe_export(
    dataframe: pd.DataFrame,
    filename: str,
    export_format: str = "csv",
) -> dict[str, Any]:
    """
    Prepare DataFrame export package.
    """

    data, name, mime = export_dataframe(
        dataframe=dataframe,
        filename=filename,
        export_format=export_format,
    )

    return build_export_package(
        data=data,
        filename=name,
        mime_type=mime,
    )

def prepare_prediction_export(
    predictions: pd.DataFrame | Any,
) -> dict[str, Any]:
    """
    Prepare a prediction CSV export.

    The batch workflow supplies a DataFrame, while the single-customer page
    supplies a PredictionResult object. Normalize the latter to one row before
    it reaches the DataFrame-only CSV exporter.
    """

    if not isinstance(predictions, pd.DataFrame):
        customer_frame = getattr(predictions, "customer_frame", None)
        if customer_frame is None or customer_frame.empty:
            raise EmptyDataError("Prediction result does not contain customer data.")

        record = customer_frame.iloc[0].to_dict()
        for field in (
            "predicted_class",
            "predicted_probability",
            "confidence",
            "risk_level",
            "recommended_action",
            "prediction_timestamp",
        ):
            record[field] = getattr(predictions, field, None)
        predictions = pd.DataFrame([record])

    data, name, mime = export_predictions_csv(
        predictions,
    )

    return build_export_package(
        data,
        name,
        mime,
    )

def prepare_model_comparison_export(
    comparison: pd.DataFrame,
) -> dict[str, Any]:
    """
    Prepare model comparison export.
    """

    data, name, mime = export_model_comparison(
        comparison,
    )

    return build_export_package(
        data,
        name,
        mime,
    )

def prepare_metrics_export(
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """
    Prepare evaluation metrics export.
    """

    data, name, mime = export_evaluation_metrics(
        metrics,
    )

    return build_export_package(
        data,
        name,
        mime,
    )

def get_supported_export_formats() -> list[str]:
    """
    Return supported export formats.
    """

    return sorted(
        SUPPORTED_EXPORT_FORMATS,
    )


def get_export_statistics(
    package: dict[str, Any],
) -> dict[str, Any]:
    """
    Return metadata describing an export package.
    """

    return {
        "filename": package["filename"],
        "mime_type": package["mime_type"],
        "size_bytes": package["size"],
        "size_kb": round(
            package["size"] / 1024,
            2,
        ),
    }

def get_export_service_information() -> dict[str, Any]:
    """
    Return metadata about the export service.

    Returns
    -------
    dict
        Service metadata.
    """

    return {
        "service": "Export Service",
        "version": "1.0.0",
        "supported_formats": sorted(
            SUPPORTED_EXPORT_FORMATS
        ),
        "default_date_format": DEFAULT_DATE_FORMAT,
    }

def log_export_success(
    filename: str,
    size_bytes: int,
) -> None:
    """
    Log a successful export operation.

    Parameters
    ----------
    filename:
        Generated filename.

    size_bytes:
        Export size.
    """

    logger.info(
        "Successfully exported '%s' (%d bytes).",
        filename,
        size_bytes,
    )

def log_export_failure(
    operation: str,
    exception: Exception,
) -> None:
    """
    Log an export failure.

    Parameters
    ----------
    operation:
        Export operation.

    exception:
        Raised exception.
    """

    logger.exception(
        "Export operation '%s' failed.",
        operation,
        exc_info=exception,
    )

__all__ = [
    "validate_dataframe",
    "validate_export_format",

    "sanitize_filename",
    "generate_filename",

    "dataframe_to_csv",
    "dataframe_to_json",
    "dataframe_to_excel",
    "dictionary_to_json",

    "export_dataframe",

    "export_predictions_csv",
    "export_predictions_excel",
    "export_predictions_json",

    "export_evaluation_metrics",
    "export_model_comparison",
    "export_classification_report",

    "export_feature_importance",
    "export_shap_values",

    "build_export_package",
    "prepare_dataframe_export",
    "prepare_prediction_export",
    "prepare_model_comparison_export",
    "prepare_metrics_export",

    "get_supported_export_formats",
    "get_export_statistics",
    "get_export_service_information",

    "log_export_success",
    "log_export_failure",

    "ExportServiceError",
    "EmptyDataError",
    "UnsupportedFormatError",
]

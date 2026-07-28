"""Feature metadata generation.

Captures everything downstream milestones (model training, SHAP
explainability, the Streamlit dashboard) will need to know about the
fitted preprocessing pipeline: how many features went in and came
out, what each engineered feature is, and what categories each
one-hot-encoded column learned.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sklearn.pipeline import Pipeline

from configs.config import settings
from configs.logging_config import get_logger
from src.preprocessing.feature_engineering import ENGINEERED_FEATURE_NAMES
from src.preprocessing.transformer import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES

logger = get_logger(__name__)


def extract_categorical_mappings(pipeline: Pipeline) -> dict[str, list[str]]:
    """Extract the learned categories for each one-hot-encoded feature.

    Args:
        pipeline: A fitted preprocessing pipeline, as built by
            :func:`src.preprocessing.pipeline.build_preprocessing_pipeline`.

    Returns:
        A mapping of categorical column name to the list of category
        values the encoder learned during fitting, in encoding order.
    """
    column_transformer = pipeline.named_steps["column_transformer"]
    encoder = column_transformer.named_transformers_["categorical"].named_steps["encoder"]
    return {
        column: categories.tolist()
        for column, categories in zip(CATEGORICAL_FEATURES, encoder.categories_, strict=True)
    }


def build_feature_metadata(
    pipeline: Pipeline, input_feature_names: list[str]
) -> dict[str, Any]:
    """Assemble structured metadata describing a fitted preprocessing pipeline.

    Args:
        pipeline: A fitted preprocessing pipeline.
        input_feature_names: Names of the raw input columns the
            pipeline was fit on (before cleaning/engineering).

    Returns:
        A dictionary with input/output feature counts, transformed
        feature names, the engineered feature list, feature-group
        membership, and categorical mappings.
    """
    column_transformer = pipeline.named_steps["column_transformer"]
    output_feature_names = column_transformer.get_feature_names_out().tolist()

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "num_input_features": len(input_feature_names),
        "input_feature_names": input_feature_names,
        "num_output_features": len(output_feature_names),
        "output_feature_names": output_feature_names,
        "engineered_feature_names": list(ENGINEERED_FEATURE_NAMES),
        "numerical_features": list(NUMERICAL_FEATURES),
        "binary_features": list(BINARY_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "categorical_mappings": extract_categorical_mappings(pipeline),
    }


def save_feature_metadata(
    metadata: dict[str, Any],
    json_path: Path = settings.feature_metadata_path,
    summary_path: Path = settings.preprocessing_summary_path,
) -> dict[str, Path]:
    """Save feature metadata as JSON and a human-readable Markdown summary.

    Args:
        metadata: Output of :func:`build_feature_metadata`.
        json_path: Destination for the JSON metadata file. Defaults to
            ``settings.feature_metadata_path``.
        summary_path: Destination for the Markdown summary file.
            Defaults to ``settings.preprocessing_summary_path``.

    Returns:
        A mapping with keys "json" and "summary_markdown" pointing to
        the saved file paths.
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    summary_path.write_text(_render_summary_markdown(metadata), encoding="utf-8")

    logger.info("Metadata saved: %s, %s", json_path, summary_path)
    return {"json": json_path, "summary_markdown": summary_path}


def _render_summary_markdown(metadata: dict[str, Any]) -> str:
    """Render the preprocessing summary as a Markdown document.

    Args:
        metadata: Output of :func:`build_feature_metadata`.

    Returns:
        The complete Markdown document as a single string.
    """
    lines = [
        f"# Preprocessing Summary — {settings.project_name}",
        f"\nGenerated at: {metadata['generated_at']}\n",
        "## Overview\n",
        f"- Input features: {metadata['num_input_features']}",
        f"- Output features (after encoding): {metadata['num_output_features']}",
        "",
        "## Preprocessing Steps\n",
        "1. Data cleaning (identifier removal, numeric coercion, whitespace trimming)",
        "2. Feature engineering (see below)",
        "3. Missing value imputation (median for numerical, most-frequent for binary/categorical)",
        "4. Categorical encoding (one-hot, unseen categories ignored at inference)",
        "5. Numerical scaling (StandardScaler; binary and encoded features excluded)",
        "",
        "## Engineered Features\n",
    ]
    lines.extend(f"- {name}" for name in metadata["engineered_feature_names"])
    lines.append("")
    lines.append("## Feature Groups\n")
    lines.append(f"- Numerical: {', '.join(metadata['numerical_features'])}")
    lines.append(f"- Binary: {', '.join(metadata['binary_features'])}")
    lines.append(f"- Categorical: {', '.join(metadata['categorical_features'])}")
    lines.append("")
    return "\n".join(lines)

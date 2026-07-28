"""The complete preprocessing pipeline.

Combines cleaning, feature engineering, and the ``ColumnTransformer``
into a single scikit-learn ``Pipeline`` that is reusable for training,
validation, testing, and inference — with no preprocessing logic
duplicated elsewhere.

Data-leakage prevention: this module never decides what data to fit
on. :func:`build_preprocessing_pipeline` only constructs an unfitted
pipeline; the caller (this milestone's demonstration code, or
Milestone 6's training script) is responsible for calling
:func:`fit_preprocessor` on the training split only.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from configs.config import settings
from configs.logging_config import get_logger
from src.preprocessing.cleaner import apply_row_level_cleaning
from src.preprocessing.feature_engineering import engineer_features
from src.preprocessing.transformer import build_transformer

logger = get_logger(__name__)


def build_preprocessing_pipeline() -> Pipeline:
    """Build the full, unfitted preprocessing pipeline.

    Stages: row-level cleaning -> feature engineering ->
    ColumnTransformer (imputation, scaling, encoding). Row-level
    cleaning never changes the number of rows, so this pipeline is
    safe to reuse on a single inference record as well as a full
    batch.

    Note:
        Batch-level cleaning (removing exact duplicate rows) is
        intentionally excluded here since it changes row count and is
        unsafe inside a scikit-learn Pipeline used for single-record
        inference. Call :func:`src.preprocessing.cleaner.clean_dataset`
        once on the full historical dataset before fitting this
        pipeline.

    Returns:
        An unfitted scikit-learn Pipeline.
    """
    return Pipeline(
        steps=[
            ("cleaning", FunctionTransformer(apply_row_level_cleaning, validate=False)),
            ("feature_engineering", FunctionTransformer(engineer_features, validate=False)),
            ("column_transformer", build_transformer()),
        ]
    )


def fit_preprocessor(pipeline: Pipeline, dataframe: pd.DataFrame) -> Pipeline:
    """Fit the preprocessing pipeline on a feature DataFrame.

    Args:
        pipeline: An unfitted pipeline from
            :func:`build_preprocessing_pipeline`.
        dataframe: Feature data to fit on. To prevent leakage, this
            must be the training split only — never the full dataset
            or a validation/test split. (This milestone does not
            perform the train/test split itself; that is Milestone
            6's responsibility.)

    Returns:
        The same pipeline object, now fitted in place.
    """
    logger.info("Preprocessing started.")
    logger.info("Fitting preprocessor on %d row(s), %d column(s).", *dataframe.shape)
    pipeline.fit(dataframe)
    logger.info("Pipeline built and fitted.")
    return pipeline


def transform_dataset(pipeline: Pipeline, dataframe: pd.DataFrame) -> pd.DataFrame:
    """Transform a feature DataFrame using an already-fitted pipeline.

    Reusable for training, validation, testing, and inference data —
    the same fitted pipeline is always used, never refit.

    Args:
        pipeline: A fitted pipeline, as returned by
            :func:`fit_preprocessor` or :func:`load_preprocessor`.
        dataframe: Feature data to transform.

    Returns:
        A DataFrame of processed features with meaningful column
        names, drawn from the ColumnTransformer's
        ``get_feature_names_out``.
    """
    transformed_array = pipeline.transform(dataframe)
    feature_names = pipeline.named_steps["column_transformer"].get_feature_names_out()
    transformed_frame = pd.DataFrame(
        transformed_array, columns=feature_names, index=dataframe.index
    )
    logger.info(
        "Feature transformation completed: %d output feature(s).", transformed_frame.shape[1]
    )
    return transformed_frame


def save_preprocessor(pipeline: Pipeline, path: Path = settings.preprocessor_path) -> Path:
    """Serialize a fitted pipeline to disk using Joblib.

    Args:
        pipeline: A fitted pipeline to serialize.
        path: Destination path. Defaults to ``settings.preprocessor_path``.

    Returns:
        The path the pipeline was saved to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info("Pipeline serialized to %s.", path)
    return path


def load_preprocessor(path: Path = settings.preprocessor_path) -> Pipeline:
    """Load a previously fitted and serialized preprocessing pipeline.

    Future modules (model training, inference, Streamlit) should use
    this instead of rebuilding and refitting a pipeline.

    Args:
        path: Location of the serialized pipeline. Defaults to
            ``settings.preprocessor_path``.

    Returns:
        The deserialized, fitted Pipeline, ready for ``.transform()``.

    Raises:
        FileNotFoundError: If no file exists at ``path``.
    """
    if not path.exists():
        raise FileNotFoundError(f"Serialized preprocessor not found: {path}")
    pipeline = joblib.load(path)
    logger.info("Pipeline loaded from %s.", path)
    return pipeline

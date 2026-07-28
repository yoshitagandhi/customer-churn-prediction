"""Dataset loading utilities.

This module is responsible only for loading the raw dataset from disk
into a Pandas DataFrame. It does not validate the dataset's contents
(see :mod:`src.data.validator`) and does not modify the data in any way.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from configs.config import settings
from configs.logging_config import get_logger
from src.utils.exceptions import DataValidationError, FileFormatError
from src.utils.helpers import verify_path_exists

logger = get_logger(__name__)


def load_dataset(file_path: Path | None = None) -> pd.DataFrame:
    """Load the raw customer churn dataset from a CSV file.

    Args:
        file_path:
            Optional dataset path. When omitted,
            ``settings.default_dataset_path`` is used.

    Returns:
        The raw dataset as a Pandas DataFrame.

    Raises:
        FileNotFoundError:
            If the dataset path does not exist.

        FileFormatError:
            If the dataset extension is unsupported.

        DataValidationError:
            If the dataset is empty or cannot be parsed.
    """
    dataset_path = file_path or settings.default_dataset_path
    dataset_path = dataset_path.expanduser().resolve()

    logger.info("Loading dataset from '%s'.", dataset_path)

    verify_path_exists(
        dataset_path,
        description="Dataset file",
        must_be_file=True,
    )

    _validate_file_extension(dataset_path)

    try:
        dataframe = pd.read_csv(dataset_path)

    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(
            f"Dataset file is empty: {dataset_path}"
        ) from exc

    except pd.errors.ParserError as exc:
        raise DataValidationError(
            f"Failed to parse CSV file: {dataset_path}"
        ) from exc

    except OSError as exc:
        raise DataValidationError(
            f"Unable to read dataset: {dataset_path}"
        ) from exc

    if dataframe.empty:
        raise DataValidationError(
            f"Dataset contains no records: {dataset_path}"
        )

    logger.info(
        "Dataset loaded successfully (%d rows, %d columns).",
        dataframe.shape[0],
        dataframe.shape[1],
    )

    logger.debug(
        "Dataset columns: %s",
        list(dataframe.columns),
    )

    return dataframe


def _validate_file_extension(
    file_path: Path,
    *,
    supported_extensions: tuple[str, ...] = settings.supported_file_extensions,
) -> None:
    """Validate that the dataset file extension is supported.

    Args:
        file_path:
            Dataset path.

        supported_extensions:
            Allowed file extensions.

    Raises:
        FileFormatError:
            If the file extension is unsupported.
    """
    suffix = file_path.suffix.lower()

    if suffix not in supported_extensions:
        supported = ", ".join(sorted(supported_extensions))

        raise FileFormatError(
            f"Unsupported dataset format '{suffix}'. "
            f"Supported formats: {supported}."
        )
"""Dataset validation against the expected schema.

This module checks whether a loaded dataset conforms to the schema
defined in :mod:`src.data.schema`. It only detects and reports
issues — it never modifies, imputes, or transforms the dataset.
Critical structural problems (an empty dataset, missing required
columns, or an invalid target column) raise
:class:`~src.utils.exceptions.DataValidationError`. Everything else
(missing values, duplicate rows, datatype mismatches) is reported as
a warning so downstream milestones can decide how to handle it.
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from configs.config import settings
from configs.logging_config import get_logger
from src.data.schema import (
    CUSTOMER_ID_COLUMN,
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
    VALID_TARGET_LABELS,
)
from src.utils.exceptions import DataValidationError

logger = get_logger(__name__)

_SEVERITY_PASSED = "PASSED"
_SEVERITY_WARNING = "WARNING"
_SEVERITY_ERROR = "ERROR"


@dataclass(frozen=True)
class ValidationIssue:
    """A single validation finding.

    Attributes:
        check_name: Short identifier of the check that produced this
            issue (e.g., "missing_values", "duplicate_rows").
        severity: One of "PASSED", "WARNING", or "ERROR".
        message: Human-readable description of the finding.
        details: Structured data supporting the finding (e.g., which
            columns are affected and by how much).
    """

    check_name: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Aggregated outcome of running all validation checks.

    Attributes:
        is_valid: True if no ERROR-severity issues were found.
        issues: Every issue produced by the validation checks,
            regardless of severity.
    """

    is_valid: bool
    issues: list[ValidationIssue]

    @property
    def passed(self) -> list[ValidationIssue]:
        """Return issues that represent successfully passed checks."""
        return [issue for issue in self.issues if issue.severity == _SEVERITY_PASSED]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Return issues that represent non-critical problems."""
        return [issue for issue in self.issues if issue.severity == _SEVERITY_WARNING]

    @property
    def errors(self) -> list[ValidationIssue]:
        """Return issues that represent critical problems."""
        return [issue for issue in self.issues if issue.severity == _SEVERITY_ERROR]

    def get_issue(self, check_name: str) -> ValidationIssue | None:
        """Return the first issue matching a given check name, if any.

        Args:
            check_name: The check identifier to look up.

        Returns:
            The matching :class:`ValidationIssue`, or None if not found.
        """
        return next((issue for issue in self.issues if issue.check_name == check_name), None)


def validate_shape(dataframe: pd.DataFrame) -> ValidationIssue:
    """Verify the dataset has at least one row and one column.

    Args:
        dataframe: The dataset to check.

    Returns:
        A ValidationIssue describing whether the dataset is non-empty.
    """
    num_rows, num_columns = dataframe.shape
    if num_rows == 0 or num_columns == 0:
        return ValidationIssue(
            check_name="dataset_shape",
            severity=_SEVERITY_ERROR,
            message=f"Dataset is empty: {num_rows} rows, {num_columns} columns.",
            details={"rows": num_rows, "columns": num_columns},
        )
    return ValidationIssue(
        check_name="dataset_shape",
        severity=_SEVERITY_PASSED,
        message=f"Dataset has {num_rows} rows and {num_columns} columns.",
        details={
            "rows": num_rows,
            "columns": num_columns,
            "memory_usage_bytes": int(dataframe.memory_usage(deep=True).sum()),
        },
    )


def validate_columns(
    dataframe: pd.DataFrame, required_columns: tuple[str, ...] = REQUIRED_COLUMNS
) -> list[ValidationIssue]:
    """Verify required columns are present and column names are unique.

    Args:
        dataframe: The dataset to check.
        required_columns: Column names that must be present. Defaults
            to the schema's required columns.

    Returns:
        A list of ValidationIssue objects, one per sub-check.
    """
    issues: list[ValidationIssue] = []

    missing_columns = sorted(set(required_columns) - set(dataframe.columns))
    if missing_columns:
        issues.append(
            ValidationIssue(
                check_name="required_columns",
                severity=_SEVERITY_ERROR,
                message=f"Missing required columns: {missing_columns}",
                details={"missing_columns": missing_columns},
            )
        )
    else:
        issues.append(
            ValidationIssue(
                check_name="required_columns",
                severity=_SEVERITY_PASSED,
                message="All required columns are present.",
            )
        )

    duplicate_columns = dataframe.columns[dataframe.columns.duplicated()].unique().tolist()
    if duplicate_columns:
        issues.append(
            ValidationIssue(
                check_name="duplicate_column_names",
                severity=_SEVERITY_ERROR,
                message=f"Duplicate column names found: {duplicate_columns}",
                details={"duplicate_columns": duplicate_columns},
            )
        )
    else:
        issues.append(
            ValidationIssue(
                check_name="duplicate_column_names",
                severity=_SEVERITY_PASSED,
                message="No duplicate column names found.",
            )
        )

    return issues


def validate_datatypes(
    dataframe: pd.DataFrame, numeric_columns: tuple[str, ...] = NUMERIC_COLUMNS
) -> ValidationIssue:
    """Detect columns whose actual dtype does not match the expected category.

    This only reports mismatches (e.g., a numeric column stored as
    text). It never converts or coerces the data.

    Args:
        dataframe: The dataset to check.
        numeric_columns: Columns expected to hold numeric data.
            Defaults to the schema's numeric columns.

    Returns:
        A ValidationIssue summarizing any datatype mismatches found.
    """
    mismatches: dict[str, str] = {}
    for column in numeric_columns:
        if column not in dataframe.columns:
            continue
        if not pd.api.types.is_numeric_dtype(dataframe[column]):
            mismatches[column] = str(dataframe[column].dtype)

    if mismatches:
        return ValidationIssue(
            check_name="datatype_consistency",
            severity=_SEVERITY_WARNING,
            message=f"Columns expected to be numeric are stored as non-numeric: {list(mismatches)}",
            details={"mismatched_columns": mismatches},
        )
    return ValidationIssue(
        check_name="datatype_consistency",
        severity=_SEVERITY_PASSED,
        message="All expected-numeric columns have numeric dtypes.",
    )


def validate_duplicates(
    dataframe: pd.DataFrame, id_column: str = CUSTOMER_ID_COLUMN
) -> list[ValidationIssue]:
    """Detect fully duplicated rows and duplicated identifier values.

    Duplicates are reported, not removed.

    Args:
        dataframe: The dataset to check.
        id_column: Name of the unique identifier column, if present.
            Defaults to the schema's customer ID column.

    Returns:
        A list of ValidationIssue objects, one per sub-check.
    """
    issues: list[ValidationIssue] = []

    duplicate_row_count = int(dataframe.duplicated().sum())
    if duplicate_row_count > 0:
        issues.append(
            ValidationIssue(
                check_name="duplicate_rows",
                severity=_SEVERITY_WARNING,
                message=f"Found {duplicate_row_count} fully duplicated row(s).",
                details={"duplicate_row_count": duplicate_row_count},
            )
        )
    else:
        issues.append(
            ValidationIssue(
                check_name="duplicate_rows",
                severity=_SEVERITY_PASSED,
                message="No fully duplicated rows found.",
                details={"duplicate_row_count": 0},
            )
        )

    if id_column in dataframe.columns:
        duplicate_id_count = int(dataframe[id_column].duplicated().sum())
        if duplicate_id_count > 0:
            issues.append(
                ValidationIssue(
                    check_name="duplicate_identifiers",
                    severity=_SEVERITY_WARNING,
                    message=f"Found {duplicate_id_count} duplicate '{id_column}' value(s).",
                    details={"duplicate_id_count": duplicate_id_count, "id_column": id_column},
                )
            )
        else:
            issues.append(
                ValidationIssue(
                    check_name="duplicate_identifiers",
                    severity=_SEVERITY_PASSED,
                    message=f"No duplicate '{id_column}' values found.",
                    details={"duplicate_id_count": 0, "id_column": id_column},
                )
            )

    return issues


def validate_missing_values(dataframe: pd.DataFrame) -> ValidationIssue:
    """Calculate missing value counts and percentages per column.

    Args:
        dataframe: The dataset to check.

    Returns:
        A ValidationIssue summarizing missing values across all
        affected columns. WARNING severity if any column has missing
        values, PASSED otherwise.
    """
    missing_counts = dataframe.isna().sum()
    affected_columns = missing_counts[missing_counts > 0]

    if affected_columns.empty:
        return ValidationIssue(
            check_name="missing_values",
            severity=_SEVERITY_PASSED,
            message="No missing values found.",
            details={},
        )

    total_rows = len(dataframe)
    missing_summary = {
        str(column): {
            "missing_count": int(count),
            "missing_percentage": round(float(count) / total_rows * 100, 2),
        }
        for column, count in affected_columns.items()
    }
    return ValidationIssue(
        check_name="missing_values",
        severity=_SEVERITY_WARNING,
        message=(
            f"Missing values found in {len(missing_summary)} column(s): {list(missing_summary)}"
        ),
        details=missing_summary,
    )


def validate_target(
    dataframe: pd.DataFrame,
    target_column: str = settings.target_column,
    valid_labels: tuple[str, ...] = VALID_TARGET_LABELS,
) -> ValidationIssue:
    """Validate the target column exists, is binary, and holds valid labels.

    Args:
        dataframe: The dataset to check.
        target_column: Name of the target column. Defaults to
            ``settings.target_column``.
        valid_labels: The set of labels the target column is allowed
            to contain. Defaults to the schema's valid target labels.

    Returns:
        A ValidationIssue describing the target column's distribution.

    Raises:
        DataValidationError: If the target column is missing or
            contains labels outside ``valid_labels``.
    """
    if target_column not in dataframe.columns:
        raise DataValidationError(f"Target column '{target_column}' is missing from the dataset.")

    observed_labels = set(dataframe[target_column].dropna().unique())
    unexpected_labels = observed_labels - set(valid_labels)
    if unexpected_labels:
        raise DataValidationError(
            f"Target column '{target_column}' contains unexpected labels: "
            f"{sorted(unexpected_labels)}. Expected one of: {valid_labels}."
        )

    value_counts = dataframe[target_column].value_counts(dropna=False)
    total = len(dataframe)
    distribution = {
        str(label): {
            "count": int(count),
            "percentage": round(float(count) / total * 100, 2),
        }
        for label, count in value_counts.items()
    }

    return ValidationIssue(
        check_name="target_validation",
        severity=_SEVERITY_PASSED,
        message=f"Target column '{target_column}' is valid with distribution: {distribution}",
        details={"target_column": target_column, "distribution": distribution},
    )


def validate_dataset(
    dataframe: pd.DataFrame,
    target_column: str = settings.target_column,
) -> ValidationResult:
    """Run the full validation suite against a dataset.

    Orchestrates shape, column, datatype, duplicate, missing-value,
    and target validation. Critical issues raise
    :class:`~src.utils.exceptions.DataValidationError` immediately;
    non-critical issues are collected and returned for reporting.

    Args:
        dataframe: The dataset to validate.
        target_column: Name of the target column. Defaults to
            ``settings.target_column``.

    Returns:
        A ValidationResult aggregating every check performed.

    Raises:
        DataValidationError: If the dataset is empty, is missing
            required columns, has duplicate column names, or has an
            invalid target column.
    """
    logger.info("Validation started.")
    issues: list[ValidationIssue] = []

    shape_issue = validate_shape(dataframe)
    issues.append(shape_issue)
    if shape_issue.severity == _SEVERITY_ERROR:
        raise DataValidationError(shape_issue.message)

    column_issues = validate_columns(dataframe)
    issues.extend(column_issues)
    critical_column_issues = [issue for issue in column_issues if issue.severity == _SEVERITY_ERROR]
    if critical_column_issues:
        raise DataValidationError("; ".join(issue.message for issue in critical_column_issues))

    issues.append(validate_datatypes(dataframe))
    issues.extend(validate_duplicates(dataframe))
    issues.append(validate_missing_values(dataframe))

    # Raises DataValidationError internally if the target is invalid.
    issues.append(validate_target(dataframe, target_column=target_column))

    warning_count = sum(1 for issue in issues if issue.severity == _SEVERITY_WARNING)
    if warning_count:
        logger.warning("Validation completed with %d warning(s).", warning_count)
    else:
        logger.info("Validation completed with no warnings.")

    return ValidationResult(is_valid=True, issues=issues)

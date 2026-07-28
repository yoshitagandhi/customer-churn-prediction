"""Custom exception classes used across the project.

Defining explicit exception types (instead of raising generic
``Exception`` or ``ValueError``) makes failures easier to catch,
log, and reason about as the project grows across milestones.
"""


class ChurnPredictionError(Exception):
    """Base exception for all project-specific errors.

    All custom exceptions in this project should inherit from this
    class so callers can catch project-specific errors with a single
    ``except ChurnPredictionError`` clause when needed.
    """


class ConfigurationError(ChurnPredictionError):
    """Raised when the project configuration is invalid or missing."""


class DataValidationError(ChurnPredictionError):
    """Raised when input data fails validation checks."""


class FileFormatError(ChurnPredictionError):
    """Raised when a file has an unsupported or unexpected format."""


class ModelTrainingError(ChurnPredictionError):
    """Raised when a model fails to train successfully."""


class PredictionError(ChurnPredictionError):
    """Raised when a model fails to produce a valid prediction."""

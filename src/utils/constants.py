"""Project-wide constants.

Centralizing constants here avoids magic numbers and duplicated
literal values scattered across the codebase. Future modules should
import from this file rather than redefining these values locally.
"""

from typing import Final

# Global random seed used wherever reproducibility is required
# (e.g., train/test splitting, model initialization).
RANDOM_STATE: Final[int] = 42

# Name of the target column in the Telco Customer Churn dataset.
TARGET_COLUMN: Final[str] = "Churn"

# Human-readable project name, used in logs, reports, and the CLI.
PROJECT_NAME: Final[str] = "Customer Churn Prediction"

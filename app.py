"""
Customer Churn Prediction Platform
Command-Line Interface

Application entry point for the customer churn training workflow.

Responsibilities
----------------
• Parse command-line arguments
• Validate repository structure
• Execute the training pipeline
• Display execution summary

Business logic is delegated to ``src.pipeline``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from configs.config import settings
from configs.logging_config import get_logger
from configs.paths import (
    FIGURES_DIR,
    LOGS_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
)
from src.pipeline import run_training_workflow

logger = get_logger(__name__)

_REQUIRED_DIRECTORIES: tuple[Path, ...] = (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    LOGS_DIR,
)

def verify_repository_structure() -> bool:
    """
    Verify that the required project directories exist.

    Returns
    -------
    bool
        True when all required directories exist.
    """

    valid = True

    for directory in _REQUIRED_DIRECTORIES:

        if directory.exists():
            logger.debug("Verified directory: %s", directory)
            continue

        logger.warning("Missing directory: %s", directory)
        valid = False

    return valid

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Train and evaluate customer churn models.",
    )

    parser.add_argument(
        "--data",
        type=Path,
        help="Path to the Telco Customer Churn dataset.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        help="Registered models to train.",
    )

    parser.add_argument(
        "--sampling",
        default=settings.default_sampling_strategy,
        help="Sampling strategy.",
    )

    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Disable hyperparameter optimization.",
    )

    return parser.parse_args()

class ChurnCLI:
    """
    Customer Churn command-line application.
    """

    def __init__(self) -> None:
        self.arguments = parse_arguments()

    def show_startup(self) -> None:
        """
        Display application information.
        """

        logger.info(
            "Starting %s v%s",
            settings.project_name,
            settings.version,
        )

        logger.info(
            "Project root: %s",
            PROJECT_ROOT,
        )

        structure_valid = verify_repository_structure()

        print(f"{settings.project_name} v{settings.version}")
        print(f"Random seed: {settings.random_seed}")
        print(f"Repository structure valid: {structure_valid}")

    def show_summary(
        self,
        result: dict[str, Any],
    ) -> None:
        """
        Display workflow results.
        """

        best_model = (
            result["evaluation_result"]
            ["best_model_info"]
            ["model_name"]
        )

        threshold = (
            result["threshold_result"]
            ["optimal_threshold"]
        )

        print(f"Best model: {best_model}")
        print(
            f"Recommended churn threshold: "
            f"{threshold:.2f}"
        )

    def run(self) -> None:
        """
        Execute the training workflow.
        """

        self.show_startup()

        if self.arguments.data is None:
            print(
                "Provide --data <path-to-telco-csv> "
                "to train models."
            )
            return

        result = run_training_workflow(
            self.arguments.data,
            model_names=(
                tuple(self.arguments.models)
                if self.arguments.models
                else None
            ),
            sampling_strategy=self.arguments.sampling,
            tune=not self.arguments.no_tune,
        )

        self.show_summary(result)

def main() -> None:
    """
    Application entry point.
    """

    ChurnCLI().run()


if __name__ == "__main__":
    main()
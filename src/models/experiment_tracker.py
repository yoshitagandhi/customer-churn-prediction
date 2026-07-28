"""Experiment tracking.

Records every training run in a structured, JSON-serializable format
— no external tracking service (e.g., MLflow) required.
"""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from configs.config import settings
from configs.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ExperimentRecord:
    """A single training run's full configuration and outcome.

    Attributes:
        experiment_id: Unique identifier for this run.
        model_name: Registered model name.
        sampling_strategy: Sampling strategy used.
        preprocessing_version: Project version the preprocessing
            pipeline came from, for traceability across code changes.
        hyperparameters: The model's parameter configuration used.
        validation_metrics: Metrics computed on the validation fold.
        training_time_seconds: Wall-clock training duration.
        timestamp: When this experiment was recorded (UTC, ISO 8601).
        random_seed: Random seed used for this run.
    """

    experiment_id: str
    model_name: str
    sampling_strategy: str
    preprocessing_version: str
    hyperparameters: dict[str, Any]
    validation_metrics: dict[str, float]
    training_time_seconds: float
    timestamp: str
    random_seed: int
    notes: str = field(default="")

    def to_dict(self) -> dict[str, Any]:
        """Return this record as a plain, JSON-serializable dictionary."""
        return asdict(self)


def log_experiment(
    model_name: str,
    sampling_strategy: str,
    hyperparameters: dict[str, Any],
    validation_metrics: dict[str, float],
    training_time_seconds: float,
    preprocessing_version: str = settings.version,
    random_state: int = settings.random_seed,
    notes: str = "",
) -> ExperimentRecord:
    """Create a structured record for a single training run.

    Args:
        model_name: Registered model name.
        sampling_strategy: Sampling strategy used.
        hyperparameters: The model's parameter configuration used.
        validation_metrics: Metrics computed on the validation fold.
        training_time_seconds: Wall-clock training duration.
        preprocessing_version: Project version the preprocessing
            pipeline came from. Defaults to ``settings.version``.
        random_state: Random seed used for this run. Defaults to
            ``settings.random_seed``.
        notes: Optional free-text context about this run.

    Returns:
        The newly created ExperimentRecord.
    """
    experiment_id = f"{model_name}_{sampling_strategy}_{uuid.uuid4().hex[:8]}"
    record = ExperimentRecord(
        experiment_id=experiment_id,
        model_name=model_name,
        sampling_strategy=sampling_strategy,
        preprocessing_version=preprocessing_version,
        hyperparameters=hyperparameters,
        validation_metrics=validation_metrics,
        training_time_seconds=round(training_time_seconds, 4),
        timestamp=datetime.now(UTC).isoformat(),
        random_seed=random_state,
        notes=notes,
    )
    logger.info("Experiment recorded: %s.", experiment_id)
    return record


def save_experiment_log(
    records: list[ExperimentRecord], path: Path = settings.experiment_log_path
) -> Path:
    """Persist every experiment record to a single JSON log file.

    Args:
        records: Experiment records to save.
        path: Destination path. Defaults to
            ``settings.experiment_log_path``.

    Returns:
        The path the log was saved to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([record.to_dict() for record in records], indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Experiment log saved to %s.", path)
    return path


def load_experiment_log(path: Path = settings.experiment_log_path) -> list[dict[str, Any]]:
    """Load a previously saved experiment log.

    Args:
        path: Location of the experiment log. Defaults to
            ``settings.experiment_log_path``.

    Returns:
        A list of experiment record dictionaries, or an empty list if
        no log file exists yet.
    """
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))

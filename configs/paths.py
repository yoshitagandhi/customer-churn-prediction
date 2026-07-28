"""Centralized filesystem path management.

This module defines every filesystem path used throughout the project
as a single source of truth. All other modules should import paths
from here instead of constructing them manually, which keeps the
project portable across operating systems and machines.

All paths are built with :class:`pathlib.Path` and directories that
are expected to hold generated artifacts (data, reports, models, logs)
are created automatically on import if they do not already exist.
"""

from pathlib import Path

# Root of the repository, resolved relative to this file's location so
# the project works correctly regardless of the current working
# directory it is executed from.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# --- Data directories ------------------------------------------------
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DATA_DIR: Path = DATA_DIR / "external"

# --- Modeling and reporting directories -------------------------------
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
METRICS_DIR: Path = REPORTS_DIR / "metrics"

# --- Supporting directories --------------------------------------------
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
ASSETS_DIR: Path = PROJECT_ROOT / "assets"

# Directories that must exist for the application to run correctly.
# These are created automatically so a fresh clone of the repository
# works out of the box without manual setup steps.
_DIRECTORIES_TO_ENSURE: tuple[Path, ...] = (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    EXTERNAL_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    NOTEBOOKS_DIR,
    LOGS_DIR,
)


def ensure_project_directories(directories: tuple[Path, ...] = _DIRECTORIES_TO_ENSURE) -> None:
    """Create all required project directories if they do not exist.

    Args:
        directories: A collection of directory paths to create. Defaults
            to the standard set of project directories required for the
            application to run.

    Returns:
        None.
    """
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# Ensure the directory structure is in place as soon as this module is
# imported, so downstream modules can safely read from or write to
# these locations without additional setup.
ensure_project_directories()

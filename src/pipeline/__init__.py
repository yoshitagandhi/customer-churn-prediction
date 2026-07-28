"""End-to-end orchestration for the customer churn workflow."""

from .training import run_training_workflow

__all__ = ["run_training_workflow"]
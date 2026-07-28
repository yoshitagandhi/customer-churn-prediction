"""Tests for business-threshold behavior used by the training workflow."""

import numpy as np
import pandas as pd
import pytest

from src.pipeline.training import _validate_class_balance, _validate_test_size
from src.threshold.decision_engine import generate_recommendations
from src.threshold.optimizer import optimize_threshold


def test_optimize_threshold_selects_lowest_cost_candidate() -> None:
    """The configured minimum-cost objective should return a tested threshold."""
    target_true = np.array([0, 0, 1, 1])
    target_proba = np.array([0.1, 0.4, 0.6, 0.9])

    result = optimize_threshold(
        target_true,
        target_proba,
        threshold_range=(0.2, 0.8),
        threshold_step=0.2,
    )

    minimum_cost = result["evaluation_table"]["business_cost"].min()
    assert result["metrics_at_optimal_threshold"]["business_cost"] == minimum_cost
    assert result["optimal_threshold"] in {0.2, 0.4, 0.6, 0.8}


def test_recommendations_keep_customer_ids_and_apply_threshold() -> None:
    """Customer recommendations must remain aligned with their probabilities."""
    probabilities = pd.Series([0.15, 0.75], index=pd.Index(["C-1", "C-2"]))

    recommendations = generate_recommendations(probabilities, optimal_threshold=0.5)

    assert recommendations["customer_id"].tolist() == ["C-1", "C-2"]
    assert recommendations["predicted_class"].tolist() == ["No", "Yes"]
    assert recommendations["risk_level"].tolist() == ["Low", "High"]


@pytest.mark.parametrize("test_size", [0, -0.1, 1, 1.1])
def test_invalid_test_size_is_rejected(test_size: float) -> None:
    """Invalid split sizes should fail before reading a dataset."""
    with pytest.raises(ValueError, match="test_size"):
        _validate_test_size(test_size)


def test_target_with_one_class_is_rejected() -> None:
    """Stratified model selection requires both retained and churned customers."""
    with pytest.raises(ValueError, match="exactly two classes"):
        _validate_class_balance(pd.Series(["No", "No"]))
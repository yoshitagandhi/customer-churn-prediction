"""Business cost estimation.

Turns a confusion matrix's outcome counts into estimated business
cost, savings, and avoided churn using a fully configurable cost
model — no monetary values are hardcoded anywhere in this module.
"""

from typing import Any

from configs.config import settings
from configs.logging_config import get_logger

logger = get_logger(__name__)


def calculate_business_cost(
    true_positives: int,
    false_positives: int,
    true_negatives: int,
    false_negatives: int,
    cost_false_positive: float = settings.cost_false_positive,
    cost_false_negative: float = settings.cost_false_negative,
    retention_campaign_cost: float = settings.retention_campaign_cost,
    customer_lifetime_value: float = settings.customer_lifetime_value,
    retention_success_rate: float = settings.retention_success_rate,
) -> dict[str, Any]:
    """Estimate the business cost/benefit of a set of confusion matrix outcomes.

    Cost model (every parameter is configurable via ``configs/config.py``):

    - True Negative: no cost — correctly left alone.
    - False Positive: ``cost_false_positive`` incurred — unnecessary
      outreach/offer to a customer who would not have churned.
    - False Negative: ``cost_false_negative`` incurred — revenue lost
      from an undetected churner who received no intervention.
    - True Positive: ``retention_campaign_cost`` incurred to run the
      intervention; ``customer_lifetime_value * retention_success_rate``
      expected to be saved, since not every contacted churner is
      assumed to be successfully retained (see ``retention_success_rate``).

    Args:
        true_positives: Count of correctly predicted churners.
        false_positives: Count of customers incorrectly flagged as churners.
        true_negatives: Count of correctly predicted non-churners.
        false_negatives: Count of undetected churners.
        cost_false_positive: Cost of an unnecessary retention contact.
            Defaults to ``settings.cost_false_positive``.
        cost_false_negative: Estimated cost of an undetected churner.
            Defaults to ``settings.cost_false_negative``.
        retention_campaign_cost: Cost of running a retention campaign
            on one flagged customer. Defaults to
            ``settings.retention_campaign_cost``.
        customer_lifetime_value: Expected revenue from retaining one
            customer. Defaults to ``settings.customer_lifetime_value``.
        retention_success_rate: Assumed probability that a retention
            campaign successfully retains a flagged churner. Defaults
            to ``settings.retention_success_rate``.

    Returns:
        A dictionary with the net business cost, estimated savings,
        expected avoided-churn count, and the cost breakdown by
        outcome type.
    """
    false_positive_cost = false_positives * cost_false_positive
    false_negative_cost = false_negatives * cost_false_negative
    campaign_cost = true_positives * retention_campaign_cost
    expected_avoided_churn = true_positives * retention_success_rate
    expected_retention_value = expected_avoided_churn * customer_lifetime_value
    estimated_savings = expected_retention_value - campaign_cost

    business_cost = (
        false_positive_cost + false_negative_cost + campaign_cost - expected_retention_value
    )

    return {
        "false_positive_cost": round(float(false_positive_cost), 2),
        "false_negative_cost": round(float(false_negative_cost), 2),
        "retention_campaign_cost_total": round(float(campaign_cost), 2),
        "expected_avoided_churn": round(float(expected_avoided_churn), 2),
        "estimated_savings": round(float(estimated_savings), 2),
        "business_cost": round(float(business_cost), 2),
    }

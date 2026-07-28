"""Explainability page.

Displays global SHAP explanations (previously generated figures — not
recomputed) and lets the user explain any individual customer's
prediction on demand, reusing Milestone 8's explanation logic.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from app.components.cards import render_business_insight_card
from app.components.charts import render_saved_figure
from app.components.forms import render_customer_form
from app.components.tables import render_shap_ranking_table
from app.services.explanation_service import explain_customer_prediction, generate_waterfall_figure
from app.utils.session import get_selected_customer
from app.utils.validators import validate_customer_input
from configs.config import settings
from configs.logging_config import get_logger

logger = get_logger(__name__)


def _render_global_section() -> None:
    """Render the global (dataset-wide) SHAP explanation section."""
    st.subheader("Global Feature Importance")
    figure_col1, figure_col2 = st.columns(2)
    with figure_col1:
        render_saved_figure("shap_bar.png", caption="Mean |SHAP| feature importance")
    with figure_col2:
        render_saved_figure("shap_beeswarm.png", caption="SHAP value distribution per feature")

    if settings.feature_importance_csv_path.exists():
        with st.expander("Feature Importance Table"):
            render_shap_ranking_table(pd.read_csv(settings.feature_importance_csv_path))


def _render_local_section() -> None:
    """Render the local (single-customer) SHAP explanation section."""
    st.subheader("Explain an Individual Prediction")

    selected_customer = get_selected_customer()
    if selected_customer:
        st.caption("Showing the customer most recently scored on the Prediction page.")
        use_selected = st.checkbox("Use most recently predicted customer", value=True)
    else:
        use_selected = False

    if use_selected and selected_customer:
        customer_data = selected_customer
    else:
        customer_data, submitted = render_customer_form()
        if not submitted:
            return
        errors = validate_customer_input(customer_data)
        if errors:
            for error in errors:
                st.error(error)
            return

    from app.utils.cache import get_cached_explainer, get_cached_model

    try:
        with st.spinner("Computing SHAP explanation..."):
            model = get_cached_model()
            explainer = get_cached_explainer(model, background_key="default")
            explanation = explain_customer_prediction(model, explainer, customer_data)
            waterfall_path: Path | None = generate_waterfall_figure(model, explainer, customer_data)
    except ImportError:
        st.warning(
            "SHAP is not installed in this environment. Install `shap` to enable this section."
        )
        return

    render_business_insight_card(explanation["business_insights"])
    if waterfall_path is not None:
        st.image(
            str(waterfall_path),
            caption="SHAP waterfall — this prediction",
            width="stretch",
        )

    with st.expander("Feature Contributions (Technical Detail)"):
        contributions = pd.DataFrame(
            explanation["prediction"]["top_positive_contributors"]
            + explanation["prediction"]["top_negative_contributors"]
        )
        if not contributions.empty:
            st.dataframe(contributions, width="stretch", hide_index=True)


def render_explainability() -> None:
    """Render the explainability page: global importance plus on-demand local explanations."""
    st.header("Explainability")
    st.write(
        "Understand which features drive the model's churn predictions overall, and why "
        "any individual customer was scored the way they were."
    )

    _render_global_section()
    st.divider()
    _render_local_section()

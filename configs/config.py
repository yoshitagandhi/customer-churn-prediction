"""Centralized project configuration.

This module defines a single, immutable configuration object that
future modules should import and use instead of hardcoding values
such as random seeds, dataset locations, or logging levels.
"""
from dataclasses import dataclass, field
from pathlib import Path

from configs.paths import MODELS_DIR, RAW_DATA_DIR, REPORTS_DIR
from src.utils.constants import TARGET_COLUMN


@dataclass(
    frozen=True,
    slots=True
    )
class ProjectConfig:
    """Immutable container for project-wide configuration values.

    Attributes:
        project_name: Human-readable name of the project.
        version: Current semantic version of the project.
        random_seed: Global random seed used for reproducibility across
            data splitting, model training, and any stochastic process.
        default_dataset_path: Default location of the raw dataset file.
        model_output_dir: Directory where trained model artifacts are
            saved.
        report_dir: Directory where generated reports, figures, and
            metrics are saved.
        logging_level: Default logging level for the application
            (e.g., "DEBUG", "INFO", "WARNING", "ERROR").
        target_column: Name of the churn target column expected in the
            dataset.
        supported_file_extensions: File extensions the data ingestion
            layer is allowed to load.
        preprocessor_path: Where the fitted preprocessing pipeline is
            serialized (Joblib).
        feature_metadata_path: Where feature metadata generated after
            preprocessing is saved (JSON).
        preprocessing_summary_path: Where the human-readable
            preprocessing summary is saved (Markdown).
        default_sampling_strategy: Which sampling strategy is used
            when none is explicitly requested.
        smote_k_neighbors: Number of nearest neighbors used by
            SMOTE-family samplers.
        sampling_report_path: Where the human-readable sampling
            experiment report is saved (Markdown).
        sampling_summary_path: Where the structured sampling
            experiment summary is saved (JSON).
        validation_size: Fraction of data held out for validation
            during model training.
        cv_folds: Number of cross-validation folds used during
            hyperparameter search.
        hyperparameter_search_iterations: Number of parameter
            combinations sampled by RandomizedSearchCV.
        scoring_metric: Primary metric optimized during
            hyperparameter search.
        default_model: Registered model name used when none is
            explicitly requested.
        best_model_path: Where the best trained model (full pipeline)
            is serialized (Joblib).
        training_metadata_path: Where training run metadata is saved
            (JSON).
        experiment_log_path: Where every experiment run is logged
            (JSON).
        model_comparison_path: Where the model comparison table is
            saved (CSV).
        experiments_dir: Directory where every trained model (not
            just the best) is serialized, for multi-model evaluation.
        positive_label: The target label representing churn, used
            wherever a binary positive class must be identified.
        evaluation_report_path: Where the human-readable evaluation
            report is saved (Markdown).
        evaluation_metrics_path: Where structured evaluation metrics
            are saved (JSON).
        model_ranking_path: Where the model ranking table is saved
            (CSV).
        shap_max_background_samples: Maximum number of background
            samples used to initialize the SHAP explainer.
        shap_top_n_features: Number of top contributing features to
            surface in feature rankings and prediction explanations.
        shap_example_customers_count: Number of example customer
            explanations included in the SHAP report.
        shap_report_path: Where the human-readable SHAP report is
            saved (Markdown).
        feature_importance_csv_path: Where the global feature
            importance ranking is saved (CSV).
        customer_explanations_path: Where example customer
            explanations are saved (JSON).
        threshold_range: (min, max) classification threshold range
            evaluated during threshold optimization.
        threshold_step: Step size between evaluated thresholds.
        optimization_objective: Which objective selects the optimal
            threshold ("min_cost", "max_f1", or
            "max_recall_min_precision").
        min_precision_constraint: Minimum precision required when
            ``optimization_objective`` is "max_recall_min_precision".
        cost_false_positive: Cost of unnecessarily contacting a
            customer who would not have churned.
        cost_false_negative: Estimated cost of an undetected churner.
        retention_campaign_cost: Cost of running a retention campaign
            on one flagged (predicted-positive) customer.
        customer_lifetime_value: Expected revenue from retaining one
            customer, used to estimate savings.
        retention_success_rate: Assumed probability that a retention
            campaign successfully retains a flagged churner.
        threshold_config_path: Where the selected threshold is saved
            for reuse during inference (JSON).
        threshold_report_path: Where the human-readable threshold
            optimization report is saved (Markdown).
        threshold_metrics_csv_path: Where the full threshold
            evaluation table is saved (CSV).
        business_decision_summary_path: Where the business decision
            summary is saved (JSON).
        app_icon: Emoji/icon shown in the Streamlit app's browser tab
            and header.
        max_batch_upload_rows: Maximum rows accepted in a single batch
            prediction CSV upload, to keep the UI responsive.
    """

    project_name: str = "Customer Churn Prediction"
    version: str = "0.1.0"
    random_seed: int = 42
    default_dataset_path: Path = field(
        default_factory=lambda: RAW_DATA_DIR / "telco_customer_churn.csv"
    )
    model_output_dir: Path = field(default_factory=lambda: MODELS_DIR)
    report_dir: Path = field(default_factory=lambda: REPORTS_DIR)
    logging_level: str = "INFO"
    target_column: str = TARGET_COLUMN
    supported_file_extensions: tuple[str, ...] = (".csv",)
    preprocessor_path: Path = field(default_factory=lambda: MODELS_DIR / "preprocessor.pkl")
    feature_metadata_path: Path = field(
        default_factory=lambda: REPORTS_DIR / "feature_metadata.json"
    )
    preprocessing_summary_path: Path = field(
        default_factory=lambda: REPORTS_DIR / "preprocessing_summary.md"
    )
    # Churn is the minority class. SMOTE improves the balance between precision
    # and recall used for retention decisions.
    default_sampling_strategy: str = "smote"
    smote_k_neighbors: int = 5
    sampling_report_path: Path = field(default_factory=lambda: REPORTS_DIR / "sampling_report.md")
    sampling_summary_path: Path = field(
        default_factory=lambda: REPORTS_DIR / "sampling_summary.json"
    )
    validation_size: float = 0.2
    cv_folds: int = 5
    hyperparameter_search_iterations: int = 20
    scoring_metric: str = "f1"
    default_model: str = "xgboost"
    best_model_path: Path = field(default_factory=lambda: MODELS_DIR / "best_model.pkl")
    training_metadata_path: Path = field(
        default_factory=lambda: REPORTS_DIR / "training_metadata.json"
    )
    experiment_log_path: Path = field(default_factory=lambda: REPORTS_DIR / "experiment_log.json")
    model_comparison_path: Path = field(
        default_factory=lambda: REPORTS_DIR / "model_comparison.csv"
    )
    experiments_dir: Path = field(default_factory=lambda: MODELS_DIR / "experiments")
    positive_label: str = "Yes"
    evaluation_report_path: Path = field(
        default_factory=lambda: REPORTS_DIR / "evaluation_report.md"
    )
    evaluation_metrics_path: Path = field(
        default_factory=lambda: REPORTS_DIR / "evaluation_metrics.json"
    )
    model_ranking_path: Path = field(default_factory=lambda: REPORTS_DIR / "model_ranking.csv")
    shap_max_background_samples: int = 100
    shap_top_n_features: int = 5
    shap_example_customers_count: int = 3
    shap_report_path: Path = field(default_factory=lambda: REPORTS_DIR / "shap_report.md")
    feature_importance_csv_path: Path = field(
        default_factory=lambda: REPORTS_DIR / "feature_importance.csv"
    )
    customer_explanations_path: Path = field(
        default_factory=lambda: REPORTS_DIR / "customer_explanations.json"
    )
    threshold_range: tuple[float, float] = (0.10, 0.90)
    threshold_step: float = 0.01
    optimization_objective: str = "min_cost"
    min_precision_constraint: float = 0.5
    cost_false_positive: float = 50.0
    cost_false_negative: float = 500.0
    retention_campaign_cost: float = 100.0
    customer_lifetime_value: float = 1000.0
    retention_success_rate: float = 0.3
    threshold_config_path: Path = field(
        default_factory=lambda: MODELS_DIR / "threshold_config.json"
    )
    threshold_report_path: Path = field(
        default_factory=lambda: REPORTS_DIR / "threshold_report.md"
    )
    threshold_metrics_csv_path: Path = field(
        default_factory=lambda: REPORTS_DIR / "threshold_metrics.csv"
    )
    business_decision_summary_path: Path = field(
        default_factory=lambda: REPORTS_DIR / "business_decision_summary.json"
    )
    app_icon: str = "CP"
    max_batch_upload_rows: int = 5000


# Singleton configuration instance to be imported across the project,
# e.g. `from configs.config import settings`.
settings = ProjectConfig()

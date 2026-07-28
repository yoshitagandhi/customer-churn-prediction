# Model Evaluation Report — Customer Churn Prediction

Generated at: 2026-07-21T22:05:33.745702+00:00

## Overall Summary

1 model(s) were evaluated on the held-out test set using ROC-AUC as the primary ranking metric (PR-AUC, F1, and Recall as tie-breakers). The best-performing model was **logistic_regression** (sampling strategy: smote).

## Metrics Table

| model_name | sampling_strategy | roc_auc | pr_auc | precision | recall | f1 | training_time_seconds |
|---|---|---|---|---|---|---|---|
| logistic_regression | smote | 0.8397 | 0.6415 | 0.5185 | 0.7554 | 0.6149 | 14.8708 |

## Best Model

- Model: **logistic_regression**
- Sampling strategy: smote
- ROC-AUC: 0.8397
- PR-AUC: 0.6415
- Precision: 0.5185
- Recall: 0.7554
- F1: 0.6149
- Brier score (calibration): 0.1632
- Selection reason: Ranked first by ROC-AUC (0.8397), with PR-AUC=0.6415, F1=0.6149, Recall=0.7554 used as tie-breakers.

## Strengths

- Strong discriminative ability (ROC-AUC=0.8397).
- High recall (0.7554): captures most actual churners.

## Weaknesses

- Predicted probabilities may be poorly calibrated (Brier score=0.1632).

## Recommendations

- Proceed to SHAP explainability (Milestone 8) to understand which features drive 'logistic_regression' predictions.
- Consider threshold optimization (Milestone 9) to adjust the precision/recall trade-off according to the business cost of false negatives vs. false positives.
- Consider probability calibration (e.g., Platt scaling or isotonic regression) before using raw predicted probabilities for business decisions.
- This model was selected automatically from evaluation metrics on the held-out test set; no winner was hardcoded.

## Figures Generated

- **roc_curve**: `C:\Users\Yoshita\Downloads\customer-churn-prediction\reports\figures\roc_curve.png`
- **precision_recall_curve**: `C:\Users\Yoshita\Downloads\customer-churn-prediction\reports\figures\precision_recall_curve.png`
- **metric_comparison**: `C:\Users\Yoshita\Downloads\customer-churn-prediction\reports\figures\metric_comparison.png`
- **classification_report_heatmap**: `C:\Users\Yoshita\Downloads\customer-churn-prediction\reports\figures\classification_report_heatmap.png`
- **confusion_matrix_logistic_regression**: `C:\Users\Yoshita\Downloads\customer-churn-prediction\reports\figures\confusion_matrix_logistic_regression.png`
- **calibration_curve**: `C:\Users\Yoshita\Downloads\customer-churn-prediction\reports\figures\calibration_curve.png`
- **learning_curve**: `C:\Users\Yoshita\Downloads\customer-churn-prediction\reports\figures\learning_curve.png`

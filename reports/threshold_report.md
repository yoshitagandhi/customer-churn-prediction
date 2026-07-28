# Threshold Optimization Report — Customer Churn Prediction

Generated at: 2026-07-21T22:05:34.738017+00:00

## Selected Threshold

- Optimization objective: **min_cost**
- Optimal threshold: **0.16**
- Precision: 0.3804
- Recall: 0.9704
- F1: 0.5466

## Comparison Highlights

- Lowest business cost: threshold=0.16 (cost=-37300.0)
- Highest F1: threshold=0.55 (F1=0.6231)
- Highest Recall: threshold=0.1 (Recall=0.9785)

## Business Cost Summary

- False positive cost: 29400.0
- False negative cost: 5500.0
- Retention campaign cost: 36100.0
- Expected avoided churn: 108.3 customer(s)
- Estimated savings: 72200.0
- Net business cost: -37300.0

## Expected Business Impact

- Total customers evaluated: 1405
- Risk level distribution:
  - Low: 497
  - Medium: 366
  - High: 343
  - Very High: 199

## Figures Generated

- **threshold_vs_precision**: `C:\Users\Yoshita\Downloads\customer-churn-prediction\reports\figures\threshold_vs_precision.png`
- **threshold_vs_recall**: `C:\Users\Yoshita\Downloads\customer-churn-prediction\reports\figures\threshold_vs_recall.png`
- **threshold_vs_f1**: `C:\Users\Yoshita\Downloads\customer-churn-prediction\reports\figures\threshold_vs_f1.png`
- **business_cost_curve**: `C:\Users\Yoshita\Downloads\customer-churn-prediction\reports\figures\threshold_vs_cost.png`
- **precision_recall_tradeoff**: `C:\Users\Yoshita\Downloads\customer-churn-prediction\reports\figures\precision_recall_tradeoff.png`
- **optimal_confusion_matrix**: `C:\Users\Yoshita\Downloads\customer-churn-prediction\reports\figures\optimal_threshold_confusion_matrix.png`

## Deployment Recommendation

Use threshold **0.16** (selected via the 'min_cost' objective) as the default classification threshold for inference and the Streamlit application, replacing the arbitrary 0.50 default. Revisit this threshold if business cost parameters change or the model is retrained.

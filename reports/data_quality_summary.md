# Data Quality Summary

Generated at: 2026-07-21T22:05:15.655986+00:00

## Dataset overview
- Rows: 7043
- Columns: 21
- Memory usage (MB): 1.864
- Target column: Churn

### Target distribution
- No: {'count': 5174, 'percentage': 73.46}
- Yes: {'count': 1869, 'percentage': 26.54}

## Validation summary
### Passed checks
- dataset_shape
- required_columns
- duplicate_column_names
- duplicate_rows
- duplicate_identifiers
- missing_values
- target_validation

### Warnings
- datatype_consistency: Columns expected to be numeric are stored as non-numeric: ['TotalCharges']

## Missing values
- No missing values detected.

## Duplicate summary
- Duplicate rows: 0
- Duplicate customer IDs: 0

## Recommendations
- Convert 'TotalCharges' to a numeric data type (currently 'str').
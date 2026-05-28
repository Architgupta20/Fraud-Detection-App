# Phase 2: Label Leakage Controls

This project creates labels with rules in `Scripts/fraud_risk_scoring.py`.  
To avoid circular learning, model training excludes all rule-input columns.

## Rule-to-feature map

| Rule input column | Used in labeling rules | Allowed as model feature |
|---|---|---|
| `payment_to_drug_cost_ratio` | Yes | No |
| `opioid_claims` | Yes | No |
| `high_payment_flag` | Yes | No |
| `high_opioid_flag` | Yes | No |
| `peer_deviation_score` | Yes | No |
| `elderly_focus_flag` | Yes | No |

## Current non-leaky feature set

All model scripts now use:

- `total_claims`
- `total_drug_cost`
- `opioid_cost`
- `antibiotic_claims`
- `avg_risk_score`
- `payment_variability`
- `adjusted_risk_payment`

## Holdout and reporting

- Holdout split is by `prescriber_id` hash (Spark scripts, and sklearn script).
- Metrics reported include:
  - per-class precision/recall/F1
  - macro-F1
  - weighted F1 (Spark evaluator)

## Run

```bash
python Models/train_sklearn.py --sample-frac 0.2
python Models/rf_removing_leakage.py
python Models/gbt_removing_leakage.py
```

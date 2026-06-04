# Risk rules (product v2.1 — binary)

Rules define **investigation priority**, not confirmed fraud.  
Implementation: `risk_rules.py` (used by `run_pipeline.py score`).

**Current version:** `2.1.0` (column `rules_version` on scored output)

## Scoring model

- Each rule adds **points** when true (multiple rules can fire).
- `rules_fired` = pipe-separated rule IDs, e.g. `payment_ratio_high|opioid_volume_high`
- `risk_points` = sum of points (alias: `fraud_risk_score`)

### Binary categories (option B)

| Points | Category | Meaning |
|--------|----------|---------|
| **0–1** | **Low** | Not priority for review queue |
| **≥ 2** | **High** | Needs review (includes former Medium 2–3 and High 4+) |

v2.0 used three tiers (High ≥4, Medium 2–3, Low 0–1). v2.1 collapses to **Low / High** for clearer triage and ML.

## Rules

| Rule ID | Points | Condition (summary) |
|---------|--------|---------------------|
| `payment_ratio_high` | 2 | `payment_to_drug_cost_ratio` > 1 |
| `opioid_volume_high` | 2 | `opioid_claims` > 100 **or** ≥ 95th percentile within `provider_type` |
| `high_payment_flag` | 1 | Average payment > $1,000 |
| `high_opioid_flag` | 1 | Opioid claims > 50% of all claims |
| `peer_outlier` | 1 | `peer_deviation_score` > 5 **or** ≥ 95th pct within `provider_type` |
| `payment_spiky` | 1 | `payment_variability` > 3 **or** ≥ 95th pct within `provider_type` |
| `antibiotic_heavy` | 1 | `antibiotic_claim_ratio` > 0.25 and `antibiotic_claims` > 50 |
| `elderly_focus` | 1 | `elderly_focus_flag` = 1 (avg patient age > 70) |
| `total_payments_high` | 1 | `total_payment_amount` > $50,000 **or** ≥ 95th pct within `provider_type` |

## ML training

All trainers import **`ML_FEATURE_COLS`** and **`LABEL_MAP`** (`Low`=0, `High`=1) from `risk_rules.py` via `Models/ml_common.py`.

After changing rules:

```bash
export BASE_DIR="$(pwd)"
python run_pipeline.py score
python Models/train_xgb.py --sample-frac 1.0 --strict-rules-version
python Models/train_sklearn.py --sample-frac 1.0 --strict-rules-version
```

## Change log

| Version | Notes |
|---------|--------|
| 1.x | First-match `when` chain; score 0/1/2 only |
| 2.0.0 | Additive points, `rules_fired`, peer percentiles |
| 2.1.0 | **Binary Low/High** — `risk_points >= 2` → High |

# Data Acquisition Guide

This repository is code-first. Raw CMS datasets are downloaded locally and kept out of git.

## Required datasets

1. **Medicare Part D Prescriber Public Use File**
   - Local filename: `Data/part_d_prescribers.csv`
   - Used by: ETL + feature engineering
2. **CMS Open Payments (General Payments)**
   - Local filename: `Data/open_payments.csv`
   - Used by: payment aggregation + joins

## Expected local structure

```text
Data/
  part_d_prescribers.csv
  open_payments.csv
  clean_prescribers.csv
  clean_payments.csv
  prescriber_level_dataset.csv
  prescriber_level_enriched.csv
  fraud_risk_scored_prescribers.csv
  Model_Data/
```

## Approximate sizes

- Raw + processed local `Data/` footprint: ~18 GB
- Model output artifacts under `Data/Model_Data/`: tens of MB

## Download sources

- CMS Open Payments data explorer/download:  
  <https://openpaymentsdata.cms.gov/>
- CMS Medicare Part D Prescriber datasets:  
  <https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers>

## Git policy

- `Data/*` stays ignored by default.
- `Data/Model_Data/` can be optionally committed for demo purposes.
- Do not commit PHI/PII-heavy raw files to public repositories.

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

## Do not open huge CSVs in Excel / Cursor preview

Files like `open_payments.csv` (~8 GB) or `merged_payment_level_dataset.csv` (~5 GB) can **freeze or shut down** a laptop if the editor or Excel tries to load them entirely into RAM.

**Safe ways to inspect:**

```bash
export BASE_DIR="$(pwd)"
# First 5 rows only
python Scripts/inspect_csv.py scored --rows 5
python Scripts/inspect_csv.py raw-payments --rows 3

# Terminal peek (no pandas)
head -n 3 Data/fraud_risk_scored_prescribers.csv
```

**For daily work**, you usually only need `fraud_risk_scored_prescribers.csv` (~400 MB). You can delete raw/intermediate files locally after a successful pipeline run (see sizes above) and re-download from CMS when you need a full refresh.

**Never double-click** `open_payments.csv` or `merged_payment_level_dataset.csv` on macOS if the default app is Excel or Numbers.

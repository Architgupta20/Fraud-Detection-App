# Data Acquisition Guide

This repository is code-first. All datasets live under `Data/` in **stage folders** (see [`Data/README.md`](../Data/README.md)).

## Folder layout

```text
Data/
  Original_Datasets/          ← you download CMS files here
    part_d_prescribers.csv
    open_payments.csv
  Cleaned_Datasets/           ← run_pipeline.py clean
    clean_prescribers.csv
    clean_payments.csv
  Aggregated_Datasets/        ← run_pipeline.py aggregate
    prescriber_level_dataset.csv
  Enriched_Datasets/          ← run_pipeline.py features
    prescriber_level_enriched.csv
  Scored_Datasets/            ← run_pipeline.py score (train ML on this)
    fraud_risk_scored_prescribers.csv
  Model_Data/                 ← training outputs (predictions, plots)
  _Spark_Temp/                ← automatic; safe to delete
```

## Required downloads (Original_Datasets)

1. **Medicare Part D Prescriber Public Use File**  
   → `Data/Original_Datasets/part_d_prescribers.csv`

2. **CMS Open Payments (General Payments)**  
   → `Data/Original_Datasets/open_payments.csv`

**Sources:**

- Open Payments: <https://openpaymentsdata.cms.gov/>
- Part D prescribers: <https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers>

## Approximate sizes

| Folder | Rough size |
|--------|------------|
| Original_Datasets | ~8 GB |
| Full local `Data/` after pipeline | ~18 GB |
| Scored_Datasets | ~400 MB |
| Model_Data | tens of MB |

## Git policy

- Everything under `Data/` is gitignored except `Model_Data/` (demo artifacts).
- Do not commit raw CMS files.

## Do not open huge CSVs in Excel / Cursor preview

Use:

```bash
export BASE_DIR="$(pwd)"
python Scripts/inspect_csv.py raw-payments --rows 3
python Scripts/inspect_csv.py scored --rows 5
```

## Data layout

Clone uses stage folders under `Data/` (see [Data/README.md](../Data/README.md)). If you have legacy flat `Data/*.csv` files, move them into the matching stage folder manually.

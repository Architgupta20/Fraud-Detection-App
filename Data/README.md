# Data folder guide

Each pipeline stage has its **own folder**. Files move forward left → right.

```
Original_Datasets     →  CMS downloads (you add these)
Cleaned_Datasets      →  after run_pipeline.py clean
Aggregated_Datasets   →  after aggregate (merged prescriber + payments)
Enriched_Datasets     →  after features
Scored_Datasets       →  after score (use this for training + app)
Model_Data            →  model predictions & charts
_Spark_Temp           →  auto-deleted temp files (ignore)
```

## What goes where

| Folder | Files | How created |
|--------|-------|-------------|
| **Original_Datasets** | `part_d_prescribers.csv`, `open_payments.csv` | You download from CMS ([docs/DATA.md](../docs/DATA.md)) |
| **Cleaned_Datasets** | `clean_prescribers.csv`, `clean_payments.csv` | `python run_pipeline.py clean` |
| **Aggregated_Datasets** | `prescriber_level_dataset.csv` (prescriber + summed payments per NPI) | `python run_pipeline.py aggregate` |
| **Enriched_Datasets** | `prescriber_level_enriched.csv` | `python run_pipeline.py features` |
| **Scored_Datasets** | `fraud_risk_scored_prescribers.csv` (Low/High only, `rules_version` 2.1.0) | `python run_pipeline.py score` |
| **Model_Data** | `fraud_detection_gbt_sklearn_predictions.csv` (in Git for deploy); `npi_risk_lookup.csv` (local, `Scripts/build_npi_lookup_index.py`) | `Models/train_*.py` / build script |

If you still have old **Low/Medium/High** scored or prediction files, delete them and re-run score + train:

```bash
bash Scripts/remove_legacy_3class_artifacts.sh
```

## Quick preview (do not open huge CSVs in Excel)

```bash
python Scripts/inspect_csv.py scored --rows 5
python Scripts/inspect_csv.py raw-payments --rows 3
```

#!/usr/bin/env bash
# Move legacy flat Data/*.csv into stage folders (safe to re-run).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
D="$ROOT/Data"

mkdir -p "$D/Original_Datasets" "$D/Cleaned_Datasets" "$D/Aggregated_Datasets" \
  "$D/Enriched_Datasets" "$D/Scored_Datasets" "$D/Model_Data" "$D/_Spark_Temp"

move_if_exists() {
  local src="$1" dest="$2"
  if [[ -f "$src" && "$src" != "$dest" ]]; then
    echo "Moving $(basename "$src") -> $(dirname "$dest")/"
    mv "$src" "$dest"
  fi
}

move_if_exists "$D/part_d_prescribers.csv" "$D/Original_Datasets/part_d_prescribers.csv"
move_if_exists "$D/open_payments.csv" "$D/Original_Datasets/open_payments.csv"
move_if_exists "$D/clean_prescribers.csv" "$D/Cleaned_Datasets/clean_prescribers.csv"
move_if_exists "$D/clean_payments.csv" "$D/Cleaned_Datasets/clean_payments.csv"
move_if_exists "$D/prescriber_level_dataset.csv" "$D/Aggregated_Datasets/prescriber_level_dataset.csv"
# merged_payment_level_dataset.csv removed from pipeline (~5 GB); delete if still present:
rm -f "$D/merged_payment_level_dataset.csv" "$D/Aggregated_Datasets/merged_payment_level_dataset.csv"
move_if_exists "$D/prescriber_level_enriched.csv" "$D/Enriched_Datasets/prescriber_level_enriched.csv"
move_if_exists "$D/fraud_risk_scored_prescribers.csv" "$D/Scored_Datasets/fraud_risk_scored_prescribers.csv"

echo "Done. See Data/README.md for the layout."

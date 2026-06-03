"""Inspect the cleaned prescribers CSV produced by run_pipeline.py clean."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace

from config import CLEAN_PRESCRIBERS_CSV

spark = SparkSession.builder.appName("CleanPrescribersInspect").getOrCreate()

file_path = str(CLEAN_PRESCRIBERS_CSV)
df = spark.read.csv(file_path, header=True, inferSchema=True)

num_rows = df.count()
num_cols = len(df.columns)
print("\n===== CLEAN PRESCRIBERS DATASET SHAPE =====")
print(f"Rows: {num_rows}")
print(f"Columns: {num_cols}")

print("\n===== CLEAN PRESCRIBERS DATASET COLUMNS =====")
for col_name in df.columns:
    print(col_name)

print("\n===== FIRST 5 ROWS (VERTICAL VIEW) =====")
rows = df.limit(5).collect()
for i, row in enumerate(rows, start=1):
    print(f"\n--- Row {i} ---")
    for col_name, val in row.asDict().items():
        print(f"{col_name}: {val}")

numeric_cols = [
    "total_claims",
    "total_drug_cost",
    "total_beneficiaries",
    "opioid_claims",
    "opioid_cost",
    "opioid_beneficiaries",
    "opioid_rate",
    "antibiotic_claims",
    "antibiotic_cost",
    "avg_patient_age",
    "female_patients",
    "male_patients",
    "avg_risk_score",
]

print("\n===== NON-NUMERIC VALUES IN NUMERIC COLUMNS =====")
for col_name in numeric_cols:
    non_numeric_count = df.filter(
        regexp_replace(col(col_name), "[0-9.+-Ee]", "") != ""
    ).count()
    print(f"Non-numeric rows in {col_name}: {non_numeric_count}")

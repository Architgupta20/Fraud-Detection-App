import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
import os
import shutil

from config import FRAUD_RISK_SCORED_CSV, PRESCRIBER_LEVEL_ENRICHED_CSV, data_path

# -------------------------------
# Step 1: Initialize Spark
# -------------------------------
spark = SparkSession.builder.appName("FraudRiskScoring").getOrCreate()

# -------------------------------
# Step 2: Load enriched dataset
# -------------------------------
file_path = str(PRESCRIBER_LEVEL_ENRICHED_CSV)
df = spark.read.csv(file_path, header=True, inferSchema=True)

print("\n===== Loaded dataset =====")
print(f"Rows: {df.count()}, Columns: {len(df.columns)}")

# -------------------------------
# Step 3: Fraud Risk Scoring Rules
# -------------------------------
df = df.withColumn(
    "fraud_risk_score",
    (
        when(col("payment_to_drug_cost_ratio") > 1, 2)  # unusually high payments vs drug cost
        .when(col("opioid_claims") > 100, 2)           # heavy opioid prescribing
        .when(col("high_payment_flag") == 1, 2)        # already flagged as high payment
        .when(col("high_opioid_flag") == 1, 2)         # already flagged as high opioid
        .when(col("peer_deviation_score") > 5, 2)      # outlier vs peers
        .when(col("elderly_focus_flag") == 1, 1)       # elderly patient focus
        .otherwise(0)                                  # normal
    )
)

# -------------------------------
# Step 4: Categorize into Low / Medium / High
# -------------------------------
df = df.withColumn(
    "fraud_risk_category",
    when(col("fraud_risk_score") >= 2, "High")
    .when(col("fraud_risk_score") == 1, "Medium")
    .otherwise("Low")
)

# -------------------------------
# Step 5: Sort dataset by prescriber_id (ascending)
# -------------------------------
df = df.orderBy(col("prescriber_id").asc())

# -------------------------------
# Step 6: Summary of Risk Categories
# -------------------------------
print("\n===== FRAUD RISK CATEGORY DISTRIBUTION =====")
df.groupBy("fraud_risk_category").count().show()

# -------------------------------
# Step 7: Save results as a single ordered CSV file
# -------------------------------
output_dir = str(data_path("fraud_risk_scored_prescribers_temp"))
final_output_path = str(FRAUD_RISK_SCORED_CSV)

# Coalesce to single file
df.coalesce(1).write.mode("overwrite").option("header", True).csv(output_dir)

# Rename the part file automatically
part_file = [f for f in os.listdir(output_dir) if f.startswith("part-") and f.endswith(".csv")][0]
shutil.move(os.path.join(output_dir, part_file), final_output_path)
shutil.rmtree(output_dir)

print("\nFraud risk scoring completed! Results saved as ordered CSV file:")
print(final_output_path)

# -------------------------------
# Step 8: Show first 10 rows (VERTICAL FORMAT)
# -------------------------------
print("\n===== FIRST 10 ROWS (ORDERED BY PRESCRIBER_ID ASCENDING) =====")
sample_rows = df.select(
    "prescriber_id",
    "provider_type",
    "state",
    "total_claims",
    "total_payment_amount",
    "payment_to_drug_cost_ratio",
    "opioid_claims",
    "peer_deviation_score",
    "fraud_risk_score",
    "fraud_risk_category"
).limit(10).collect()

for i, row in enumerate(sample_rows, start=1):
    print(f"\n--- Row {i} ---")
    for col_name, value in row.asDict().items():
        print(f"{col_name}: {value}")

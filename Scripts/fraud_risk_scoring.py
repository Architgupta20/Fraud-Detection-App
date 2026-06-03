"""Standalone Spark job: apply product risk rules to enriched prescriber data."""

import os
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from config import FRAUD_RISK_SCORED_CSV, PRESCRIBER_LEVEL_ENRICHED_CSV, data_path
from risk_rules import RULES_VERSION, apply_risk_scoring_spark

spark = SparkSession.builder.appName("FraudRiskScoring").getOrCreate()

file_path = str(PRESCRIBER_LEVEL_ENRICHED_CSV)
df = spark.read.csv(file_path, header=True, inferSchema=True)

print("\n===== Loaded dataset =====")
print(f"Rows: {df.count()}, Columns: {len(df.columns)}")
print(f"Rules version: {RULES_VERSION}")

df = apply_risk_scoring_spark(df)

print("\n===== RISK CATEGORY DISTRIBUTION =====")
df.groupBy("fraud_risk_category").count().orderBy("fraud_risk_category").show(truncate=False)

df = df.orderBy(col("prescriber_id").asc())

output_dir = str(data_path("fraud_risk_scored_prescribers_temp"))
final_output_path = str(FRAUD_RISK_SCORED_CSV)

df.coalesce(1).write.mode("overwrite").option("header", True).csv(output_dir)

part_file = [f for f in os.listdir(output_dir) if f.startswith("part-") and f.endswith(".csv")][0]
shutil.move(os.path.join(output_dir, part_file), final_output_path)
shutil.rmtree(output_dir)

print("\nRisk scoring completed:")
print(final_output_path)

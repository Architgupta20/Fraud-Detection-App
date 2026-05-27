# # feature_engineering_prescribers.py

# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, when, lit, avg

# # -------------------------------
# # Step 1: Initialize Spark
# # -------------------------------
# spark = SparkSession.builder.appName("FeatureEngineeringPrescribers").getOrCreate()

# # -------------------------------
# # Step 2: Load prescriber-level dataset
# # -------------------------------
# file_path = "/Users/architgupta280/Desktop/LTI/Project/Data/prescriber_level_dataset.csv"
# df = spark.read.csv(file_path, header=True, inferSchema=True)

# print("===== Original Dataset Shape =====")
# print(f"Rows: {df.count()}, Columns: {len(df.columns)}")

# # -------------------------------
# # Step 3: Feature Engineering
# # -------------------------------

# # Avoid divide-by-zero issues using when/otherwise
# df = df.withColumn("payment_per_claim",
#                    when(col("total_claims") > 0, col("total_payment_amount") / col("total_claims"))
#                    .otherwise(0.0))

# df = df.withColumn("payment_to_drug_cost_ratio",
#                    when(col("total_drug_cost") > 0, col("total_payment_amount") / col("total_drug_cost"))
#                    .otherwise(0.0))

# df = df.withColumn("opioid_claim_ratio",
#                    when(col("total_claims") > 0, col("opioid_claims") / col("total_claims"))
#                    .otherwise(0.0))

# df = df.withColumn("antibiotic_claim_ratio",
#                    when(col("total_claims") > 0, col("antibiotic_claims") / col("total_claims"))
#                    .otherwise(0.0))

# # Flags
# df = df.withColumn("high_opioid_flag", when(col("opioid_claim_ratio") > 0.5, lit(1)).otherwise(lit(0)))
# df = df.withColumn("payment_variability",
#                    when(col("avg_payment_amount") > 0, col("max_payment_amount") / col("avg_payment_amount"))
#                    .otherwise(0.0))

# df = df.withColumn("high_payment_flag",
#                    when(col("avg_payment_amount") > 1000, lit(1)).otherwise(lit(0)))  # threshold = 1000, tune later

# df = df.withColumn("female_to_male_ratio",
#                    when(col("male_patients") > 0, col("female_patients") / col("male_patients"))
#                    .otherwise(0.0))

# df = df.withColumn("elderly_focus_flag", when(col("avg_patient_age") > 70, lit(1)).otherwise(lit(0)))

# df = df.withColumn("adjusted_risk_payment", col("total_payment_amount") * col("avg_risk_score"))

# # Peer Deviation Score → Compare each provider’s avg payment to avg of same provider_type
# provider_avg = df.groupBy("provider_type").agg(avg("avg_payment_amount").alias("peer_avg_payment"))
# df = df.join(provider_avg, on="provider_type", how="left")
# df = df.withColumn("peer_deviation_score",
#                    when(col("peer_avg_payment") > 0, col("avg_payment_amount") / col("peer_avg_payment"))
#                    .otherwise(0.0))

# # -------------------------------
# # Step 4: Save enriched dataset
# # -------------------------------
# output_path = "/Users/architgupta280/Desktop/LTI/Project/Data/prescriber_level_enriched.csv"

# # Save as single CSV
# import os, glob, shutil
# temp_folder = output_path.replace(".csv", "_temp")

# df.coalesce(1).write.csv(temp_folder, header=True, mode="overwrite")

# # Move and rename part file
# part_file = glob.glob(os.path.join(temp_folder, "part-*.csv"))[0]
# shutil.move(part_file, output_path)
# shutil.rmtree(temp_folder)

# print("===== Feature Engineering Completed =====")
# print(f"Rows: {df.count()}, Columns: {len(df.columns)}")
# print("Enriched prescriber dataset saved successfully!")

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pyspark.sql import SparkSession

from config import PRESCRIBER_LEVEL_ENRICHED_CSV

# -------------------------------
# Step 1: Initialize Spark
# -------------------------------
spark = SparkSession.builder.appName("ViewEnrichedPrescribers").getOrCreate()

# -------------------------------
# Step 2: Load enriched dataset
# -------------------------------

file_path = str(PRESCRIBER_LEVEL_ENRICHED_CSV)
df = spark.read.csv(file_path, header=True, inferSchema=True)

print(f"\n===== Dataset Loaded =====")
print(f"Rows: {df.count()}, Columns: {len(df.columns)}")
print(df.columns)


# -------------------------------
# Step 3: Show first 20 rows vertically
# -------------------------------
print("\n===== First 20 Rows (Vertical Format) =====")
df.show(20, truncate=False, vertical=True)

# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, when, sum as spark_sum, avg as spark_avg, max as spark_max, count as spark_count
# import os
# import glob
# import shutil

# # -------------------------------
# # Step 1: Initialize Spark
# # -------------------------------
# spark = SparkSession.builder.appName("PrescriberLevelAggregation").getOrCreate()

# # -------------------------------
# # Step 2: File paths
# # -------------------------------
# prescribers_file = "/Users/architgupta280/Desktop/LTI/Project/Data/clean_prescribers.csv"
# payments_file = "/Users/architgupta280/Desktop/LTI/Project/Data/clean_payments.csv"
# output_temp_folder = "/Users/architgupta280/Desktop/LTI/Project/Data/temp_prescriber_level"
# output_final_csv = "/Users/architgupta280/Desktop/LTI/Project/Data/prescriber_level_dataset.csv"

# # -------------------------------
# # Step 3: Load datasets
# # -------------------------------
# prescribers_df = spark.read.csv(prescribers_file, header=True, inferSchema=False)
# payments_df = spark.read.csv(payments_file, header=True, inferSchema=False)

# # -------------------------------
# # Step 4: Clean numeric columns in prescribers
# # Use when().otherwise(0) to avoid casting errors
# # -------------------------------
# numeric_cols = [
#     "total_claims", "total_drug_cost", "total_beneficiaries", "opioid_claims",
#     "opioid_cost", "opioid_beneficiaries", "avg_patient_age", "female_patients",
#     "male_patients", "avg_risk_score", "antibiotic_claims", "antibiotic_cost",
#     "opioid_rate"
# ]

# for col_name in numeric_cols:
#     if col_name in prescribers_df.columns:
#         prescribers_df = prescribers_df.withColumn(
#             col_name,
#             when((col(col_name).isNull()) | (col(col_name) == "None"), 0)
#             .otherwise(when(col(col_name).rlike("^[0-9.]+$"), col(col_name).cast("double")).otherwise(0))
#         )

# # -------------------------------
# # Step 5: Fill string columns in prescribers
# # -------------------------------
# string_cols = [c for c in prescribers_df.columns if c not in numeric_cols]
# prescribers_df = prescribers_df.fillna("NA", subset=string_cols)

# # -------------------------------
# # Step 6: Clean numeric column in payments
# # -------------------------------
# payments_df = payments_df.withColumn(
#     "Payment_Amount",
#     when((col("Payment_Amount").isNull()) | (col("Payment_Amount") == "None"), 0)
#     .otherwise(when(col("Payment_Amount").rlike("^[0-9.]+$"), col("Payment_Amount").cast("double")).otherwise(0))
# )

# # -------------------------------
# # Step 7: Aggregate payments per prescriber
# # -------------------------------
# aggregated_payments = payments_df.groupBy("NPI").agg(
#     spark_sum("Payment_Amount").alias("total_payment_amount"),
#     spark_count("Payment_Amount").alias("num_payments"),
#     spark_avg("Payment_Amount").alias("avg_payment_amount"),
#     spark_max("Payment_Amount").alias("max_payment_amount")
# )

# # -------------------------------
# # Step 8: Merge aggregated payments with prescribers
# # -------------------------------
# merged_df = prescribers_df.join(
#     aggregated_payments,
#     prescribers_df["prescriber_id"] == aggregated_payments["NPI"],
#     "left"
# ).drop("NPI")  # Drop NPI to avoid duplication

# # -------------------------------
# # Step 9: Fill nulls in aggregated columns
# # -------------------------------
# merged_df = merged_df.fillna({"total_payment_amount": 0, "num_payments": 0, "avg_payment_amount": 0, "max_payment_amount": 0})

# -------------------------------
# Step 10: Show first 5 rows vertically
# -------------------------------
# print("First 5 rows of aggregated prescriber dataset (vertical format):")
# for i, row in enumerate(merged_df.take(5), 1):
#     print(f"--- Row {i} ---")
#     for col_name in merged_df.columns:
#         print(f"{col_name}: {row[col_name]}")
#     print()

# # -------------------------------
# # Step 11: Save as single CSV using temp folder
# # -------------------------------
# merged_df.coalesce(1).write.csv(output_temp_folder, header=True, mode="overwrite")
# part_file = glob.glob(os.path.join(output_temp_folder, "part-*.csv"))[0]
# shutil.move(part_file, output_final_csv)
# shutil.rmtree(output_temp_folder)

# print(f"Aggregated prescriber-level dataset saved successfully: {output_final_csv}")

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pyspark.sql import SparkSession

from config import PRESCRIBER_LEVEL_CSV

# -------------------------------
# Step 1: Initialize Spark
# -------------------------------
spark = SparkSession.builder.appName("CheckShape").getOrCreate()

# -------------------------------
# Step 2: Load dataset
# -------------------------------

file_path = str(PRESCRIBER_LEVEL_CSV)
df = spark.read.csv(file_path, header=True, inferSchema=True)

# -------------------------------
# Step 3: Print rows & columns
# -------------------------------
num_rows = df.count()
num_cols = len(df.columns)

print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_cols}")

# -------------------------------
# Step 4: Print first 5 rows vertically
# -------------------------------
print("First 5 rows of aggregated prescriber dataset (vertical format):")
for i, row in enumerate(df.take(20), 1):
    print(f"--- Row {i} ---")
    for col_name in df.columns:
        print(f"{col_name}: {row[col_name]}")
    print()

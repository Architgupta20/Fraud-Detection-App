# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, expr
# import os
# import shutil
# import glob

# # -------------------------------
# # Step 1: Initialize Spark
# # -------------------------------
# spark = SparkSession.builder.appName("SafeMergeAndSave").getOrCreate()

# # -------------------------------
# # Step 2: File paths
# # -------------------------------
# prescribers_file = "/Users/architgupta280/Desktop/LTI/Project/Data/clean_prescribers.csv"
# payments_file = "/Users/architgupta280/Desktop/LTI/Project/Data/clean_payments.csv"
# output_file = "/Users/architgupta280/Desktop/LTI/Project/Data/merged_payment_level_dataset.csv"
# temp_folder = "/Users/architgupta280/Desktop/LTI/Project/Data/temp_merged_dataset"

# # -------------------------------
# # Step 3: Load datasets
# # -------------------------------
# prescribers_df = spark.read.csv(prescribers_file, header=True, inferSchema=True)
# payments_df = spark.read.csv(payments_file, header=True, inferSchema=True)

# # -------------------------------
# # Step 4: Detect non-numeric values in NPI / prescriber_id
# # -------------------------------
# print("Rows with non-numeric prescriber_id in prescribers_df:")
# prescribers_df.filter(~col("prescriber_id").cast("bigint").isNotNull()).show(10)

# print("Rows with non-numeric NPI in payments_df:")
# payments_df.filter(~col("NPI").cast("bigint").isNotNull()).show(10)

# # -------------------------------
# # Step 5: Safe casting for join
# # -------------------------------
# prescribers_df = prescribers_df.withColumn("prescriber_id_num", expr("try_cast(prescriber_id as bigint)"))
# payments_df = payments_df.withColumn("NPI_num", expr("try_cast(NPI as bigint)"))

# # Optional: remove rows where numeric IDs could not be parsed
# prescribers_df = prescribers_df.filter(col("prescriber_id_num").isNotNull())
# payments_df = payments_df.filter(col("NPI_num").isNotNull())

# # -------------------------------
# # Step 6: Drop duplicate columns & Merge datasets
# # -------------------------------
# # Drop duplicates from prescribers to avoid column clashes
# prescribers_df = prescribers_df.drop("first_name", "last_name", "city", "state", "zip")

# # Merge
# merged_df = payments_df.join(
#     prescribers_df,
#     payments_df["NPI_num"] == prescribers_df["prescriber_id_num"],
#     "left"
# )

# # Drop temporary numeric columns
# merged_df = merged_df.drop("NPI_num", "prescriber_id_num")

# # -------------------------------
# # Step 7: Show first 5 rows vertically
# # -------------------------------
# print("First 5 rows of merged dataset (vertical format):")
# for i, row in enumerate(merged_df.take(5), 1):
#     print(f"\n--- Row {i} ---")
#     for col_name in merged_df.columns:
#         print(f"{col_name}: {row[col_name]}")

# # -------------------------------
# # Step 8: Save merged dataset as single CSV (temp folder method)
# # -------------------------------
# merged_df.coalesce(1).write.csv(temp_folder, header=True, mode="overwrite")

# # Move the single CSV file to the final output location
# part_file = glob.glob(os.path.join(temp_folder, "part-*.csv"))[0]
# shutil.move(part_file, output_file)
# shutil.rmtree(temp_folder)

# print(f"Merged dataset saved as single CSV file: {output_file}")




import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pyspark.sql import SparkSession

from config import MERGED_PAYMENT_LEVEL_CSV

# -------------------------------
# Initialize Spark
# -------------------------------
spark = SparkSession.builder.appName("DatasetShape").getOrCreate()

# -------------------------------
# File paths
# -------------------------------

merged_file = str(MERGED_PAYMENT_LEVEL_CSV)

# -------------------------------
# Load merged dataset
# -------------------------------
merged_df = spark.read.csv(merged_file, header=True, inferSchema=True)

# -------------------------------
# Get number of rows and columns
# -------------------------------
num_rows = merged_df.count()
num_cols = len(merged_df.columns)

# -------------------------------
# Print dataset shape with headings
# -------------------------------
print("\n===== DATASET SHAPE =====")
print(f"Rows: {num_rows}")
print(f"Columns: {num_cols}")

# -------------------------------
# Print first 5 rows (vertical)
# -------------------------------
print("\n===== FIRST 5 ROWS =====")
for i, row in enumerate(merged_df.take(20), 1):
    print(f"\n--- Row {i} ---")
    for col_name in merged_df.columns:
        print(f"{col_name}: {row[col_name]}")










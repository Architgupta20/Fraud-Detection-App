# from pyspark.sql import SparkSession
# import os
# import shutil
# import glob

# # -------------------------------
# # Step 1: Initialize Spark
# # -------------------------------
# spark = SparkSession.builder.appName("PaymentsEDA").getOrCreate()

# # -------------------------------
# # Step 2: Load Payments dataset
# # -------------------------------
# payments_path = "/Users/architgupta280/Desktop/LTI/Project/Data/open_payments.csv"
# payments = spark.read.csv(payments_path, header=True, inferSchema=True)

# # -------------------------------
# # Step 3: Select important columns
# # -------------------------------
# important_cols = [
#     "Covered_Recipient_NPI",
#     "Covered_Recipient_First_Name",
#     "Covered_Recipient_Last_Name",
#     "Covered_Recipient_Specialty_1",
#     "Recipient_City",
#     "Recipient_State",
#     "Recipient_Zip_Code",
#     "Total_Amount_of_Payment_USDollars",
#     "Date_of_Payment",
#     "Form_of_Payment_or_Transfer_of_Value",
#     "Nature_of_Payment_or_Transfer_of_Value",
#     "Submitting_Applicable_Manufacturer_or_Applicable_GPO_Name",
#     "Name_of_Drug_or_Biological_or_Device_or_Medical_Supply_1",
#     "Product_Category_or_Therapeutic_Area_1",
#     "Physician_Ownership_Indicator",
#     "Third_Party_Payment_Recipient_Indicator",
#     "Charity_Indicator",
#     "Dispute_Status_for_Publication"
# ]

# payments_selected = payments.select(important_cols)

# # -------------------------------
# # Step 4: Rename columns
# # -------------------------------
# renamed_cols = {
#     "Covered_Recipient_NPI": "NPI",
#     "Covered_Recipient_First_Name": "First_Name",
#     "Covered_Recipient_Last_Name": "Last_Name",
#     "Covered_Recipient_Specialty_1": "Specialty",
#     "Recipient_City": "City",
#     "Recipient_State": "State",
#     "Recipient_Zip_Code": "Zip",
#     "Total_Amount_of_Payment_USDollars": "Payment_Amount",
#     "Date_of_Payment": "Payment_Date",
#     "Form_of_Payment_or_Transfer_of_Value": "Payment_Form",
#     "Nature_of_Payment_or_Transfer_of_Value": "Payment_Nature",
#     "Submitting_Applicable_Manufacturer_or_Applicable_GPO_Name": "Manufacturer",
#     "Name_of_Drug_or_Biological_or_Device_or_Medical_Supply_1": "Drug_or_Device",
#     "Product_Category_or_Therapeutic_Area_1": "Therapeutic_Area",
#     "Physician_Ownership_Indicator": "Ownership",
#     "Third_Party_Payment_Recipient_Indicator": "Third_Party_Recipient",
#     "Charity_Indicator": "Charity",
#     "Dispute_Status_for_Publication": "Dispute_Status"
# }

# for old, new in renamed_cols.items():
#     payments_selected = payments_selected.withColumnRenamed(old, new)

# # -------------------------------
# # Step 5: Save as single CSV
# # -------------------------------
# temp_folder = "/Users/architgupta280/Desktop/LTI/Project/Data/temp_clean_payments"
# final_csv = "/Users/architgupta280/Desktop/LTI/Project/Data/clean_payments.csv"

# # Write as single part file in temp folder
# payments_selected.coalesce(1).write.csv(temp_folder, header=True, mode="overwrite")

# # Move and rename part file
# #  to final CSV
# part_file = glob.glob(os.path.join(temp_folder, "part-*.csv"))[0]
# shutil.move(part_file, final_csv)
# shutil.rmtree(temp_folder)

# print("Cleaned payments dataset saved as single CSV successfully!")








# PRINTING THE SHAPE AND FIRST 5 ROWS OF THE CLEANED DATASET:
# from pyspark.sql import SparkSession

# # -------------------------------
# # Step 1: Initialize Spark
# # -------------------------------
# spark = SparkSession.builder.appName("PaymentsDatasetCheck").getOrCreate()

# # -------------------------------
# # Step 2: Load clean payments dataset
# # -------------------------------
# payments_file = "/Users/architgupta280/Desktop/LTI/Project/Data/clean_payments.csv"
# payments_df = spark.read.csv(payments_file, header=True, inferSchema=True)

# # -------------------------------
# # Step 3: Dataset shape
# # -------------------------------
# row_count = payments_df.count()
# col_count = len(payments_df.columns)
# print("\n===== DATASET SHAPE =====")
# print(f"Rows: {row_count}")
# print(f"Columns: {col_count}")

# # -------------------------------
# # Step 4: First 5 rows (vertical display)
# # -------------------------------
# print("\n===== FIRST 5 ROWS =====")
# for i, row in enumerate(payments_df.take(5), 1):
#     print(f"\n--- Row {i} ---")
#     for col_name in payments_df.columns:
#         print(f"{col_name}: {row[col_name]}")
















# CHECK FOR NON NUMERIC VALUES:
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col

# # -------------------------------
# # Step 1: Initialize Spark
# # -------------------------------
# spark = SparkSession.builder.appName("NonNumericSample").getOrCreate()

# # -------------------------------
# # Step 2: Load CSV
# # -------------------------------
# file_path = "/Users/architgupta280/Desktop/LTI/Project/Data/clean_payments.csv"
# df = spark.read.csv(file_path, header=True, inferSchema=True)

# # -------------------------------
# # Step 3: Define numeric regex
# # -------------------------------
# numeric_regex_payment = r"^-?\d+(\.\d+)?$"
# numeric_regex_zip = r"^\d+$"

# # -------------------------------
# # Step 4: Filter non-numeric rows
# # -------------------------------
# non_numeric_payment_df = df.filter(~col("Payment_Amount").rlike(numeric_regex_payment))
# non_numeric_zip_df = df.filter(~col("Zip").rlike(numeric_regex_zip))

# # -------------------------------
# # Step 5: Count and show samples
# # -------------------------------
# print(f"Non-numeric rows in Payment_Amount: {non_numeric_payment_df.count()}")
# print("Sample non-numeric Payment_Amount values:")
# non_numeric_payment_df.select("Payment_Amount").show(10, truncate=False)

# print(f"\nNon-numeric rows in Zip: {non_numeric_zip_df.count()}")
# print("Sample non-numeric Zip values:")
# non_numeric_zip_df.select("Zip").show(10, truncate=False)










import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, when, trim
import os
import shutil
import glob

from config import CLEAN_PAYMENTS_CSV, data_path

# -------------------------------
# Step 1: Initialize Spark
# -------------------------------
spark = SparkSession.builder.appName("CleanPayments").getOrCreate()

# -------------------------------
# Step 2: Load cleaned CSV
# -------------------------------
file_path = str(CLEAN_PAYMENTS_CSV)
df = spark.read.csv(file_path, header=True, inferSchema=False)

# -------------------------------
# Step 3: Fix Payment_Amount
# -------------------------------
df = df.withColumn("Payment_Amount_cleaned", trim(regexp_extract(col("Payment_Amount"), r'[\d\.]+', 0)))
df = df.withColumn(
    "Payment_Amount",
    when(col("Payment_Amount_cleaned") != "", col("Payment_Amount_cleaned").cast("double")).otherwise(0.0)
).drop("Payment_Amount_cleaned")

# -------------------------------
# Step 4: Fix Zip (keep first 5 digits)c
# -------------------------------
df = df.withColumn("Zip", regexp_extract(col("Zip"), r'^(\d{5})', 1))

# -------------------------------
# Step 5: Count remaining non-numeric rows
# -------------------------------
non_numeric_payment_count = df.filter(~col("Payment_Amount").cast("double").isNotNull()).count()
non_numeric_zip_count = df.filter(~col("Zip").rlike(r"^\d{5}$")).count()

print(f"Remaining non-numeric rows in Payment_Amount after cleaning: {non_numeric_payment_count}")
print(f"Remaining non-numeric rows in Zip after cleaning: {non_numeric_zip_count}")

# -------------------------------
# Step 6: Verify sample rows
# -------------------------------
print("Sample rows after cleaning Payment_Amount and Zip:")
df.select("Payment_Amount", "Zip").show(10, truncate=False)

# -------------------------------
# Step 7: Save cleaned dataset
# -------------------------------
temp_folder = str(data_path("temp_clean_payments"))
output_file = file_path

df.coalesce(1).write.csv(temp_folder, header=True, mode="overwrite")
part_file = glob.glob(os.path.join(temp_folder, "part-*.csv"))[0]
shutil.move(part_file, output_file)
shutil.rmtree(temp_folder)

print(f"Cleaned dataset overwritten at: {output_file}")

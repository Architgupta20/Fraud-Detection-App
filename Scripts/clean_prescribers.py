# from pyspark.sql import SparkSession
# import os
# import shutil
# import glob

# # -------------------------------
# # Step 1: Initialize Spark
# # -------------------------------
# spark = SparkSession.builder.appName("CleanPrescribers").getOrCreate()

# # -------------------------------
# # Step 2: Load Prescribers dataset
# # -------------------------------
# file_path = "/Users/architgupta280/Desktop/LTI/Project/Data/part_d_prescribers.csv"
# df = spark.read.csv(file_path, header=True, inferSchema=True)

# # -------------------------------
# # Step 3: Select important columns
# # -------------------------------
# important_cols = {
#     "PRSCRBR_NPI": "prescriber_id",
#     "Prscrbr_First_Name": "first_name",
#     "Prscrbr_Last_Org_Name": "last_name",
#     "Prscrbr_Crdntls": "credentials",
#     "Prscrbr_Type": "provider_type",
#     "Prscrbr_State_Abrvtn": "state",
#     "Prscrbr_City": "city",
#     "Prscrbr_zip5": "zip",
#     "Tot_Clms": "total_claims",
#     "Tot_Drug_Cst": "total_drug_cost",
#     "Tot_Benes": "total_beneficiaries",
#     "Opioid_Tot_Clms": "opioid_claims",
#     "Opioid_Tot_Drug_Cst": "opioid_cost",
#     "Opioid_Tot_Benes": "opioid_beneficiaries",
#     "Opioid_Prscrbr_Rate": "opioid_rate",
#     "Antbtc_Tot_Clms": "antibiotic_claims",
#     "Antbtc_Tot_Drug_Cst": "antibiotic_cost",
#     "Bene_Avg_Age": "avg_patient_age",
#     "Bene_Feml_Cnt": "female_patients",
#     "Bene_Male_Cnt": "male_patients",
#     "Bene_Avg_Risk_Scre": "avg_risk_score"
# }

# df_selected = df.select([df[k].alias(v) for k, v in important_cols.items()])

# # -------------------------------
# # Step 4: Save as single CSV
# # -------------------------------
# temp_folder = "/Users/architgupta280/Desktop/LTI/Project/Data/temp_clean_prescribers"
# final_csv = "/Users/architgupta280/Desktop/LTI/Project/Data/clean_prescribers.csv"

# # Write as single part file in temp folder
# df_selected.coalesce(1).write.csv(temp_folder, header=True, mode="overwrite")

# # Move and rename part file to final CSV
# part_file = glob.glob(os.path.join(temp_folder, "part-*.csv"))[0]
# shutil.move(part_file, final_csv)
# shutil.rmtree(temp_folder)

# print("Cleaned prescribers dataset saved as single CSV successfully!")

# import os
# import shutil
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, when

# # -------------------------------
# # Initialize Spark
# # -------------------------------
# spark = SparkSession.builder.appName("CleanPrescribers").getOrCreate()

# # -------------------------------
# # Load Prescribers dataset
# # -------------------------------
# file_path = "/Users/architgupta280/Desktop/LTI/Project/Data/clean_prescribers.csv"
# df = spark.read.csv(file_path, header=True, inferSchema=True)

# # -------------------------------
# # Fill None/null numeric columns with 0
# # -------------------------------
# numeric_cols = ["total_claims", "total_drug_cost", "total_beneficiaries",
#                 "opioid_claims", "opioid_cost", "opioid_beneficiaries",
#                 "opioid_rate", "antibiotic_claims", "antibiotic_cost",
#                 "avg_patient_age", "female_patients", "male_patients",
#                 "avg_risk_score"]

# for col_name in numeric_cols:
#     df = df.withColumn(col_name, when(col(col_name).isNull(), 0).otherwise(col(col_name)))

# # -------------------------------
# # Save to a temporary CSV first
# # -------------------------------
# temp_file = "/Users/architgupta280/Desktop/LTI/Project/Data/temp_prescribers.csv"
# df.coalesce(1).write.option("header", True).mode("overwrite").csv(temp_file)

# # -------------------------------
# # Move the single part file to overwrite original CSV
# # -------------------------------
# # Get the actual CSV part file
# for f in os.listdir(temp_file):
#     if f.startswith("part-") and f.endswith(".csv"):
#         shutil.move(os.path.join(temp_file, f), file_path)

# # Remove the temporary folder
# shutil.rmtree(temp_file)

# print("Prescribers CSV updated successfully!")








# # -------------------------------
# # Step 1: Import necessary libraries
# # -------------------------------
# from pyspark.sql import SparkSession

# # -------------------------------
# # Step 2: Initialize Spark
# # -------------------------------
# spark = SparkSession.builder.appName("CleanPrescribersAnalysis").getOrCreate()

# # -------------------------------
# # Step 3: Load your cleaned prescribers dataset
# # -------------------------------
# file_path = "/Users/architgupta280/Desktop/LTI/Project/Data/clean_prescribers.csv"
# clean_prescribers = spark.read.csv(file_path, header=True, inferSchema=True)

# # -------------------------------
# # Step 4: Print the shape of the dataset
# # -------------------------------
# num_rows = clean_prescribers.count()
# num_cols = len(clean_prescribers.columns)
# print(f"Shape of clean_prescribers dataset: ({num_rows}, {num_cols})")

# # -------------------------------
# # Step 5: Show the first 5 rows vertically
# # -------------------------------
# clean_prescribers.show(5, vertical=True)







import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, when

from config import CLEAN_PRESCRIBERS_CSV

# -------------------------------
# Step 1: Initialize Spark
# -------------------------------
spark = SparkSession.builder.appName("CleanPrescribers").getOrCreate()

# -------------------------------
# Step 2: Load cleaned Prescribers dataset
# -------------------------------
file_path = str(CLEAN_PRESCRIBERS_CSV)
df = spark.read.csv(file_path, header=True, inferSchema=True)

# -------------------------------
# Step 3: Print dataset shape
# -------------------------------
num_rows = df.count()
num_cols = len(df.columns)
print("\n===== CLEAN PRESCRIBERS DATASET SHAPE =====")
print(f"Rows: {num_rows}")
print(f"Columns: {num_cols}")

# -------------------------------
# Step 4: Print all column names
# -------------------------------
print("\n===== CLEAN PRESCRIBERS DATASET COLUMNS =====")
for col_name in df.columns:
    print(col_name)

# -------------------------------
# Step 5: Show first 5 rows vertically
# -------------------------------
print("\n===== FIRST 5 ROWS (VERTICAL VIEW) =====")
rows = df.limit(5).collect()
for i, row in enumerate(rows, start=1):
    print(f"\n--- Row {i} ---")
    for col_name, val in row.asDict().items():
        print(f"{col_name}: {val}")

# -------------------------------
# Step 6: Check numeric columns for non-numeric values
# -------------------------------
numeric_cols = [
    "total_claims", "total_drug_cost", "total_beneficiaries",
    "opioid_claims", "opioid_cost", "opioid_beneficiaries",
    "opioid_rate", "antibiotic_claims", "antibiotic_cost",
    "avg_patient_age", "female_patients", "male_patients",
    "avg_risk_score"
]

print("\n===== NON-NUMERIC VALUES IN NUMERIC COLUMNS =====")
for col_name in numeric_cols:
    # Remove valid numbers, see if anything remains
    non_numeric_count = df.filter(regexp_replace(col(col_name), "[0-9.+-Ee]", "") != "").count()
    print(f"Non-numeric rows in {col_name}: {non_numeric_count}")

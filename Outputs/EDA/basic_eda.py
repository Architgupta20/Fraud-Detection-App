from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum

# -------------------------------
# Step 1: Initialize Spark
# -------------------------------
spark = SparkSession.builder.appName("HealthcareEDA").getOrCreate()

# -------------------------------
# Step 2: Load merged dataset
# -------------------------------
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import MERGED_PAYMENT_LEVEL_CSV

merged_file = str(MERGED_PAYMENT_LEVEL_CSV)
merged_df = spark.read.csv(merged_file, header=True, inferSchema=True)

# -------------------------------
# Step 3: Shape of dataset
# -------------------------------
row_count = merged_df.count()
col_count = len(merged_df.columns)
print("\n===== DATASET SHAPE =====")
print(f"Rows: {row_count}")
print(f"Columns: {col_count}")

# -------------------------------
# Step 4: Schema (vertical display)
# -------------------------------
print("\n===== DATASET SCHEMA =====")
for col_name, dtype in merged_df.dtypes:
    print(f"{col_name}: {dtype}")

# -------------------------------
# Step 5: Missing values per column (vertical)
# -------------------------------
print("\n===== MISSING VALUES =====")
missing = merged_df.select([
    sum(col(c).isNull().cast("int")).alias(c) for c in merged_df.columns
]).collect()[0]
for col_name in merged_df.columns:
    print(f"{col_name}: {missing[col_name]}")

# -------------------------------
# Step 6: Summary statistics (vertical)
# -------------------------------
print("\n===== SUMMARY STATISTICS =====")
numeric_cols = [c for c, t in merged_df.dtypes if t in ("int", "double", "float", "long")]
summary = merged_df.describe(numeric_cols).collect()

for row in summary:
    print(f"\n--- {row['summary']} ---")
    for col_name in numeric_cols:
        print(f"{col_name}: {row[col_name]}")

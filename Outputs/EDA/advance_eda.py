from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, mean as _mean, stddev as _stddev

# -------------------------------
# Step 1: Initialize Spark
# -------------------------------
spark = SparkSession.builder.appName("HealthcareEDA_Advanced").getOrCreate()

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
# Step 3: Unique values per column
# -------------------------------
print("\n===== UNIQUE VALUES PER COLUMN =====")
for col_name in merged_df.columns:
    unique_count = merged_df.select(col_name).distinct().count()
    print(f"{col_name}: {unique_count} unique values")

# -------------------------------
# Step 4: Top 10 frequent values for key categorical columns
# -------------------------------
categorical_cols = ["State", "Specialty", "Provider_Type", "Zip"]
print("\n===== TOP 10 FREQUENT VALUES IN CATEGORICAL COLUMNS =====")
for col_name in categorical_cols:
    print(f"\nColumn: {col_name}")
    merged_df.groupBy(col_name).count().orderBy(col("count").desc()).show(10, truncate=False)

# -------------------------------
# Step 5: Outlier detection in Payment_Amount
# -------------------------------
stats = merged_df.select(
    _mean(col("Payment_Amount")).alias("mean"),
    _stddev(col("Payment_Amount")).alias("std")
).collect()[0]

mean_val, std_val = stats["mean"], stats["std"]
outliers_df = merged_df.filter(
    (col("Payment_Amount") > mean_val + 3*std_val) | 
    (col("Payment_Amount") < mean_val - 3*std_val)
)

print(f"\n===== OUTLIERS IN PAYMENT_AMOUNT =====")
print(f"Mean: {mean_val}, StdDev: {std_val}")
print(f"Number of outlier rows: {outliers_df.count()}")
outliers_df.select("NPI", "Payment_Amount", "Payment_Date").show(10, truncate=False)

# -------------------------------
# Step 6: Duplicate row check
# -------------------------------
duplicate_count = merged_df.count() - merged_df.distinct().count()
print(f"\n===== DUPLICATE ROWS =====")
print(f"Duplicate rows count: {duplicate_count}")

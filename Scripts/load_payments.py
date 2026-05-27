import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pyspark.sql import SparkSession

from config import OPEN_PAYMENTS_CSV

# Start Spark session
spark = SparkSession.builder.appName("PaymentsEDA").getOrCreate()

# File path
payments_path = str(OPEN_PAYMENTS_CSV)

# Load Payments dataset
payments = spark.read.csv(payments_path, header=True, inferSchema=True)

# Step 1: Print dataset shape
num_rows = payments.count()
num_cols = len(payments.columns)
print("\n===== OPEN PAYMENTS DATASET SHAPE =====")
print(f"Rows: {num_rows}")
print(f"Columns: {num_cols}")

# Step 2: Print all column names
print("\n===== OPEN PAYMENTS DATASET COLUMNS =====")
for col in payments.columns:
    print(col)

# Step 3: Show first 10 rows of the dataset
print("\n===== FIRST 10 ROWS OF OPEN PAYMENTS DATASET =====")
payments.show(10, truncate=False, vertical=True)

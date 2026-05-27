import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pyspark.sql import SparkSession

from config import PART_D_PRESCRIBERS_CSV

# Start Spark session
spark = SparkSession.builder.appName("PrescribersEDA").getOrCreate()

# File path
prescribers_path = str(PART_D_PRESCRIBERS_CSV)

# Load Prescribers dataset
prescribers = spark.read.csv(prescribers_path, header=True, inferSchema=True)

# Step 1: Print dataset shape
num_rows = prescribers.count()
num_cols = len(prescribers.columns)
print("\n===== PRESCRIBERS DATASET SHAPE =====")
print(f"Rows: {num_rows}")
print(f"Columns: {num_cols}")

# Step 2: Print all column names
print("\n===== PRESCRIBERS DATASET COLUMNS =====")
for col in prescribers.columns:
    print(col)
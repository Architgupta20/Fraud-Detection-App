from pyspark.sql import SparkSession
from pyspark.sql.functions import col, desc
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Step 1: Initialize Spark
# -------------------------------
spark = SparkSession.builder.appName("HealthcareEDA_Graphs").getOrCreate()

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
# 1. Distribution of Payment_Amount
# -------------------------------
payment_sample = merged_df.select("Payment_Amount").dropna().limit(10000).toPandas()

plt.figure(figsize=(10,6))
plt.hist(payment_sample["Payment_Amount"], bins=50, color="blue", edgecolor="black")
plt.title("Distribution of Payment Amounts")
plt.xlabel("Payment Amount")
plt.ylabel("Frequency")
plt.show()

# -------------------------------
# 2. Top 10 States by Record Count
# -------------------------------
state_counts = (
    merged_df.filter(col("State").isNotNull())
    .groupBy("State")
    .count()
    .orderBy(desc("count"))
    .limit(10)
    .toPandas()
)

plt.figure(figsize=(10,6))
plt.bar(state_counts["State"].astype(str), state_counts["count"], color="green")
plt.title("Top 10 States by Record Count")
plt.xlabel("State")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.show()

# -------------------------------
# 3. Top 10 Provider Types
# -------------------------------
provider_counts = (
    merged_df.filter(col("Provider_Type").isNotNull())
    .groupBy("Provider_Type")
    .count()
    .orderBy(desc("count"))
    .limit(10)
    .toPandas()
)

plt.figure(figsize=(12,6))
plt.bar(provider_counts["Provider_Type"].astype(str), provider_counts["count"], color="orange")
plt.title("Top 10 Provider Types by Record Count")
plt.xlabel("Provider Type")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.show()

# -------------------------------
# 4. Top 10 Specialties
# -------------------------------
specialty_counts = (
    merged_df.filter(col("Specialty").isNotNull())
    .groupBy("Specialty")
    .count()
    .orderBy(desc("count"))
    .limit(10)
    .toPandas()
)

plt.figure(figsize=(12,6))
plt.bar(specialty_counts["Specialty"].astype(str), specialty_counts["count"], color="purple")
plt.title("Top 10 Specialties by Record Count")
plt.xlabel("Specialty")
plt.ylabel("Count")
plt.xticks(rotation=75, ha="right")
plt.show()

# -------------------------------
# 5. Top 10 Zip Codes
# -------------------------------
zip_counts = (
    merged_df.filter(col("Zip").isNotNull())
    .groupBy("Zip")
    .count()
    .orderBy(desc("count"))
    .limit(10)
    .toPandas()
)

plt.figure(figsize=(10,6))
plt.bar(zip_counts["Zip"].astype(str), zip_counts["count"], color="red")
plt.title("Top 10 Zip Codes by Record Count")
plt.xlabel("Zip Code")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.show()


# -------------------------------
# Correlation Matrix Heatmap
# -------------------------------
plt.figure(figsize=(10,6))
sns.heatmap(merged_df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()

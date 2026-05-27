#!/usr/bin/env python3
"""
Runnable ETL pipeline for prescriber risk scoring.

Stages:
  clean      - raw CMS CSVs -> clean_prescribers.csv, clean_payments.csv
  aggregate  - prescriber-level merge + payment-level merge
  features   - engineered columns -> prescriber_level_enriched.csv
  score      - rule-based labels -> fraud_risk_scored_prescribers.csv
  all        - run clean -> aggregate -> features -> score

Examples:
  python run_pipeline.py clean
  python run_pipeline.py all
  python run_pipeline.py score --skip-missing-inputs
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    avg,
    col,
    count,
    expr,
    lit,
    max as spark_max,
    regexp_extract,
    sum as spark_sum,
    trim,
    when,
)
from pyspark.sql.functions import avg as spark_avg

from config import (
    BASE_DIR,
    CLEAN_PAYMENTS_CSV,
    CLEAN_PRESCRIBERS_CSV,
    FRAUD_RISK_SCORED_CSV,
    MERGED_PAYMENT_LEVEL_CSV,
    OPEN_PAYMENTS_CSV,
    PART_D_PRESCRIBERS_CSV,
    PRESCRIBER_LEVEL_CSV,
    PRESCRIBER_LEVEL_ENRICHED_CSV,
)


def get_spark(app_name: str = "PrescriberRiskPipeline") -> SparkSession:
    return SparkSession.builder.appName(app_name).getOrCreate()


def save_single_csv(df: DataFrame, final_path: Path, order_by: str | None = None) -> None:
    """Write a Spark DataFrame to a single CSV file."""
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = final_path.with_suffix(final_path.suffix + ".spark_tmp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    out = df.orderBy(col(order_by).asc()) if order_by and order_by in df.columns else df
    out.coalesce(1).write.mode("overwrite").option("header", True).csv(str(temp_dir))

    part_files = glob.glob(str(temp_dir / "part-*.csv"))
    if not part_files:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"No part file written under {temp_dir}")

    if final_path.exists():
        final_path.unlink()
    shutil.move(part_files[0], final_path)
    shutil.rmtree(temp_dir)
    print(f"Saved: {final_path}")


def require_file(path: Path, stage: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{stage} requires input file: {path}")


def stage_clean(spark: SparkSession) -> None:
    """Raw CMS files -> clean_prescribers.csv, clean_payments.csv."""
    require_file(PART_D_PRESCRIBERS_CSV, "clean")
    require_file(OPEN_PAYMENTS_CSV, "clean")

    prescribers = spark.read.csv(str(PART_D_PRESCRIBERS_CSV), header=True, inferSchema=True)
    important_cols = {
        "PRSCRBR_NPI": "prescriber_id",
        "Prscrbr_First_Name": "first_name",
        "Prscrbr_Last_Org_Name": "last_name",
        "Prscrbr_Crdntls": "credentials",
        "Prscrbr_Type": "provider_type",
        "Prscrbr_State_Abrvtn": "state",
        "Prscrbr_City": "city",
        "Prscrbr_zip5": "zip",
        "Tot_Clms": "total_claims",
        "Tot_Drug_Cst": "total_drug_cost",
        "Tot_Benes": "total_beneficiaries",
        "Opioid_Tot_Clms": "opioid_claims",
        "Opioid_Tot_Drug_Cst": "opioid_cost",
        "Opioid_Tot_Benes": "opioid_beneficiaries",
        "Opioid_Prscrbr_Rate": "opioid_rate",
        "Antbtc_Tot_Clms": "antibiotic_claims",
        "Antbtc_Tot_Drug_Cst": "antibiotic_cost",
        "Bene_Avg_Age": "avg_patient_age",
        "Bene_Feml_Cnt": "female_patients",
        "Bene_Male_Cnt": "male_patients",
        "Bene_Avg_Risk_Scre": "avg_risk_score",
    }
    prescribers_selected = prescribers.select(
        [prescribers[k].alias(v) for k, v in important_cols.items()]
    )
    save_single_csv(prescribers_selected, CLEAN_PRESCRIBERS_CSV)

    payments = spark.read.csv(str(OPEN_PAYMENTS_CSV), header=True, inferSchema=True)
    important_payment_cols = [
        "Covered_Recipient_NPI",
        "Covered_Recipient_First_Name",
        "Covered_Recipient_Last_Name",
        "Covered_Recipient_Specialty_1",
        "Recipient_City",
        "Recipient_State",
        "Recipient_Zip_Code",
        "Total_Amount_of_Payment_USDollars",
        "Date_of_Payment",
        "Form_of_Payment_or_Transfer_of_Value",
        "Nature_of_Payment_or_Transfer_of_Value",
        "Submitting_Applicable_Manufacturer_or_Applicable_GPO_Name",
        "Name_of_Drug_or_Biological_or_Device_or_Medical_Supply_1",
        "Product_Category_or_Therapeutic_Area_1",
        "Physician_Ownership_Indicator",
        "Third_Party_Payment_Recipient_Indicator",
        "Charity_Indicator",
        "Dispute_Status_for_Publication",
    ]
    payments_selected = payments.select(important_payment_cols)
    renamed = {
        "Covered_Recipient_NPI": "NPI",
        "Covered_Recipient_First_Name": "First_Name",
        "Covered_Recipient_Last_Name": "Last_Name",
        "Covered_Recipient_Specialty_1": "Specialty",
        "Recipient_City": "City",
        "Recipient_State": "State",
        "Recipient_Zip_Code": "Zip",
        "Total_Amount_of_Payment_USDollars": "Payment_Amount",
        "Date_of_Payment": "Payment_Date",
        "Form_of_Payment_or_Transfer_of_Value": "Payment_Form",
        "Nature_of_Payment_or_Transfer_of_Value": "Payment_Nature",
        "Submitting_Applicable_Manufacturer_or_Applicable_GPO_Name": "Manufacturer",
        "Name_of_Drug_or_Biological_or_Device_or_Medical_Supply_1": "Drug_or_Device",
        "Product_Category_or_Therapeutic_Area_1": "Therapeutic_Area",
        "Physician_Ownership_Indicator": "Ownership",
        "Third_Party_Payment_Recipient_Indicator": "Third_Party_Recipient",
        "Charity_Indicator": "Charity",
        "Dispute_Status_for_Publication": "Dispute_Status",
    }
    for old, new in renamed.items():
        payments_selected = payments_selected.withColumnRenamed(old, new)

    # Normalize payment amount and zip (from Scripts/clean_payments.py)
    payments_selected = payments_selected.withColumn(
        "Payment_Amount_cleaned", trim(regexp_extract(col("Payment_Amount"), r"[\d\.]+", 0))
    ).withColumn(
        "Payment_Amount",
        when(col("Payment_Amount_cleaned") != "", col("Payment_Amount_cleaned").cast("double")).otherwise(
            0.0
        ),
    ).drop("Payment_Amount_cleaned")
    payments_selected = payments_selected.withColumn("Zip", regexp_extract(col("Zip"), r"^(\d{5})", 1))

    save_single_csv(payments_selected, CLEAN_PAYMENTS_CSV)


def _clean_prescriber_numerics(df: DataFrame) -> DataFrame:
    numeric_cols = [
        "total_claims",
        "total_drug_cost",
        "total_beneficiaries",
        "opioid_claims",
        "opioid_cost",
        "opioid_beneficiaries",
        "avg_patient_age",
        "female_patients",
        "male_patients",
        "avg_risk_score",
        "antibiotic_claims",
        "antibiotic_cost",
        "opioid_rate",
    ]
    for col_name in numeric_cols:
        if col_name in df.columns:
            df = df.withColumn(
                col_name,
                when((col(col_name).isNull()) | (col(col_name) == "None"), 0)
                .otherwise(
                    when(col(col_name).rlike("^[0-9.]+$"), col(col_name).cast("double")).otherwise(0)
                ),
            )
    string_cols = [c for c in df.columns if c not in numeric_cols]
    return df.fillna("NA", subset=string_cols)


def stage_aggregate(spark: SparkSession) -> None:
    """Build prescriber_level_dataset.csv and merged_payment_level_dataset.csv."""
    require_file(CLEAN_PRESCRIBERS_CSV, "aggregate")
    require_file(CLEAN_PAYMENTS_CSV, "aggregate")

    prescribers_df = spark.read.csv(str(CLEAN_PRESCRIBERS_CSV), header=True, inferSchema=False)
    payments_df = spark.read.csv(str(CLEAN_PAYMENTS_CSV), header=True, inferSchema=False)

    prescribers_df = _clean_prescriber_numerics(prescribers_df)
    payments_df = payments_df.withColumn(
        "Payment_Amount",
        when((col("Payment_Amount").isNull()) | (col("Payment_Amount") == "None"), 0).otherwise(
            when(col("Payment_Amount").rlike("^[0-9.]+$"), col("Payment_Amount").cast("double")).otherwise(0)
        ),
    )

    aggregated_payments = payments_df.groupBy("NPI").agg(
        spark_sum("Payment_Amount").alias("total_payment_amount"),
        count("Payment_Amount").alias("num_payments"),
        spark_avg("Payment_Amount").alias("avg_payment_amount"),
        spark_max("Payment_Amount").alias("max_payment_amount"),
    )

    merged_df = prescribers_df.join(
        aggregated_payments,
        prescribers_df["prescriber_id"] == aggregated_payments["NPI"],
        "left",
    ).drop("NPI")
    merged_df = merged_df.fillna(
        {
            "total_payment_amount": 0,
            "num_payments": 0,
            "avg_payment_amount": 0,
            "max_payment_amount": 0,
        }
    )
    save_single_csv(merged_df, PRESCRIBER_LEVEL_CSV, order_by="prescriber_id")

    prescribers_for_join = prescribers_df.drop("first_name", "last_name", "city", "state", "zip")
    prescribers_for_join = prescribers_for_join.withColumn(
        "prescriber_id_num", expr("try_cast(prescriber_id as bigint)")
    )
    payments_for_join = payments_df.withColumn("NPI_num", expr("try_cast(NPI as bigint)"))
    prescribers_for_join = prescribers_for_join.filter(col("prescriber_id_num").isNotNull())
    payments_for_join = payments_for_join.filter(col("NPI_num").isNotNull())

    payment_level = payments_for_join.join(
        prescribers_for_join,
        payments_for_join["NPI_num"] == prescribers_for_join["prescriber_id_num"],
        "left",
    ).drop("NPI_num", "prescriber_id_num")
    save_single_csv(payment_level, MERGED_PAYMENT_LEVEL_CSV)


def stage_features(spark: SparkSession) -> None:
    """Engineer features -> prescriber_level_enriched.csv."""
    require_file(PRESCRIBER_LEVEL_CSV, "features")
    df = spark.read.csv(str(PRESCRIBER_LEVEL_CSV), header=True, inferSchema=True)

    df = df.withColumn(
        "payment_per_claim",
        when(col("total_claims") > 0, col("total_payment_amount") / col("total_claims")).otherwise(0.0),
    )
    df = df.withColumn(
        "payment_to_drug_cost_ratio",
        when(col("total_drug_cost") > 0, col("total_payment_amount") / col("total_drug_cost")).otherwise(0.0),
    )
    df = df.withColumn(
        "opioid_claim_ratio",
        when(col("total_claims") > 0, col("opioid_claims") / col("total_claims")).otherwise(0.0),
    )
    df = df.withColumn(
        "antibiotic_claim_ratio",
        when(col("total_claims") > 0, col("antibioid_claims") / col("total_claims")).otherwise(0.0),
    )
    df = df.withColumn("high_opioid_flag", when(col("opioid_claim_ratio") > 0.5, lit(1)).otherwise(lit(0)))
    df = df.withColumn(
        "payment_variability",
        when(col("avg_payment_amount") > 0, col("max_payment_amount") / col("avg_payment_amount")).otherwise(0.0),
    )
    df = df.withColumn(
        "high_payment_flag", when(col("avg_payment_amount") > 1000, lit(1)).otherwise(lit(0))
    )
    df = df.withColumn(
        "female_to_male_ratio",
        when(col("male_patients") > 0, col("female_patients") / col("male_patients")).otherwise(0.0),
    )
    df = df.withColumn("elderly_focus_flag", when(col("avg_patient_age") > 70, lit(1)).otherwise(lit(0)))
    df = df.withColumn("adjusted_risk_payment", col("total_payment_amount") * col("avg_risk_score"))

    provider_avg = df.groupBy("provider_type").agg(avg("avg_payment_amount").alias("peer_avg_payment"))
    df = df.join(provider_avg, on="provider_type", how="left")
    df = df.withColumn(
        "peer_deviation_score",
        when(col("peer_avg_payment") > 0, col("avg_payment_amount") / col("peer_avg_payment")).otherwise(0.0),
    )

    save_single_csv(df, PRESCRIBER_LEVEL_ENRICHED_CSV, order_by="prescriber_id")
    print(f"Rows: {df.count()}, Columns: {len(df.columns)}")


def stage_score(spark: SparkSession) -> None:
    """Apply rule-based risk labels -> fraud_risk_scored_prescribers.csv."""
    require_file(PRESCRIBER_LEVEL_ENRICHED_CSV, "score")
    df = spark.read.csv(str(PRESCRIBER_LEVEL_ENRICHED_CSV), header=True, inferSchema=True)

    df = df.withColumn(
        "fraud_risk_score",
        (
            when(col("payment_to_drug_cost_ratio") > 1, 2)
            .when(col("opioid_claims") > 100, 2)
            .when(col("high_payment_flag") == 1, 2)
            .when(col("high_opioid_flag") == 1, 2)
            .when(col("peer_deviation_score") > 5, 2)
            .when(col("elderly_focus_flag") == 1, 1)
            .otherwise(0)
        ),
    )
    df = df.withColumn(
        "fraud_risk_category",
        when(col("fraud_risk_score") >= 2, "High")
        .when(col("fraud_risk_score") == 1, "Medium")
        .otherwise("Low"),
    )

    print("\n===== FRAUD RISK CATEGORY DISTRIBUTION =====")
    df.groupBy("fraud_risk_category").count().show()

    save_single_csv(df, FRAUD_RISK_SCORED_CSV, order_by="prescriber_id")


STAGES = {
    "clean": stage_clean,
    "aggregate": stage_aggregate,
    "features": stage_features,
    "score": stage_score,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run prescriber risk ETL pipeline stages.")
    parser.add_argument(
        "stage",
        choices=list(STAGES.keys()) + ["all"],
        help="Pipeline stage to run (or 'all' for clean -> score)",
    )
    parser.add_argument(
        "--app-name",
        default="PrescriberRiskPipeline",
        help="Spark application name",
    )
    args = parser.parse_args()

    print(f"BASE_DIR: {BASE_DIR}")
    spark = get_spark(args.app_name)
    try:
        if args.stage == "all":
            for name in ("clean", "aggregate", "features", "score"):
                print(f"\n{'=' * 60}\nRunning stage: {name}\n{'=' * 60}")
                STAGES[name](spark)
        else:
            STAGES[args.stage](spark)
    finally:
        spark.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())

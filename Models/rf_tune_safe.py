# rf_tune_safe_combined.py
"""
Random Forest tuning (safe mode) + combined oversampling & undersampling on training set.
Balances classes to median, tunes with TrainValidationSplit (light), saves predictions
and confusion matrix image.
"""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import glob
import shutil
import traceback
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, hash as hash_func
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder

# optional plotting libs
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOTTING = True
except Exception:
    HAS_PLOTTING = False

from config import FRAUD_RISK_SCORED_CSV, data_path, model_data_path

# ----------------------------
# CONFIG (edit paths if needed)
# ----------------------------
INPUT_CSV = str(FRAUD_RISK_SCORED_CSV)
TEMP_DIR = str(data_path("_temp_rf_combined"))
FINAL_OUTPUT_CSV = str(model_data_path("fraud_detection_rf_combined_predictions.csv"))
CONFUSION_PNG = str(model_data_path("confusion_matrix_rf_combined.png"))

USE_ELDERLY_FLAG = False

# Spark / resource safety
SPARK_DRIVER_MEMORY = "6g"
SHUFFLE_PARTITIONS = "200"
PARALLELISM = 1
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
MAX_OVERSAMPLE_MULTIPLIER = 10.0  # cap oversampling fraction to avoid runaway replication

# Small RF hyperparameter grid (safe for local)
RF_NUM_TREES = [30, 50]
RF_MAX_DEPTHS = [6, 10]
RF_FEATURE_SUBSETS = ["auto"]  # leave simple; you can add ["sqrt","log2"] if you want

# ----------------------------
# START SPARK
# ----------------------------
spark = (
    SparkSession.builder.appName("RF_Tune_Safe_Combined")
    .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
    .config("spark.sql.shuffle.partitions", SHUFFLE_PARTITIONS)
    .config("spark.default.parallelism", PARALLELISM)
    .getOrCreate()
)

def resample_combined(train_df, label_col="fraud_label", seed=42):
    """Combined oversampling + undersampling to median class count."""
    counts = train_df.groupBy(label_col).count().collect()
    counts_map = {r[label_col]: r["count"] for r in counts}
    labels = sorted(counts_map.keys())
    # median target = integer rounded average across classes (robust enough)
    median_c = int(sum(counts_map.values()) / len(counts_map))
    print(f"Class counts before resample: {counts_map}, median target = {median_c}")

    def df_for_label(l): return train_df.filter(col(label_col) == l)

    new_parts = []
    for l in labels:
        df_l = df_for_label(l)
        c = counts_map[l]
        if c == median_c:
            new_parts.append(df_l)
        elif c > median_c:
            frac = float(median_c) / float(c)
            print(f"Undersampling label {l}: {c} -> {int(c*frac)} (fraction={frac:.3f})")
            # sample without replacement
            new_parts.append(df_l.sample(withReplacement=False, fraction=frac, seed=seed))
        else:
            frac = float(median_c) / float(c)
            if frac > MAX_OVERSAMPLE_MULTIPLIER:
                frac = MAX_OVERSAMPLE_MULTIPLIER
                print(f"Oversample capped for label {l}")
            print(f"Oversampling label {l}: {c} -> {int(c*frac)} (fraction={frac:.3f})")
            # sample with replacement to increase minority
            new_parts.append(df_l.sample(withReplacement=True, fraction=frac, seed=seed))
    # union all parts
    res = new_parts[0]
    for p in new_parts[1:]:
        res = res.union(p)

    print("Resampled counts:")
    res.groupBy(label_col).count().orderBy(label_col).show(truncate=False)
    return res

def save_single_csv_from_spark(df, temp_dir, final_path):
    """Write df.coalesce(1) to temp_dir then move single part-*.csv to final_path."""
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(temp_dir)
    part_files = glob.glob(os.path.join(temp_dir, "part-*.csv"))
    if not part_files:
        print("Warning: no part file found in temp write directory. Prediction save skipped.")
        return False
    part_file = part_files[0]
    dst_dir = os.path.dirname(final_path)
    if dst_dir and not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)
    if os.path.exists(final_path):
        os.remove(final_path)
    shutil.move(part_file, final_path)
    shutil.rmtree(temp_dir)
    return True

def plot_and_save_confusion(cm_df, out_path, labels=[0,1,2]):
    """cm_df: pandas DataFrame with columns ['fraud_label','prediction','count']"""
    matrix = pd.DataFrame(0, index=labels, columns=labels)
    for _, r in cm_df.iterrows():
        matrix.at[int(r["fraud_label"]), int(r["prediction"])] = int(r["count"])
    plt.figure(figsize=(6,5))
    im = plt.imshow(matrix.values, interpolation='nearest', cmap='Blues')
    plt.title("Confusion Matrix (RF combined resample)")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_labels = ["Low (0)", "Medium (1)", "High (2)"]
    plt.xticks(np.arange(len(labels)), tick_labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(labels)), tick_labels)
    thresh = matrix.values.max() / 2.0 if matrix.values.max() > 0 else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            c = matrix.values[i, j]
            plt.text(j, i, f"{c}", horizontalalignment="center",
                     color="white" if c > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Confusion matrix image saved to: {out_path}")

def main():
    try:
        # Load dataset
        df = spark.read.csv(INPUT_CSV, header=True, inferSchema=True)
        print("\n===== Loaded Dataset =====")
        print(f"Rows: {df.count()}, Columns: {len(df.columns)}")

        if "fraud_risk_category" not in df.columns:
            raise ValueError("Expected column 'fraud_risk_category' in input CSV. Run fraud scoring first.")

        # Map label -> numeric
        df = df.withColumn(
            "fraud_label",
            when(col("fraud_risk_category") == "Low", 0)
            .when(col("fraud_risk_category") == "Medium", 1)
            .otherwise(2)
        )

        # Features list (exclude elderly flag optionally)
        base_features = [
            "total_claims", "total_drug_cost", "opioid_claims", "opioid_cost",
            "antibiotic_claims", "payment_to_drug_cost_ratio", "peer_deviation_score",
            "avg_risk_score", "payment_variability", "adjusted_risk_payment",
            "high_payment_flag", "high_opioid_flag"
        ]
        feature_cols = base_features + (["elderly_focus_flag"] if USE_ELDERLY_FLAG else [])

        # Cast/create feature cols to double to avoid type issues
        for c in set(feature_cols):
            if c in df.columns:
                df = df.withColumn(c, col(c).cast("double"))
            else:
                df = df.withColumn(c, when(col(c).isNull(), 0.0).otherwise(col(c)).cast("double"))

        print("\n===== Original Class Distribution =====")
        df.groupBy("fraud_risk_category").count().orderBy("count", ascending=False).show(truncate=False)

        # Split by prescriber_id hashing to avoid leakage
        if "prescriber_id" in df.columns:
            df = df.withColumn("_split", (hash_func(col("prescriber_id")) % 10000) / 10000.0)
            train_df = df.filter(col("_split") < TRAIN_RATIO).drop("_split")
            test_df = df.filter(col("_split") >= TRAIN_RATIO).drop("_split")
        else:
            train_df, test_df = df.randomSplit([TRAIN_RATIO, 1.0 - TRAIN_RATIO], seed=RANDOM_SEED)

        print(f"\nTrain rows: {train_df.count()}, Test rows: {test_df.count()}")

        # Combined resampling on training set
        train_df = resample_combined(train_df, label_col="fraud_label", seed=RANDOM_SEED)

        # Pipeline: assembler -> RandomForest
        assembler = VectorAssembler(inputCols=list(set(feature_cols)), outputCol="features", handleInvalid="keep")
        rf = RandomForestClassifier(labelCol="fraud_label", featuresCol="features", seed=RANDOM_SEED)
        pipeline = Pipeline(stages=[assembler, rf])

        # Param grid (small)
        paramGrid = (
            ParamGridBuilder()
            .addGrid(rf.numTrees, RF_NUM_TREES)
            .addGrid(rf.maxDepth, RF_MAX_DEPTHS)
            .addGrid(rf.featureSubsetStrategy, RF_FEATURE_SUBSETS)
            .build()
        )

        evaluator = MulticlassClassificationEvaluator(labelCol="fraud_label", predictionCol="prediction", metricName="f1")

        # TrainValidationSplit (lighter than full CV)
        tvs = TrainValidationSplit(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=evaluator, trainRatio=0.8, parallelism=PARALLELISM)

        print("\nStarting TrainValidationSplit tuning (safe mode) for Random Forest...")
        tvs_model = tvs.fit(train_df)
        print("Tuning complete.")

        # Evaluate on test set
        best_model = tvs_model.bestModel
        predictions = best_model.transform(test_df).cache()

        evaluator_acc = MulticlassClassificationEvaluator(labelCol="fraud_label", predictionCol="prediction", metricName="accuracy")
        evaluator_f1 = MulticlassClassificationEvaluator(labelCol="fraud_label", predictionCol="prediction", metricName="f1")

        accuracy = evaluator_acc.evaluate(predictions)
        f1_score = evaluator_f1.evaluate(predictions)

        print("\n===== MODEL PERFORMANCE (RF Tuned - Safe) =====")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Weighted F1 Score: {f1_score:.4f}")

        # Confusion matrix
        print("\n===== CONFUSION MATRIX (counts) =====")
        cm = predictions.groupBy("fraud_label", "prediction").count().orderBy("fraud_label", "prediction")
        cm.show(truncate=False)

        # Per-class metrics
        print("\n===== PER-CLASS METRICS =====")
        for lbl in [0, 1, 2]:
            tp = predictions.filter((col("fraud_label") == lbl) & (col("prediction") == lbl)).count()
            fp = predictions.filter((col("fraud_label") != lbl) & (col("prediction") == lbl)).count()
            fn = predictions.filter((col("fraud_label") == lbl) & (col("prediction") != lbl)).count()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_local = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            print(f"Label {lbl} -> Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1_local:.3f} (TP={tp}, FP={fp}, FN={fn})")

        # Feature importances (if RandomForestModel available)
        print("\n===== FEATURE IMPORTANCES =====")
        try:
            rf_model = best_model.stages[-1]  # RandomForestClassificationModel
            importances = rf_model.featureImportances.toArray()
            feat_list = list(set(feature_cols))
            for feat, imp in zip(feat_list, importances):
                print(f"{feat}: {imp:.6f}")
        except Exception as e:
            print("Could not extract feature importances:", e)

        # Save predictions as single CSV
        predictions = predictions.withColumn(
            "predicted_category",
            when(col("prediction") == 0, "Low")
            .when(col("prediction") == 1, "Medium")
            .otherwise("High")
        )

        final_cols = [
            "prescriber_id", "first_name", "last_name", "provider_type", "state",
            "total_claims", "total_payment_amount", "payment_to_drug_cost_ratio",
            "peer_deviation_score", "fraud_risk_category", "predicted_category"
        ]

        sorted_preds = predictions.orderBy(col("prescriber_id").asc())
        success = save_single_csv_from_spark(sorted_preds.select(*[c for c in final_cols if c in predictions.columns]), TEMP_DIR, FINAL_OUTPUT_CSV)
        if success:
            print(f"\nSaved predictions: {FINAL_OUTPUT_CSV}")

        # Confusion matrix image (optional)
        if HAS_PLOTTING:
            try:
                cm_pd = cm.toPandas()
                if cm_pd.empty:
                    print("No confusion rows to plot.")
                else:
                    plot_and_save_confusion(cm_pd, CONFUSION_PNG, labels=[0,1,2])
            except Exception as e:
                print("Could not produce confusion plot (error):", e)
        else:
            print("Plotting skipped because pandas/matplotlib not available in environment.")

        # small sample printed vertically
        print("\n===== SAMPLE PREDICTIONS (first 10 rows, vertical) =====")
        sample = sorted_preds.select(*[c for c in final_cols if c in predictions.columns]).limit(10).collect()
        for i, row in enumerate(sample, start=1):
            print(f"\n--- Row {i} ---")
            rdict = row.asDict()
            for c in [c for c in final_cols if c in predictions.columns]:
                print(f"{c}: {rdict.get(c)}")

    except Exception as ex:
        print("\nERROR during execution:")
        traceback.print_exc()
    finally:
        spark.stop()
        print("\nSpark stopped. Done.")

if __name__ == "__main__":
    main()

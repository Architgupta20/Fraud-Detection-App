# gbt_removing_leakage.py
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
from pyspark.sql.functions import col, when, hash as hash_func, lit
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier, OneVsRest
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# optional plotting libs
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
except Exception:
    HAS_PLOTTING = False

from config import FRAUD_RISK_SCORED_CSV, data_path, model_data_path

# ----------------------------
# CONFIG - update paths if needed
# ----------------------------
INPUT_CSV = str(FRAUD_RISK_SCORED_CSV)
TEMP_DIR = str(data_path("_temp_preds_gbt_no_leak"))
FINAL_OUTPUT_CSV = str(model_data_path("fraud_detection_gbt_predictions.csv"))
CONFUSION_PNG = str(model_data_path("confusion_matrix_gbt.png"))

# GBT hyperparameters (tweak if you want)
GBT_MAX_ITER = 50
GBT_MAX_DEPTH = 6
RANDOM_SEED = 42
TRAIN_RATIO = 0.8

# ----------------------------
# Start Spark
# ----------------------------
spark = SparkSession.builder.appName("FraudDetection_GBT_NoLeak").getOrCreate()

def main():
    try:
        # ----------------------------
        # Load data
        # ----------------------------
        df = spark.read.csv(INPUT_CSV, header=True, inferSchema=True)
        print("\n===== Loaded Dataset =====")
        print(f"Rows: {df.count()}, Columns: {len(df.columns)}")

        # ensure label exists
        if "fraud_risk_category" not in df.columns:
            raise ValueError("Expected 'fraud_risk_category' column in input CSV. Run fraud risk scoring first.")

        # numeric label
        df = df.withColumn(
            "fraud_label",
            when(col("fraud_risk_category") == "Low", 0)
            .when(col("fraud_risk_category") == "Medium", 1)
            .otherwise(2)
        )

        # ----------------------------
        # FEATURES: exclude all columns used directly by scoring rules.
        # Rule inputs removed: payment_to_drug_cost_ratio, opioid_claims,
        # high_payment_flag, high_opioid_flag, peer_deviation_score, elderly_focus_flag
        # ----------------------------
        feature_cols = [
            "total_claims",
            "total_drug_cost",
            "opioid_cost",
            "antibiotic_claims",
            "avg_risk_score",
            "payment_variability",
            "adjusted_risk_payment",
        ]

        # Defensive casting / creation: ensure every feature exists and is double
        for c in feature_cols:
            if c in df.columns:
                df = df.withColumn(c, col(c).cast("double"))
            else:
                # create missing numeric column filled with zeros
                df = df.withColumn(c, lit(0.0))

        # Show distribution
        print("\n===== Class Distribution =====")
        df.groupBy("fraud_risk_category").count().orderBy("count", ascending=False).show(truncate=False)

        # ----------------------------
        # Train/test split (hashing prescriber_id to avoid prescriber leakage)
        # ----------------------------
        if "prescriber_id" in df.columns:
            df = df.withColumn("_split", (hash_func(col("prescriber_id")) % 10000) / 10000.0)
            train_df = df.filter(col("_split") < TRAIN_RATIO).drop("_split")
            test_df = df.filter(col("_split") >= TRAIN_RATIO).drop("_split")
        else:
            train_df, test_df = df.randomSplit([TRAIN_RATIO, 1.0 - TRAIN_RATIO], seed=RANDOM_SEED)

        print(f"\nTrain rows: {train_df.count()}, Test rows: {test_df.count()}")

        # ----------------------------
        # Pipeline: VectorAssembler -> OneVsRest(GBT)
        # ----------------------------
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")
        gbt = GBTClassifier(labelCol="fraud_label", featuresCol="features", maxIter=GBT_MAX_ITER,
                            maxDepth=GBT_MAX_DEPTH, seed=RANDOM_SEED)
        ovr = OneVsRest(classifier=gbt, labelCol="fraud_label", featuresCol="features", predictionCol="prediction")
        pipeline = Pipeline(stages=[assembler, ovr])

        print("\nTraining GBT (OneVsRest) ...")
        model = pipeline.fit(train_df)
        print("Model training complete.")

        # ----------------------------
        # Predict & evaluate
        # ----------------------------
        predictions = model.transform(test_df).cache()

        evaluator_acc = MulticlassClassificationEvaluator(labelCol="fraud_label", predictionCol="prediction", metricName="accuracy")
        evaluator_f1 = MulticlassClassificationEvaluator(labelCol="fraud_label", predictionCol="prediction", metricName="f1")

        accuracy = evaluator_acc.evaluate(predictions)
        f1_score = evaluator_f1.evaluate(predictions)

        print("\n===== MODEL PERFORMANCE (GBT No Leakage) =====")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Weighted F1 Score: {f1_score:.4f}")

        # Confusion matrix counts
        print("\n===== CONFUSION MATRIX (counts) =====")
        cm = predictions.groupBy("fraud_label", "prediction").count().orderBy("fraud_label", "prediction")
        cm.show(truncate=False)

        # Per-class metrics
        print("\n===== PER-CLASS METRICS =====")
        per_class_f1 = []
        for lbl in [0, 1, 2]:
            tp = predictions.filter((col("fraud_label") == lbl) & (col("prediction") == lbl)).count()
            fp = predictions.filter((col("fraud_label") != lbl) & (col("prediction") == lbl)).count()
            fn = predictions.filter((col("fraud_label") == lbl) & (col("prediction") != lbl)).count()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_local = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            per_class_f1.append(f1_local)
            print(f"Label {lbl} -> Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1_local:.3f} (TP={tp}, FP={fp}, FN={fn})")
        print(f"Macro-F1: {sum(per_class_f1) / len(per_class_f1):.4f}")

        # Feature importances (approx average across OneVsRest models if possible)
        print("\n===== FEATURE IMPORTANCES (approx avg across OVR GBT models) =====")
        try:
            ovr_model = model.stages[-1]
            if hasattr(ovr_model, "models"):
                importances_sum = None
                for gm in ovr_model.models:
                    arr = gm.featureImportances.toArray()
                    if importances_sum is None:
                        importances_sum = arr
                    else:
                        importances_sum += arr
                avg_importances = importances_sum / len(ovr_model.models)
                for feat, imp in zip(feature_cols, avg_importances):
                    print(f"{feat}: {imp:.6f}")
            else:
                print("No per-class models available to extract importances.")
        except Exception as e:
            print("Could not compute feature importances:", e)

        # ----------------------------
        # Save predictions as single CSV (no part files left behind)
        # ----------------------------
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)

        predictions = predictions.withColumn(
            "predicted_category",
            when(col("prediction") == 0, "Low")
            .when(col("prediction") == 1, "Medium")
            .otherwise("High")
        )

        # choose final columns; if prescriber_id exists include it
        final_cols = []
        if "prescriber_id" in predictions.columns:
            final_cols.append("prescriber_id")
        for c in ["first_name", "last_name", "provider_type", "state",
                  "total_claims", "total_payment_amount", "payment_to_drug_cost_ratio",
                  "fraud_risk_category"]:
            if c in predictions.columns:
                final_cols.append(c)
        final_cols += ["predicted_category"]

        sorted_preds = predictions
        if "prescriber_id" in predictions.columns:
            sorted_preds = predictions.orderBy(col("prescriber_id").asc())

        sorted_preds.select(*final_cols).coalesce(1).write.mode("overwrite").option("header", True).csv(TEMP_DIR)

        part_files = glob.glob(os.path.join(TEMP_DIR, "part-*.csv"))
        if not part_files:
            print("Warning: no part file found in temp write directory. Prediction save skipped.")
        else:
            part_file = part_files[0]
            try:
                if os.path.exists(FINAL_OUTPUT_CSV):
                    os.remove(FINAL_OUTPUT_CSV)
                shutil.move(part_file, FINAL_OUTPUT_CSV)
                print(f"\nSaved predictions as single CSV: {FINAL_OUTPUT_CSV}")
            except Exception as e:
                print("Could not move part file to final destination:", e)
            finally:
                # cleanup temp dir if exists
                if os.path.exists(TEMP_DIR):
                    shutil.rmtree(TEMP_DIR)

        # ----------------------------
        # Confusion matrix image
        # ----------------------------
        if HAS_PLOTTING:
            try:
                cm_pd = cm.toPandas()
                labels = [0, 1, 2]
                matrix = pd.DataFrame(0, index=labels, columns=labels)
                for _, r in cm_pd.iterrows():
                    matrix.at[int(r["fraud_label"]), int(r["prediction"])] = int(r["count"])

                plt.figure(figsize=(6, 5))
                im = plt.imshow(matrix.values, interpolation='nearest', cmap='Blues')
                plt.title("Confusion Matrix (GBT - no leakage)")
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

                # ensure directory exists
                out_dir = os.path.dirname(CONFUSION_PNG)
                if out_dir and not os.path.exists(out_dir):
                    os.makedirs(out_dir, exist_ok=True)

                plt.savefig(CONFUSION_PNG, dpi=200)
                plt.close()
                print(f"Confusion matrix image saved to: {CONFUSION_PNG}")
            except Exception as e:
                print("Could not produce confusion plot:", e)
        else:
            print("Plotting libs not available; skipping confusion image.")

        # ----------------------------
        # Print sample 10 rows vertically
        # ----------------------------
        print("\n===== SAMPLE PREDICTIONS (first 10 rows, vertical) =====")
        sample = sorted_preds.select(*[c for c in final_cols if c in sorted_preds.columns]).limit(10).collect()
        for i, row in enumerate(sample, start=1):
            print(f"\n--- Row {i} ---")
            rdict = row.asDict()
            for c in [c for c in final_cols if c in sorted_preds.columns]:
                print(f"{c}: {rdict.get(c)}")

    except Exception:
        print("\nERROR during execution:")
        traceback.print_exc()
    finally:
        spark.stop()
        print("\nSpark stopped. Done.")

if __name__ == "__main__":
    main()

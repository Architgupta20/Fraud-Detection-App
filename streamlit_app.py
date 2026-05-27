# streamlit_app.py
import os
import streamlit as st
import pandas as pd
from typing import Dict

# ---------- CONFIG (works for local + Docker/Render) ----------
# On Render the repo is copied into /app inside the container.
BASE_DIR = os.getenv("BASE_DIR", "/app")

# If you later add a Spark model under Models/, you can set MODEL_PATH via env var
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "Models", "spark_pipeline_model"))

# Your small model-data files are uploaded to the repo root on GitHub, so use BASE_DIR
MODEL_DATA_DIR = os.getenv("MODEL_DATA_DIR", BASE_DIR)

# ---------- FALLBACKS (the files you uploaded to repo root) ----------
PREDICTIONS_FILES = [
    "fraud_detection_gbt_combined_predictions.csv",
    "fraud_detection_gbt_predictions.csv",
    "fraud_detection_gbt_safe_predictions.csv",
    "fraud_detection_rf_predictions.csv",
]
FALLBACK_PREDICTIONS = next(
    (os.path.join(MODEL_DATA_DIR, p)
     for p in PREDICTIONS_FILES
     if os.path.exists(os.path.join(MODEL_DATA_DIR, p))),
    None
)

CONFUSION_FILES = [
    "confusion_matrix_gbt_combined.png",
    "confusion_matrix_gbt_safe.png",
    "confusion_matrix_gbt.png",
    "confusion_matrix_rf.png",
]
FOUND_CONFUSION_IMG = next(
    (os.path.join(MODEL_DATA_DIR, p)
     for p in CONFUSION_FILES
     if os.path.exists(os.path.join(MODEL_DATA_DIR, p))),
    None
)

FEATURE_COLS = [
    "total_claims", "total_drug_cost", "opioid_claims", "opioid_cost",
    "antibiotic_claims", "payment_to_drug_cost_ratio", "peer_deviation_score",
    "avg_risk_score", "payment_variability", "adjusted_risk_payment",
    "high_payment_flag", "high_opioid_flag", "elderly_focus_flag"
]

# ---------- OPTIONAL SPARK IMPORT ----------
try:
    from pyspark.sql import SparkSession
    from pyspark.ml.pipeline import PipelineModel
    from pyspark.sql.types import StructType, StructField, DoubleType, StringType
    SPARK_AVAILABLE = True
except Exception:
    SPARK_AVAILABLE = False

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="AI-Based Healthcare Claim Fraud Detection", layout="wide")
st.title("AI-Based Healthcare Claim Fraud Detection")

# Sidebar settings
st.sidebar.header("Settings")
# If Spark is available in the environment and a model exists at MODEL_PATH, user can enable it.
# Default behavior: prefer CSV fallback (safe for Render free tier) if a fallback CSV exists.
use_spark = st.sidebar.checkbox("Load Spark model (heavy)", value=False if not SPARK_AVAILABLE else False)
use_csv_fallback = st.sidebar.checkbox("Use CSV predictions fallback (fast)", value=True if FALLBACK_PREDICTIONS else False)

# ---------- TRY LOADING SPARK MODEL (only when requested) ----------
pipeline_model, spark, predictions_df = None, None, None

if use_spark and SPARK_AVAILABLE and os.path.exists(MODEL_PATH):
    try:
        with st.spinner("Loading Spark pipeline..."):
            spark = SparkSession.builder.master("local[*]") \
                .appName("FraudStreamlitApp") \
                .config("spark.ui.enabled", "false") \
                .getOrCreate()
            pipeline_model = PipelineModel.load(MODEL_PATH)
            st.sidebar.success("Spark pipeline loaded.")
    except Exception as e:
        pipeline_model = None
        use_spark = False
        st.sidebar.error(f"Failed to load Spark model: {e}")

# ---------- LOAD CSV FALLBACK (if available) ----------
if use_csv_fallback and FALLBACK_PREDICTIONS:
    try:
        predictions_df = pd.read_csv(FALLBACK_PREDICTIONS)
        st.sidebar.success(f"Loaded fallback predictions: {os.path.basename(FALLBACK_PREDICTIONS)}")
    except Exception as e:
        predictions_df = None
        use_csv_fallback = False
        st.sidebar.error(f"Failed to load fallback CSV: {e}")

# ---------- HELPER FUNCTIONS ----------
def map_label_to_category(label_val):
    try:
        val = float(label_val)
    except Exception:
        return str(label_val)
    return {0.0: "Low", 1.0: "Medium", 2.0: "High"}.get(val, str(val))

def softmax(arr):
    import math
    exps = [math.exp(float(a)) for a in arr]
    s = sum(exps)
    return [e / s for e in exps] if s != 0 else [0.0 for _ in exps]

# ---------- PREDICTION FUNCTIONS ----------
def predict_with_pipeline_single(row_dict: Dict):
    if pipeline_model is None or spark is None:
        raise RuntimeError("Spark pipeline not loaded.")
    pdf = pd.DataFrame([row_dict])
    for c in FEATURE_COLS + ["prescriber_id", "first_name", "last_name", "provider_type", "state"]:
        if c not in pdf.columns:
            pdf[c] = None

    # If pyspark types are available, create an explicit schema to avoid inference issues
    try:
        schema = StructType([
            StructField("prescriber_id", StringType(), True),
            StructField("first_name", StringType(), True),
            StructField("last_name", StringType(), True),
            StructField("provider_type", StringType(), True),
            StructField("state", StringType(), True),
        ] + [
            StructField(c, DoubleType(), True) for c in FEATURE_COLS
        ])
        sdf = spark.createDataFrame(pdf, schema=schema)
    except Exception:
        # fallback: let Spark infer schema
        sdf = spark.createDataFrame(pdf)

    out = pipeline_model.transform(sdf)
    select_cols = [c for c in ("prescriber_id", "prediction", "probability", "rawPrediction", "fraud_risk_category") if c in out.columns]
    row = out.select(*select_cols).collect()[0].asDict()

    prob_vec = None
    if "probability" in row and row.get("probability") is not None:
        try:
            prob_vec = list(row["probability"])
        except Exception:
            prob_vec = str(row["probability"])
    elif "rawPrediction" in row and row.get("rawPrediction") is not None:
        try:
            raw = list(row["rawPrediction"])
            prob_vec = softmax(raw)
        except Exception:
            prob_vec = None

    return {
        "prescriber_id": row.get("prescriber_id"),
        "prediction": row.get("prediction"),
        "predicted_category": map_label_to_category(row.get("prediction")),
        "probability": prob_vec,
        "original_label": row.get("fraud_risk_category")
    }

def predict_with_csv_model(pdf: pd.DataFrame):
    if predictions_df is None:
        raise RuntimeError("No fallback predictions CSV loaded.")
    if "prescriber_id" not in pdf.columns:
        raise ValueError("Uploaded CSV must contain 'prescriber_id'.")
    merged = pdf.merge(predictions_df, on="prescriber_id", how="left")
    if "predicted_category" not in merged.columns and "prediction" in merged.columns:
        merged["predicted_category"] = merged["prediction"].apply(map_label_to_category)
    return merged

# ---------- UI LAYOUT ----------
tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction (CSV Upload)", "Explore Model Outputs"])

# --- TAB 1: SINGLE PREDICTION ---
with tab1:
    st.header("Single Prescriber Prediction")
    left_col, right_col = st.columns([1, 1])

    with left_col:
        prescriber_id = st.text_input("Prescriber ID", "")
        first_name = st.text_input("First Name", "")
        last_name = st.text_input("Last Name", "")
        provider_type = st.text_input("Provider Type", "")
        state = st.text_input("State", "")

    with right_col:
        numeric_inputs = {f: st.number_input(f, value=0.0, step=0.1, format="%.6f") for f in FEATURE_COLS}

    st.markdown("---")
    _, center, _ = st.columns([1, 1, 1])
    with center:
        if st.button("Predict"):
            row = {
                "prescriber_id": prescriber_id or None,
                "first_name": first_name or None,
                "last_name": last_name or None,
                "provider_type": provider_type or None,
                "state": state or None
            }
            row.update(numeric_inputs)
            try:
                if use_spark and pipeline_model is not None:
                    pred = predict_with_pipeline_single(row)
                    st.success(f"Predicted Category: {pred['predicted_category']}")
                    st.write("Numeric Label:", pred["prediction"])
                    st.write("Probabilities (Low, Medium, High):")
                    st.write(pred["probability"] if pred["probability"] else "Unavailable")
                elif predictions_df is not None:
                    if prescriber_id:
                        matched = predictions_df[predictions_df["prescriber_id"].astype(str) == str(prescriber_id)]
                        if not matched.empty:
                            st.dataframe(matched.T)
                        else:
                            st.warning("No match found in precomputed predictions CSV.")
                    else:
                        st.warning("Provide prescriber_id for lookup in fallback CSV.")
                else:
                    st.warning("No model or fallback CSV available.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    st.markdown("### Notes")
    st.markdown(
        "- Enter prescriber details and numeric features, then click **Predict**.\n"
        "- If Spark is not loaded, use CSV fallback under **Batch Prediction (CSV Upload)** tab.\n"
        "- Categories: Low = 0, Medium = 1, High = 2.\n"
        "- Probabilities are shown as `[p_low, p_medium, p_high]`."
    )

# --- TAB 2: BATCH PREDICTION ---
with tab2:
    st.header("Batch Prediction (CSV Upload)")
    st.write("Upload a CSV with prescriber_id and feature columns.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.write(f"Uploaded {len(uploaded_df)} rows.")
            if use_spark and pipeline_model:
                sdf = spark.createDataFrame(uploaded_df)
                preds = pipeline_model.transform(sdf)
                out_cols = [c for c in ("prescriber_id", "prediction", "probability", "fraud_risk_category") if c in preds.columns]
                pdf_out = preds.select(*out_cols).toPandas()
                if "prediction" in pdf_out.columns:
                    pdf_out["predicted_category"] = pdf_out["prediction"].apply(map_label_to_category)
                st.dataframe(pdf_out.head(200))
                st.download_button("Download Predictions", pdf_out.to_csv(index=False).encode("utf-8"), "predictions.csv")
            elif predictions_df is not None:
                merged = predict_with_csv_model(uploaded_df)
                st.dataframe(merged.head(200))
                st.download_button("Download Results", merged.to_csv(index=False).encode("utf-8"), "predictions_joined.csv")
            else:
                st.warning("No model or fallback CSV available for batch prediction.")
        except Exception as e:
            st.error(f"Upload failed: {e}")

# --- TAB 3: EXPLORE OUTPUTS ---
with tab3:
    st.header("Explore Model Outputs")
    if predictions_df is not None:
        st.dataframe(predictions_df.head(50))
    else:
        st.info("No fallback predictions CSV found in the repository root.")

    if FOUND_CONFUSION_IMG:
        try:
            st.image(FOUND_CONFUSION_IMG, caption="Confusion Matrix")
        except Exception as e:
            st.warning(f"Could not load confusion matrix image: {e}")

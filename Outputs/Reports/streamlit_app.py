# # streamlit_app.py
# import os
# import streamlit as st
# import pandas as pd
# from typing import Dict

# # Try importing Spark (optional)
# try:
#     from pyspark.sql import SparkSession
#     from pyspark.ml.pipeline import PipelineModel
#     SPARK_AVAILABLE = True
# except Exception:
#     SPARK_AVAILABLE = False

# # ---------- CONFIG ----------
# BASE_DIR = os.path.expanduser("/Users/architgupta280/Desktop/LTI/Project")
# MODEL_PATH = os.path.join(BASE_DIR, "Models", "spark_pipeline_model")
# MODEL_DATA_DIR = os.path.join(BASE_DIR, "Data", "Model_Data")

# # Fallback CSV (choose first available)
# PREDICTIONS_FILES = [
#     "fraud_detection_gbt_predictions.csv",
#     "fraud_detection_gbt_safe_predictions.csv",
#     "fraud_detection_rf_predictions.csv",
# ]
# FALLBACK_PREDICTIONS = next(
#     (os.path.join(MODEL_DATA_DIR, p) for p in PREDICTIONS_FILES if os.path.exists(os.path.join(MODEL_DATA_DIR, p))),
#     None
# )

# # Confusion image (choose first available)
# CONFUSION_FILES = [
#     "confusion_matrix_gbt_safe.png",
#     "confusion_matrix_gbt.png",
#     "confusion_matrix_rf.png",
# ]
# FOUND_CONFUSION_IMG = next(
#     (os.path.join(MODEL_DATA_DIR, p) for p in CONFUSION_FILES if os.path.exists(os.path.join(MODEL_DATA_DIR, p))),
#     None
# )

# FEATURE_COLS = [
#     "total_claims", "total_drug_cost", "opioid_claims", "opioid_cost",
#     "antibiotic_claims", "payment_to_drug_cost_ratio", "peer_deviation_score",
#     "avg_risk_score", "payment_variability", "adjusted_risk_payment",
#     "high_payment_flag", "high_opioid_flag", "elderly_focus_flag"
# ]

# # ---------- STREAMLIT UI ----------
# st.set_page_config(page_title="AI-Based Healthcare Claim Fraud Detection", layout="wide")
# st.title("AI-Based Healthcare Claim Fraud Detection")

# # Sidebar (minimal — no status text)
# st.sidebar.header("Settings")
# use_spark = st.sidebar.checkbox("Load Spark model (heavy)", value=False if not SPARK_AVAILABLE else True)
# use_csv_fallback = st.sidebar.checkbox("Use CSV predictions fallback (fast)", value=True if FALLBACK_PREDICTIONS else False)

# # Try loading Spark pipeline
# pipeline_model, spark, predictions_df = None, None, None

# if use_spark and SPARK_AVAILABLE and os.path.exists(MODEL_PATH):
#     try:
#         with st.spinner("Loading Spark pipeline..."):
#             spark = SparkSession.builder.appName("FraudStreamlitApp").config("spark.ui.enabled", "false").getOrCreate()
#             pipeline_model = PipelineModel.load(MODEL_PATH)
#     except Exception:
#         pipeline_model = None
#         use_spark = False

# # Load fallback CSV if available
# if use_csv_fallback and FALLBACK_PREDICTIONS:
#     try:
#         predictions_df = pd.read_csv(FALLBACK_PREDICTIONS)
#     except Exception:
#         predictions_df = None
#         use_csv_fallback = False

# # Helper functions
# def map_label_to_category(label_val):
#     try:
#         val = float(label_val)
#     except Exception:
#         return str(label_val)
#     return {0.0: "Low", 1.0: "Medium", 2.0: "High"}.get(val, str(val))

# def softmax(arr):
#     import math
#     exps = [math.exp(float(a)) for a in arr]
#     s = sum(exps)
#     return [e / s for e in exps] if s != 0 else [0.0 for _ in exps]

# def predict_with_pipeline_single(row_dict: Dict):
#     if pipeline_model is None or spark is None:
#         raise RuntimeError("Spark pipeline not loaded.")
#     pdf = pd.DataFrame([row_dict])
#     for c in FEATURE_COLS + ["prescriber_id", "first_name", "last_name", "provider_type", "state"]:
#         if c not in pdf.columns:
#             pdf[c] = None
#     sdf = spark.createDataFrame(pdf)
#     out = pipeline_model.transform(sdf)
#     select_cols = [c for c in ("prescriber_id", "prediction", "probability", "rawPrediction", "fraud_risk_category") if c in out.columns]
#     row = out.select(*select_cols).collect()[0].asDict()
#     prob_vec = None
#     if "probability" in row and row.get("probability") is not None:
#         try:
#             prob_vec = list(row["probability"])
#         except Exception:
#             prob_vec = str(row["probability"])
#     elif "rawPrediction" in row and row.get("rawPrediction") is not None:
#         try:
#             raw = list(row["rawPrediction"])
#             prob_vec = softmax(raw)
#         except Exception:
#             prob_vec = None
#     return {
#         "prescriber_id": row.get("prescriber_id"),
#         "prediction": row.get("prediction"),
#         "predicted_category": map_label_to_category(row.get("prediction")),
#         "probability": prob_vec,
#         "original_label": row.get("fraud_risk_category")
#     }

# def predict_with_csv_model(pdf: pd.DataFrame):
#     if predictions_df is None:
#         raise RuntimeError("No fallback predictions CSV loaded.")
#     if "prescriber_id" not in pdf.columns:
#         raise ValueError("Uploaded CSV must contain 'prescriber_id'.")
#     merged = pdf.merge(predictions_df, on="prescriber_id", how="left")
#     if "predicted_category" not in merged.columns and "prediction" in merged.columns:
#         merged["predicted_category"] = merged["prediction"].apply(map_label_to_category)
#     return merged

# # ---------- UI Tabs ----------
# tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction (CSV Upload)", "Explore Model Outputs"])

# # --- TAB 1: SINGLE PREDICTION ---
# with tab1:
#     st.header("Single Prescriber Prediction")

#     left_col, right_col = st.columns([1, 1])
#     with left_col:
#         prescriber_id = st.text_input("Prescriber ID", "")
#         first_name = st.text_input("First Name", "")
#         last_name = st.text_input("Last Name", "")
#         provider_type = st.text_input("Provider Type", "")
#         state = st.text_input("State", "")
#     with right_col:
#         numeric_inputs = {f: st.number_input(f, value=0.0, step=0.1, format="%.6f") for f in FEATURE_COLS}

#     st.markdown("---")
#     _, center, _ = st.columns([1, 1, 1])
#     with center:
#         if st.button("Predict"):
#             row = {
#                 "prescriber_id": prescriber_id or None,
#                 "first_name": first_name or None,
#                 "last_name": last_name or None,
#                 "provider_type": provider_type or None,
#                 "state": state or None
#             }
#             row.update(numeric_inputs)
#             try:
#                 if use_spark and pipeline_model is not None:
#                     pred = predict_with_pipeline_single(row)
#                     st.success(f"Predicted Category: **{pred['predicted_category']}**")
#                     st.write("Numeric Label:", pred["prediction"])
#                     st.write("Probabilities (Low, Medium, High):")
#                     st.write(pred["probability"] if pred["probability"] else "Unavailable")
#                 elif predictions_df is not None:
#                     if prescriber_id:
#                         matched = predictions_df[predictions_df["prescriber_id"].astype(str) == str(prescriber_id)]
#                         if not matched.empty:
#                             st.dataframe(matched.T)
#                         else:
#                             st.warning("No match found in precomputed predictions CSV.")
#                     else:
#                         st.warning("Provide prescriber_id for lookup in fallback CSV.")
#                 else:
#                     st.warning("No model or fallback CSV available.")
#             except Exception as e:
#                 st.error(f"Prediction failed: {e}")

#     st.markdown("### Notes")
#     st.markdown(
#         "- Enter prescriber details and numeric features, then click **Predict**.\n"
#         "- If Spark is not loaded, use CSV fallback under Batch Prediction tab.\n"
#         "- Categories: Low = 0, Medium = 1, High = 2.\n"
#         "- Probabilities are shown as [p_low, p_medium, p_high]."
#     )

# # --- TAB 2: BATCH PREDICTION ---
# with tab2:
#     st.header("Batch Prediction (CSV Upload)")
#     st.write("Upload a CSV with prescriber_id and feature columns.")
#     uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
#     if uploaded_file:
#         try:
#             uploaded_df = pd.read_csv(uploaded_file)
#             st.write(f"Uploaded {len(uploaded_df)} rows.")
#             if use_spark and pipeline_model:
#                 sdf = spark.createDataFrame(uploaded_df)
#                 preds = pipeline_model.transform(sdf)
#                 out_cols = [c for c in ("prescriber_id", "prediction", "probability", "fraud_risk_category") if c in preds.columns]
#                 pdf_out = preds.select(*out_cols).toPandas()
#                 if "prediction" in pdf_out.columns:
#                     pdf_out["predicted_category"] = pdf_out["prediction"].apply(map_label_to_category)
#                 st.dataframe(pdf_out.head(200))
#                 st.download_button("Download Predictions", pdf_out.to_csv(index=False).encode("utf-8"), "predictions.csv")
#             elif predictions_df is not None:
#                 merged = predict_with_csv_model(uploaded_df)
#                 st.dataframe(merged.head(200))
#                 st.download_button("Download Results", merged.to_csv(index=False).encode("utf-8"), "predictions_joined.csv")
#         except Exception as e:
#             st.error(f"Upload failed: {e}")

# # --- TAB 3: EXPLORE OUTPUTS ---
# with tab3:
#     st.header("Explore Model Outputs")
#     if predictions_df is not None:
#         st.dataframe(predictions_df.head(50))
#     if FOUND_CONFUSION_IMG:
#         st.image(FOUND_CONFUSION_IMG, caption="Confusion Matrix", use_container_width=True)

# streamlit_app.py
import os
import streamlit as st
import pandas as pd
from typing import Dict

# ---------- CONFIG (works for local + Docker/Render) ----------
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import BASE_DIR as _CONFIG_BASE_DIR
from config import MODEL_DATA_DIR as _CONFIG_MODEL_DATA_DIR
from config import SPARK_PIPELINE_MODEL_DIR

BASE_DIR = os.getenv("BASE_DIR", str(_CONFIG_BASE_DIR))
MODEL_PATH = os.getenv("MODEL_PATH", str(SPARK_PIPELINE_MODEL_DIR))
MODEL_DATA_DIR = os.getenv("MODEL_DATA_DIR", str(_CONFIG_MODEL_DATA_DIR))

# ---------- FALLBACKS ----------
PREDICTIONS_FILES = [
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

# Input guardrails for single-record prediction UI:
# (min, max, default, step)
FEATURE_RANGES = {
    "total_claims": (0.0, 1_000_000.0, 0.0, 1.0),
    "total_drug_cost": (0.0, 100_000_000.0, 0.0, 100.0),
    "opioid_claims": (0.0, 1_000_000.0, 0.0, 1.0),
    "opioid_cost": (0.0, 100_000_000.0, 0.0, 100.0),
    "antibiotic_claims": (0.0, 1_000_000.0, 0.0, 1.0),
    "payment_to_drug_cost_ratio": (0.0, 1000.0, 0.0, 0.01),
    "peer_deviation_score": (0.0, 1000.0, 0.0, 0.01),
    "avg_risk_score": (0.0, 10.0, 0.0, 0.01),
    "payment_variability": (0.0, 1000.0, 0.0, 0.01),
    "adjusted_risk_payment": (0.0, 1_000_000_000.0, 0.0, 100.0),
    "high_payment_flag": (0.0, 1.0, 0.0, 1.0),
    "high_opioid_flag": (0.0, 1.0, 0.0, 1.0),
    "elderly_focus_flag": (0.0, 1.0, 0.0, 1.0),
}

# ---------- SPARK IMPORT ----------
try:
    from pyspark.sql import SparkSession
    from pyspark.ml.pipeline import PipelineModel
    from pyspark.sql.types import StructType, StructField, DoubleType, StringType
    SPARK_AVAILABLE = True
except Exception:
    SPARK_AVAILABLE = False

# ---------- STREAMLIT CONFIG ----------
st.set_page_config(page_title="AI-Based Healthcare Claim Fraud Detection", layout="wide")
st.title("AI-Based Healthcare Claim Fraud Detection")

# Sidebar
st.sidebar.header("Settings")
use_spark = st.sidebar.checkbox("Load Spark model (heavy)", value=False if not SPARK_AVAILABLE else True)
use_csv_fallback = st.sidebar.checkbox("Use CSV predictions fallback (fast)", value=True if FALLBACK_PREDICTIONS else False)

# ---------- LOAD MODEL ----------
pipeline_model, spark, predictions_df = None, None, None

if use_spark and SPARK_AVAILABLE and os.path.exists(MODEL_PATH):
    try:
        with st.spinner("Loading Spark pipeline..."):
            spark = SparkSession.builder.master("local[*]") \
                .appName("FraudStreamlitApp") \
                .config("spark.ui.enabled", "false") \
                .getOrCreate()
            pipeline_model = PipelineModel.load(MODEL_PATH)
    except Exception as e:
        pipeline_model = None
        use_spark = False
        st.sidebar.error(f"Failed to load Spark model: {e}")

# ---------- LOAD CSV FALLBACK ----------
if use_csv_fallback and FALLBACK_PREDICTIONS:
    try:
        predictions_df = pd.read_csv(FALLBACK_PREDICTIONS)
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


def parse_numeric_inputs(raw_inputs: Dict[str, str]):
    parsed = {}
    errors = []
    for feat, raw_val in raw_inputs.items():
        min_v, max_v, default_v, _ = FEATURE_RANGES.get(feat, (0.0, 1_000_000.0, 0.0, 0.1))
        text = str(raw_val).strip()
        if text == "":
            value = float(default_v)
        else:
            try:
                value = float(text)
            except ValueError:
                errors.append(f"{feat}: enter a valid number.")
                continue
        if value < min_v or value > max_v:
            errors.append(f"{feat}: must be between {min_v:g} and {max_v:g}.")
            continue
        parsed[feat] = value
    return parsed, errors

# ---------- FIXED FUNCTION (no type inference error) ----------
def predict_with_pipeline_single(row_dict: Dict):
    if pipeline_model is None or spark is None:
        raise RuntimeError("Spark pipeline not loaded.")
    
    # Create Pandas DataFrame
    pdf = pd.DataFrame([row_dict])

    # Ensure all feature columns exist
    for c in FEATURE_COLS + ["prescriber_id", "first_name", "last_name", "provider_type", "state"]:
        if c not in pdf.columns:
            pdf[c] = None

    # Explicit schema for Spark
    schema = StructType([
        StructField("prescriber_id", StringType(), True),
        StructField("first_name", StringType(), True),
        StructField("last_name", StringType(), True),
        StructField("provider_type", StringType(), True),
        StructField("state", StringType(), True),
        StructField("total_claims", DoubleType(), True),
        StructField("total_drug_cost", DoubleType(), True),
        StructField("opioid_claims", DoubleType(), True),
        StructField("opioid_cost", DoubleType(), True),
        StructField("antibiotic_claims", DoubleType(), True),
        StructField("payment_to_drug_cost_ratio", DoubleType(), True),
        StructField("peer_deviation_score", DoubleType(), True),
        StructField("avg_risk_score", DoubleType(), True),
        StructField("payment_variability", DoubleType(), True),
        StructField("adjusted_risk_payment", DoubleType(), True),
        StructField("high_payment_flag", DoubleType(), True),
        StructField("high_opioid_flag", DoubleType(), True),
        StructField("elderly_focus_flag", DoubleType(), True)
    ])

    # Create Spark DataFrame
    sdf = spark.createDataFrame(pdf, schema=schema)

    # Predict
    out = pipeline_model.transform(sdf)
    select_cols = [c for c in ("prescriber_id", "prediction", "probability", "rawPrediction", "fraud_risk_category") if c in out.columns]
    row = out.select(*select_cols).collect()[0].asDict()

    # Handle probability
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

# ---------- UI ----------
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
        numeric_inputs_raw = {}
        st.caption("Enter values manually. Placeholder shows accepted range.")
        for f in FEATURE_COLS:
            min_v, max_v, _, _ = FEATURE_RANGES.get(f, (0.0, 1_000_000.0, 0.0, 0.1))
            numeric_inputs_raw[f] = st.text_input(
                f,
                value="",
                placeholder=f"{min_v:g} to {max_v:g}",
                help=f"Allowed range: {min_v:g} to {max_v:g}",
            )

        with st.expander("Accepted ranges (all features)"):
            range_df = pd.DataFrame(
                [
                    {"feature": feat, "min": vals[0], "max": vals[1], "default": vals[2]}
                    for feat, vals in FEATURE_RANGES.items()
                ]
            )
            st.dataframe(range_df, hide_index=True, use_container_width=True)

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
            numeric_inputs, input_errors = parse_numeric_inputs(numeric_inputs_raw)
            if input_errors:
                st.error("Please fix input values:\n- " + "\n- ".join(input_errors))
                st.stop()
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
        except Exception as e:
            st.error(f"Upload failed: {e}")

# --- TAB 3: EXPLORE OUTPUTS ---
with tab3:
    st.header("Explore Model Outputs")
    if predictions_df is not None:
        st.dataframe(predictions_df.head(50))
    if FOUND_CONFUSION_IMG:
        st.image(FOUND_CONFUSION_IMG, caption="Confusion Matrix")

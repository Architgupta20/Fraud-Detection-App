import os
import streamlit as st
import pandas as pd
import joblib
from typing import Dict, List, Optional, Tuple

# ---------- CONFIG (works for local + Docker/Render) ----------
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import BASE_DIR as _CONFIG_BASE_DIR
from config import GBT_SKLEARN_PKL
from config import MODEL_DATA_DIR as _CONFIG_MODEL_DATA_DIR
from config import SPARK_PIPELINE_MODEL_DIR
from config import XGB_CALIBRATED_PKL
from risk_rules import CLASS_NAMES, INV_LABEL_MAP, ML_FEATURE_COLS

BASE_DIR = os.getenv("BASE_DIR", str(_CONFIG_BASE_DIR))
MODEL_PATH = os.getenv("MODEL_PATH", str(SPARK_PIPELINE_MODEL_DIR))
MODEL_DATA_DIR = os.getenv("MODEL_DATA_DIR", str(_CONFIG_MODEL_DATA_DIR))
SKLEARN_MODEL_PATH = os.getenv("SKLEARN_MODEL_PATH", str(GBT_SKLEARN_PKL))
XGB_MODEL_PATH = os.getenv("XGB_MODEL_PATH", str(XGB_CALIBRATED_PKL))

PREDICTIONS_FILES = {
    "xgb": "fraud_detection_xgb_predictions.csv",
    "sklearn": "fraud_detection_gbt_sklearn_predictions.csv",
}


def _predictions_path(key: str) -> Optional[str]:
    name = PREDICTIONS_FILES.get(key)
    if not name:
        return None
    path = os.path.join(MODEL_DATA_DIR, name)
    return path if os.path.exists(path) else None

FOUND_CONFUSION_IMG = None  # optional; add PNG paths under Model_Data/ if you save plots from training

FEATURE_COLS = [
    "total_claims", "total_drug_cost", "opioid_claims", "opioid_cost",
    "antibiotic_claims", "payment_to_drug_cost_ratio", "peer_deviation_score",
    "avg_risk_score", "payment_variability", "adjusted_risk_payment",
    "high_payment_flag", "high_opioid_flag", "elderly_focus_flag"
]

# Input guardrails for single-record prediction UI:
# (min, max, default, step)
FEATURE_RANGES = {
    "total_claims": (0.0, 200_000.0, 0.0, 1.0),
    "total_drug_cost": (0.0, 20_000_000.0, 0.0, 100.0),
    "opioid_claims": (0.0, 50_000.0, 0.0, 1.0),
    "opioid_cost": (0.0, 10_000_000.0, 0.0, 100.0),
    "antibiotic_claims": (0.0, 50_000.0, 0.0, 1.0),
    "payment_to_drug_cost_ratio": (0.0, 20.0, 0.0, 0.01),
    "peer_deviation_score": (0.0, 20.0, 0.0, 0.01),
    "avg_risk_score": (0.0, 5.0, 0.0, 0.01),
    "payment_variability": (0.0, 50.0, 0.0, 0.01),
    "adjusted_risk_payment": (0.0, 50_000_000.0, 0.0, 100.0),
    "high_payment_flag": (0.0, 1.0, 0.0, 1.0),
    "high_opioid_flag": (0.0, 1.0, 0.0, 1.0),
    "elderly_focus_flag": (0.0, 1.0, 0.0, 1.0),
}

FEATURE_DESCRIPTIONS = {
    "total_claims": (
        "Real-world meaning: every time this doctor writes a prescription that Medicare Part D pays for, "
        "that counts as one claim. Example: 5,000 claims means they issued about 5,000 prescriptions "
        "in the reporting period."
    ),
    "total_drug_cost": (
        "Real-world meaning: the total dollar value of all medicines this doctor prescribed. "
        "Example: two doctors may write the same number of prescriptions, but one prescribes "
        "expensive brands and will have much higher drug cost."
    ),
    "opioid_claims": (
        "Real-world meaning: how many of this doctor's prescriptions were for opioid pain medicines "
        "(like oxycodone/hydrocodone classes). High values can mean heavy opioid prescribing."
    ),
    "opioid_cost": (
        "Real-world meaning: how much money (in dollars) this doctor's opioid prescriptions cost overall. "
        "A doctor can have moderate claim count but very high opioid cost if drugs are expensive."
    ),
    "antibiotic_claims": (
        "Real-world meaning: how many antibiotic prescriptions this doctor wrote. "
        "Very high antibiotic volume may indicate broad or repeated antibiotic use."
    ),
    "payment_to_drug_cost_ratio": (
        "Real-world meaning: money received from drug companies divided by total prescribing spend. "
        "If this is 1.5, they received $1.50 in industry payments for every $1 of drugs prescribed — "
        "often reviewed as a potential conflict-of-interest signal."
    ),
    "peer_deviation_score": (
        "Real-world meaning: compares this doctor's typical payment size to other doctors of the same type. "
        "1.0 means 'about average for peers'; 5.0 means they receive around 5x the typical peer payment level."
    ),
    "avg_risk_score": (
        "Real-world meaning: how sick this doctor's patient panel is on average (Medicare risk score). "
        "Higher score means patients are generally older/sicker, which should be considered before "
        "judging high cost or high prescribing."
    ),
    "payment_variability": (
        "Real-world meaning: whether payments are steady or spiky. "
        "High value means one or a few very large payments are much bigger than their usual payment size."
    ),
    "adjusted_risk_payment": (
        "Real-world meaning: total industry payments adjusted for patient sickness level. "
        "Helps compare doctors fairly when one mostly treats very high-risk patients."
    ),
    "high_payment_flag": (
        "Real-world meaning: quick yes/no flag. "
        "1 = this doctor's average single payment is above $1,000 (unusually large typical payment); "
        "0 = not above that threshold."
    ),
    "high_opioid_flag": (
        "Real-world meaning: quick yes/no flag. "
        "1 = more than 50% of this doctor's prescriptions are opioids; "
        "0 = opioids are not the majority of their prescribing."
    ),
    "elderly_focus_flag": (
        "Real-world meaning: quick yes/no flag. "
        "1 = average patient age is above 70 (practice mainly serves elderly patients); "
        "0 = not primarily elderly-focused."
    ),
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
APP_TITLE = "Prescriber Risk Prioritization"
DISCLAIMER = (
    "Labels are **rule-based review priority** (Low / High), not confirmed fraud. "
    "Use this tool to **prioritize human review**, not as legal proof of wrongdoing."
)

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.info(DISCLAIMER)

# Sidebar (minimal — model outputs live on Explore tab)
st.sidebar.header("Settings")
use_spark = st.sidebar.checkbox(
    "Load Spark model (heavy)",
    value=False,
    disabled=not SPARK_AVAILABLE,
)

xgb_available = os.path.exists(XGB_MODEL_PATH)
sklearn_available = os.path.exists(SKLEARN_MODEL_PATH)

# ---------- LOAD MODEL (XGB preferred for Single / Batch tabs) ----------
pipeline_model, spark, ml_bundle = None, None, None

if xgb_available:
    try:
        ml_bundle = joblib.load(XGB_MODEL_PATH)
    except Exception as e:
        st.sidebar.error(f"Failed to load XGBoost model: {e}")
elif sklearn_available:
    try:
        ml_bundle = joblib.load(SKLEARN_MODEL_PATH)
    except Exception as e:
        st.sidebar.error(f"Failed to load sklearn model: {e}")

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


@st.cache_data(show_spinner="Loading predictions CSV...")
def load_predictions_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "prescriber_id" in df.columns:
        df["prescriber_id"] = df["prescriber_id"].astype(str)
    return df


def render_explore_model_outputs(model_key: str, label: str) -> None:
    """Show precomputed predictions for one trained model."""
    path = _predictions_path(model_key)
    if not path:
        st.warning(f"No predictions file for {label}. Run the matching train script.")
        return
    try:
        df = load_predictions_csv(path)
    except Exception as e:
        st.error(f"Could not load {PREDICTIONS_FILES[model_key]}: {e}")
        return
    st.caption(f"{len(df):,} rows · `{PREDICTIONS_FILES[model_key]}`")
    lookup_npi = st.text_input(
        "Filter by prescriber_id (optional)",
        key=f"explore_npi_{model_key}",
    )
    if lookup_npi.strip():
        view_df = df[df["prescriber_id"] == lookup_npi.strip()]
        if view_df.empty:
            st.warning("No rows for that prescriber_id.")
        else:
            st.dataframe(view_df, use_container_width=True)
    else:
        st.dataframe(df.head(100), use_container_width=True)
        if "predicted_category" in df.columns:
            st.bar_chart(df["predicted_category"].value_counts())

# ---------- HELPER FUNCTIONS ----------
def map_label_to_category(label_val):
    try:
        val = int(float(label_val))
    except Exception:
        return str(label_val)
    return INV_LABEL_MAP.get(val, str(val))


def format_probabilities(probs, num_classes: Optional[int] = None) -> str:
    if not probs:
        return "Unavailable"
    vec = [float(p) for p in probs]
    n = num_classes or len(vec)
    if n == 2 and len(vec) >= 2:
        return ", ".join(f"{name}: {vec[i]:.3f}" for i, name in enumerate(CLASS_NAMES))
    return str([round(p, 4) for p in vec])

def softmax(arr):
    import math
    exps = [math.exp(float(a)) for a in arr]
    s = sum(exps)
    return [e / s for e in exps] if s != 0 else [0.0 for _ in exps]


def format_bound(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:g}"


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
            errors.append(
                f"{feat}: must be between {format_bound(min_v)} and {format_bound(max_v)}."
            )
            continue
        parsed[feat] = value
    return parsed, errors


def get_fired_rules(row: Dict) -> List[Tuple[str, str]]:
    from risk_rules import evaluate_rules_for_row

    use_pct = any(
        k in row
        for k in (
            "opioid_volume_pct_flag",
            "peer_outlier_pct_flag",
            "payment_spiky_pct_flag",
            "total_payments_pct_flag",
        )
    )
    _, _, display = evaluate_rules_for_row(row, use_percentile_flags=use_pct)
    if display:
        return display
    fired_raw = row.get("rules_fired") or ""
    if fired_raw:
        from risk_rules import RULE_LABELS, _rule_detail

        return [
            (RULE_LABELS.get(rid, rid), _rule_detail(rid, row))
            for rid in str(fired_raw).split("|")
            if rid
        ]
    return []


def get_top_model_features(row: Dict, bundle: Dict, top_n: int = 5) -> pd.DataFrame:
    feature_cols = bundle["feature_cols"]
    model = bundle["model"]
    importances = model.feature_importances_
    ranked = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)[:top_n]
    return pd.DataFrame(
        [
            {
                "feature": feat,
                "your_value": row.get(feat, 0),
                "model_importance": round(float(imp), 4),
            }
            for feat, imp in ranked
        ]
    )


def render_why_flagged(row: Dict, bundle: Optional[Dict] = None):
    st.subheader("Why this result?")
    fired = get_fired_rules(row)
    if fired:
        st.markdown("**Rule signals triggered**")
        for name, detail in fired:
            st.markdown(f"- **{name}:** {detail}")
    else:
        st.markdown("**Rule signals triggered:** none of the configured risk rules fired.")

    if bundle is not None:
        st.markdown("**Top model features (global importance)**")
        st.dataframe(get_top_model_features(row, bundle), hide_index=True, use_container_width=True)


def predict_with_ml_bundle_single(row_dict: Dict, bundle: Dict) -> Dict:
    feature_cols = bundle["feature_cols"]
    scaler = bundle["scaler"]
    model = bundle["model"]
    values = [[float(row_dict.get(c, 0) or 0) for c in feature_cols]]
    scaled = scaler.transform(values)
    pred = int(model.predict(scaled)[0])
    probs = model.predict_proba(scaled)[0]
    return {
        "prediction": pred,
        "predicted_category": map_label_to_category(pred),
        "probability": [float(p) for p in probs],
    }


def predict_with_ml_bundle_batch(pdf: pd.DataFrame, bundle: Dict) -> pd.DataFrame:
    feature_cols = bundle["feature_cols"]
    scaler = bundle["scaler"]
    model = bundle["model"]
    X = pdf.reindex(columns=feature_cols, fill_value=0.0).astype(float)
    scaled = scaler.transform(X.values)
    preds = model.predict(scaled)
    probs = model.predict_proba(scaled)
    out = pdf.copy()
    out["prediction"] = preds
    out["predicted_category"] = [map_label_to_category(p) for p in preds]
    if probs.shape[1] >= 2:
        out["p_low"] = probs[:, 0]
        out["p_high"] = probs[:, 1]
    return out


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
        model_feats = (
            ml_bundle.get("feature_cols", ML_FEATURE_COLS) if ml_bundle else ML_FEATURE_COLS
        )
        st.caption(
            "Model uses: "
            + ", ".join(model_feats)
            + ". Other fields help explain rule signals."
        )
        for f in FEATURE_COLS:
            min_v, max_v, _, _ = FEATURE_RANGES.get(f, (0.0, 1_000_000.0, 0.0, 0.1))
            desc = FEATURE_DESCRIPTIONS.get(f, "No description available.")
            numeric_inputs_raw[f] = st.text_input(
                f,
                value="",
                placeholder=f"{format_bound(min_v)} to {format_bound(max_v)}",
                help=desc,
            )

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
                if ml_bundle is not None:
                    pred = predict_with_ml_bundle_single(row, ml_bundle)
                    st.success(f"Predicted Category: {pred['predicted_category']}")
                    st.write("Numeric Label:", pred["prediction"])
                    nc = ml_bundle.get("num_classes", len(pred["probability"]))
                    st.write("Probabilities:", format_probabilities(pred["probability"], nc))
                    render_why_flagged(row, ml_bundle)
                elif use_spark and pipeline_model is not None:
                    pred = predict_with_pipeline_single(row)
                    st.success(f"Predicted Category: {pred['predicted_category']}")
                    st.write("Numeric Label:", pred["prediction"])
                    st.write("Probabilities:", format_probabilities(pred.get("probability")))
                    render_why_flagged(row)
                else:
                    st.warning(
                        "No live model loaded. Run Models/train_xgb.py or train_sklearn.py."
                    )
            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.markdown("### Notes")
st.markdown(
    "- Enter prescriber details and numeric features, then click **Predict**.\n"
    "- **Explore Model Outputs** uses sub-tabs for XGBoost vs sklearn prediction files.\n"
    "- Rule labels (v2.1): **Low** (0–1 risk points), **High** (≥ 2 points).\n"
    "- ML categories: Low = 0, High = 1 (`p_low`, `p_high` on batch export)."
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
            if ml_bundle is not None:
                pdf_out = predict_with_ml_bundle_batch(uploaded_df, ml_bundle)
                st.dataframe(pdf_out.head(200))
                st.download_button(
                    "Download Predictions",
                    pdf_out.to_csv(index=False).encode("utf-8"),
                    "predictions.csv",
                )
            elif use_spark and pipeline_model:
                sdf = spark.createDataFrame(uploaded_df)
                preds = pipeline_model.transform(sdf)
                out_cols = [c for c in ("prescriber_id", "prediction", "probability", "fraud_risk_category") if c in preds.columns]
                pdf_out = preds.select(*out_cols).toPandas()
                if "prediction" in pdf_out.columns:
                    pdf_out["predicted_category"] = pdf_out["prediction"].apply(map_label_to_category)
                st.dataframe(pdf_out.head(200))
                st.download_button("Download Predictions", pdf_out.to_csv(index=False).encode("utf-8"), "predictions.csv")
            else:
                st.warning("No live model loaded. Run training first.")
        except Exception as e:
            st.error(f"Upload failed: {e}")

# --- TAB 3: EXPLORE OUTPUTS ---
with tab3:
    st.header("Explore Model Outputs")
    explore_models: List[Tuple[str, str]] = []
    if _predictions_path("xgb"):
        explore_models.append(("xgb", "XGBoost"))
    if _predictions_path("sklearn"):
        explore_models.append(("sklearn", "sklearn GBT"))

    if not explore_models:
        st.warning(
            "No predictions CSV found under `Data/Model_Data/`. "
            "Run `Models/train_xgb.py` and/or `Models/train_sklearn.py`, then refresh."
        )
    elif len(explore_models) == 1:
        key, label = explore_models[0]
        render_explore_model_outputs(key, label)
    else:
        mini_tabs = st.tabs([label for _, label in explore_models])
        for (key, label), mini_tab in zip(explore_models, mini_tabs):
            with mini_tab:
                render_explore_model_outputs(key, label)

    if FOUND_CONFUSION_IMG:
        st.image(FOUND_CONFUSION_IMG, caption="Confusion Matrix")

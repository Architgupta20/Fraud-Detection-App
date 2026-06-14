import os
import sqlite3
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
from config import API_BASE_URL as _CONFIG_API_BASE_URL
from config import APP_API_KEY as _CONFIG_APP_API_KEY
from config import APP_PASSWORD as _CONFIG_APP_PASSWORD
from config import FRAUD_RISK_SCORED_CSV
from config import GBT_SKLEARN_PKL
from config import MODEL_DATA_DIR as _CONFIG_MODEL_DATA_DIR
from config import NPI_LOOKUP_CSV
from config import NPI_LOOKUP_SQLITE
from config import NPI_LOOKUP_SQLITE_GZ
from config import RISK_RULES_VERSION
from config import SPARK_PIPELINE_MODEL_DIR
from config import XGB_CALIBRATED_PKL
from risk_rules import CLASS_NAMES, INV_LABEL_MAP, ML_FEATURE_COLS

BASE_DIR = os.getenv("BASE_DIR", str(_CONFIG_BASE_DIR))
API_BASE_URL = os.getenv("API_BASE_URL", _CONFIG_API_BASE_URL).rstrip("/")
APP_PASSWORD = os.getenv("APP_PASSWORD", _CONFIG_APP_PASSWORD)
APP_API_KEY = os.getenv("APP_API_KEY", _CONFIG_APP_API_KEY)
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


# Columns for chunked NPI search (subset of scored file)
NPI_LOOKUP_COLUMNS = [
    "prescriber_id", "first_name", "last_name", "state", "city", "provider_type",
    "fraud_risk_category", "risk_points", "rules_fired", "rules_version",
    "total_claims", "total_drug_cost", "opioid_claims", "payment_to_drug_cost_ratio",
    "peer_deviation_score", "avg_risk_score", "payment_variability", "adjusted_risk_payment",
    "high_payment_flag", "high_opioid_flag", "elderly_focus_flag",
    "antibiotic_claim_ratio", "antibiotic_claims", "total_payment_amount",
    "opioid_volume_pct_flag", "peer_outlier_pct_flag", "payment_spiky_pct_flag",
    "total_payments_pct_flag",
]

US_STATE_OPTIONS = [
    "All",
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN",
    "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH",
    "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT",
    "VT", "VA", "WA", "WV", "WI", "WY",
]


def resolve_npi_lookup_path() -> Optional[str]:
    """Prefer SQLite index (deploy); fall back to CSV / full scored file."""
    if os.path.exists(str(NPI_LOOKUP_SQLITE)):
        return str(NPI_LOOKUP_SQLITE)
    if os.path.exists(str(NPI_LOOKUP_SQLITE_GZ)):
        return str(NPI_LOOKUP_SQLITE_GZ)
    slim = str(NPI_LOOKUP_CSV)
    if os.path.exists(slim):
        return slim
    scored = str(FRAUD_RISK_SCORED_CSV)
    if os.path.exists(scored):
        return scored
    return None


@st.cache_resource(show_spinner="Preparing NPI lookup index (first load)...")
def get_npi_sqlite_connection(db_path: str) -> sqlite3.Connection:
    """Decompress .sqlite.gz once per container, then open indexed DB."""
    import gzip
    import shutil

    path = db_path
    if path.endswith(".gz"):
        out = path[: -len(".gz")]
        if not os.path.exists(out):
            with gzip.open(path, "rb") as f_in, open(out, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        path = out
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def lookup_npi_in_risk_file(csv_path: str, npi: str) -> Optional[Dict]:
    """NPI search — SQLite (fast) or chunked CSV (local fallback)."""
    target = str(npi).strip()
    if not target:
        return None

    if csv_path.endswith(".sqlite") or csv_path.endswith(".sqlite.gz"):
        conn = get_npi_sqlite_connection(csv_path)
        cur = conn.execute(
            "SELECT * FROM prescribers WHERE prescriber_id = ? LIMIT 1",
            (target,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    usecols = None
    if csv_path.endswith("npi_risk_lookup.csv"):
        usecols = NPI_LOOKUP_COLUMNS
    else:
        usecols = lambda c: c in NPI_LOOKUP_COLUMNS or c == "prescriber_id"
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=250_000):
        if "prescriber_id" not in chunk.columns:
            break
        chunk["prescriber_id"] = chunk["prescriber_id"].astype(str)
        hit = chunk[chunk["prescriber_id"] == target]
        if not hit.empty:
            return hit.iloc[0].to_dict()
    return None


def lookup_ml_prediction_for_npi(npi: str) -> Optional[Dict]:
    path = _predictions_path("sklearn")
    if not path:
        return None
    df = lookup_prescriber_in_csv(path, npi)
    if df.empty:
        return None
    return df.iloc[0].to_dict()


@st.cache_resource(show_spinner=False, ttl=60)
def _api_ready() -> bool:
    import api_client

    api_client.API_BASE_URL = API_BASE_URL
    return api_client.api_is_ready()


def lookup_prescriber_via_api(npi: str) -> Optional[Dict]:
    import api_client

    api_client.API_BASE_URL = API_BASE_URL
    return api_client.fetch_prescriber(npi)


def render_prescriber_browse() -> None:
    import api_client

    api_client.API_BASE_URL = API_BASE_URL
    st.subheader("Browse prescribers")
    st.caption("Filter rule-based review priority from Postgres (via API).")
    c1, c2, c3 = st.columns(3)
    with c1:
        risk_filter = st.selectbox("Review priority", ["All", "Low", "High"], key="browse_risk")
    with c2:
        state_filter = st.selectbox("State", US_STATE_OPTIONS, key="browse_state")
    with c3:
        limit = st.number_input("Max rows", min_value=10, max_value=500, value=50, step=10, key="browse_limit")
    if st.button("Search prescribers", key="browse_btn"):
        with st.spinner("Querying API..."):
            try:
                data = api_client.fetch_prescribers(
                    risk=risk_filter,
                    state=state_filter if state_filter != "All" else None,
                    limit=int(limit),
                )
            except Exception as exc:
                st.error(f"API request failed: {exc}")
                return
        st.caption(f"Showing {len(data['items']):,} of {data['total']:,} matching prescribers.")
        if data["items"]:
            st.dataframe(pd.DataFrame(data["items"]), use_container_width=True, hide_index=True)
        else:
            st.info("No prescribers match those filters.")


def require_analyst_login() -> bool:
    """Step 7: password gate for analyst queue (skipped when APP_PASSWORD unset)."""
    if not APP_PASSWORD:
        return True
    if st.session_state.get("analyst_authenticated"):
        return True
    st.subheader("Analyst login")
    st.caption("Enter the app password to update review status and export the queue.")
    with st.form("analyst_login_form"):
        entered = st.text_input("Password", type="password")
        if st.form_submit_button("Log in", type="primary"):
            if entered == APP_PASSWORD:
                st.session_state["analyst_authenticated"] = True
                st.rerun()
            st.error("Incorrect password.")
    return False


def render_analyst_logout_button() -> None:
    if APP_PASSWORD and st.session_state.get("analyst_authenticated"):
        if st.sidebar.button("Log out (analyst)", key="analyst_logout"):
            st.session_state.pop("analyst_authenticated", None)
            st.rerun()


def render_risk_dashboard() -> None:
    import api_client

    api_client.API_BASE_URL = API_BASE_URL
    if not _api_ready():
        st.warning("Risk dashboard requires the Postgres API. Set `API_BASE_URL` and deploy the API service.")
        return

    try:
        summary = api_client.fetch_stats_summary()
        by_state = api_client.fetch_stats_by_state(limit=15)
        top_risk = api_client.fetch_top_risk(limit=10)
    except Exception as exc:
        st.error(f"Could not load stats: {exc}")
        return

    st.subheader("Population overview")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total prescribers", f"{summary['total_prescribers']:,}")
    cat_map = {row["category"]: row["count"] for row in summary["by_category"]}
    m2.metric("Low priority", f"{cat_map.get('Low', 0):,}")
    m3.metric("High priority", f"{cat_map.get('High', 0):,}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Review priority mix**")
        st.bar_chart(pd.Series({row["category"]: row["count"] for row in summary["by_category"]}))
    with c2:
        st.markdown("**Top states by prescriber count**")
        if by_state["items"]:
            st.bar_chart(
                pd.Series({row["state"]: row["count"] for row in by_state["items"]}),
                horizontal=True,
            )

    st.markdown("**Top risk prescribers**")
    if top_risk["items"]:
        st.dataframe(pd.DataFrame(top_risk["items"]), use_container_width=True, hide_index=True)

    try:
        oig = api_client.fetch_oig_overlap(sample_limit=10)
    except Exception:
        oig = None
    if oig and oig.get("oig_exclusion_count", 0) > 0:
        st.markdown("---")
        st.subheader("OIG exclusion screening")
        st.caption("Cross-check against HHS OIG LEIE (federal exclusion list, NPI matches only).")
        o1, o2 = st.columns(2)
        o1.metric("OIG exclusions loaded (NPI)", f"{oig['oig_exclusion_count']:,}")
        o2.metric("Matches in our prescriber panel", f"{oig['prescriber_overlap_count']:,}")
        if oig.get("sample"):
            st.markdown("**Sample overlapping NPIs**")
            st.dataframe(pd.DataFrame(oig["sample"]), use_container_width=True, hide_index=True)


def _format_oig_date(raw: Optional[str]) -> str:
    if not raw or str(raw).strip() in ("", "00000000", "nan"):
        return "—"
    text = str(raw).strip()
    if len(text) == 8 and text.isdigit():
        return f"{text[4:6]}/{text[6:8]}/{text[0:4]}"
    return text


def render_oig_check(npi: str) -> None:
    import api_client

    if not _api_ready():
        return
    api_client.API_BASE_URL = API_BASE_URL
    st.markdown("**OIG exclusion check (LEIE)**")
    try:
        result = api_client.fetch_oig_check(npi)
    except Exception as exc:
        st.warning(f"OIG check unavailable: {exc}")
        return
    if result.get("on_exclusion_list") and result.get("exclusion"):
        ex = result["exclusion"]
        st.error(
            "Federal OIG exclusion list match — this NPI appears on the LEIE. "
            "Treat as a compliance red flag (not automatic proof of current program eligibility)."
        )
        name = " ".join(
            p for p in (ex.get("first_name"), ex.get("last_name")) if p and str(p) != "nan"
        ).strip() or ex.get("business_name") or "—"
        st.markdown(
            f"- **Name / entity:** {name}\n"
            f"- **Exclusion type:** `{ex.get('exclusion_type', '—')}`\n"
            f"- **Exclusion date:** {_format_oig_date(ex.get('exclusion_date'))}\n"
            f"- **Reinstatement date:** {_format_oig_date(ex.get('reinstatement_date'))}"
        )
    else:
        st.success("No OIG LEIE match for this NPI (NPI-indexed subset).")


REVIEW_STATUS_OPTIONS = ["All", "pending", "reviewed", "needs_followup"]



def render_analyst_queue() -> None:
    import api_client

    api_client.API_BASE_URL = API_BASE_URL
    if not _api_ready():
        st.warning("Analyst queue requires the Postgres API.")
        return
    if not require_analyst_login():
        return

    st.subheader("Review queue")
    st.caption("Filter High-priority prescribers, update review status, and export CSV.")
    c1, c2, c3 = st.columns(3)
    with c1:
        status_filter = st.selectbox("Review status", REVIEW_STATUS_OPTIONS, key="queue_status")
    with c2:
        state_filter = st.selectbox("State", US_STATE_OPTIONS, key="queue_state")
    with c3:
        limit = st.number_input("Max rows", min_value=10, max_value=500, value=50, step=10, key="queue_limit")

    if st.button("Load queue", key="queue_load_btn"):
        with st.spinner("Loading queue..."):
            try:
                data = api_client.fetch_reviews(
                    status=status_filter,
                    state=state_filter if state_filter != "All" else None,
                    limit=int(limit),
                )
                st.session_state["queue_data"] = data
            except Exception as exc:
                st.error(f"Could not load queue: {exc}")

    data = st.session_state.get("queue_data")
    if data:
        st.caption(f"Showing {len(data['items']):,} of {data['total']:,} queue rows.")
        if data["items"]:
            st.dataframe(pd.DataFrame(data["items"]), use_container_width=True, hide_index=True)
        else:
            st.info("No rows match these filters.")

    st.markdown("---")
    st.markdown("**Update a prescriber review**")
    u1, u2 = st.columns([1, 2])
    with u1:
        update_npi = st.text_input("NPI", key="queue_update_npi", placeholder="e.g. 1003000142")
        update_status = st.selectbox(
            "Status",
            ["pending", "reviewed", "needs_followup"],
            format_func=lambda s: {"pending": "Pending", "reviewed": "Reviewed", "needs_followup": "Needs follow-up"}[s],
            key="queue_update_status",
        )
    with u2:
        update_note = st.text_area("Note (optional)", key="queue_update_note", height=100)
    if st.button("Save review", key="queue_save_btn"):
        if not update_npi.strip():
            st.warning("Enter an NPI to update.")
        else:
            try:
                api_client.upsert_review(
                    update_npi.strip(),
                    status=update_status,
                    note=update_note.strip() or None,
                    api_key=APP_API_KEY or None,
                )
                st.success(f"Saved review for NPI {update_npi.strip()}.")
                st.session_state.pop("queue_data", None)
            except Exception as exc:
                st.error(f"Save failed: {exc}")

    if st.button("Export queue CSV", key="queue_export_btn"):
        try:
            st.session_state["queue_export_csv"] = api_client.export_reviews_csv(
                status=status_filter,
                state=state_filter if state_filter != "All" else None,
                api_key=APP_API_KEY or None,
            )
        except Exception as exc:
            st.error(f"Export failed: {exc}")

    if st.session_state.get("queue_export_csv"):
        st.download_button(
            "Download CSV",
            st.session_state["queue_export_csv"].encode("utf-8"),
            file_name="review_queue_export.csv",
            mime="text/csv",
            key="queue_download_btn",
        )


def render_npi_lookup_result(row: Dict, ml_row: Optional[Dict] = None) -> None:
    name = " ".join(
        p for p in (row.get("first_name"), row.get("last_name")) if p and str(p) != "nan"
    ).strip() or "—"
    category = row.get("fraud_risk_category", "—")
    points = row.get("risk_points", "—")

    if category == "High":
        st.error(f"Review priority: **High** ({points} risk points)")
    else:
        st.success(f"Review priority: **Low** ({points} risk points)")

    c1, c2, c3 = st.columns(3)
    c1.metric("NPI", row.get("prescriber_id", "—"))
    c2.metric("State", row.get("state", "—"))
    c3.metric("Specialty", row.get("provider_type", "—"))
    rules_ver = row.get("rules_version") or RISK_RULES_VERSION
    city = row.get("city") or ""
    location = f" · {city}" if city and str(city) != "nan" else ""
    st.markdown(f"**{name}**{location} · rules v{rules_ver}")

    if ml_row is not None:
        st.markdown("**ML model (sklearn GBT)**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted", ml_row.get("predicted_category", "—"))
        if "p_low" in ml_row and "p_high" in ml_row:
            m2.metric("P(Low)", f"{float(ml_row['p_low']):.3f}")
            m3.metric("P(High)", f"{float(ml_row['p_high']):.3f}")

    st.markdown("**Key signals**")
    sig_cols = st.columns(4)
    sig_cols[0].metric("Claims", row.get("total_claims", "—"))
    sig_cols[1].metric("Pay/cost ratio", f"{float(row.get('payment_to_drug_cost_ratio') or 0):.3f}" if row.get("payment_to_drug_cost_ratio") is not None else "—")
    sig_cols[2].metric("Opioid claims", row.get("opioid_claims", "—"))
    sig_cols[3].metric("Peer deviation", f"{float(row.get('peer_deviation_score') or 0):.2f}" if row.get("peer_deviation_score") is not None else "—")

    fired = get_fired_rules(row)
    st.markdown("**Rules fired**")
    if fired:
        for rule_name, detail in fired:
            st.markdown(f"- **{rule_name}:** {detail}")
    else:
        st.markdown("_No rule signals fired for this prescriber._")
    raw = row.get("rules_fired")
    if raw and str(raw).strip():
        st.caption(f"Rule IDs: `{raw}`")

    npi = str(row.get("prescriber_id", "")).strip()
    if npi:
        st.markdown("---")
        render_oig_check(npi)

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

st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="🏥")
st.title(APP_TITLE)
st.caption("Medicare Part D prescriber review priority · rules v2.1 · ML · OIG screening")
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
# Docker/Render set SKLEARN_MODEL_PATH — use sklearn (bundled in image) for live predict
prefer_sklearn = bool(os.getenv("SKLEARN_MODEL_PATH"))

# ---------- LOAD MODEL (sklearn on deploy; XGB locally when only that .pkl exists) ----------
pipeline_model, spark, ml_bundle, loaded_model_name = None, None, None, None

if prefer_sklearn and sklearn_available:
    try:
        ml_bundle = joblib.load(SKLEARN_MODEL_PATH)
        loaded_model_name = "sklearn GBT"
    except Exception as e:
        st.sidebar.error(f"Failed to load sklearn model: {e}")
elif xgb_available:
    try:
        ml_bundle = joblib.load(XGB_MODEL_PATH)
        loaded_model_name = "XGBoost"
    except Exception as e:
        st.sidebar.error(f"Failed to load XGBoost model: {e}")
elif sklearn_available:
    try:
        ml_bundle = joblib.load(SKLEARN_MODEL_PATH)
        loaded_model_name = "sklearn GBT"
    except Exception as e:
        st.sidebar.error(f"Failed to load sklearn model: {e}")

if loaded_model_name:
    st.sidebar.caption(f"Live model: {loaded_model_name}")

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


def render_sidebar_status() -> None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("System status")
    if _api_ready():
        st.sidebar.success("Postgres API connected")
    elif resolve_npi_lookup_path():
        st.sidebar.info("File index (SQLite)")
    else:
        st.sidebar.warning("No lookup data source")
    if loaded_model_name:
        st.sidebar.caption(f"Prediction model: {loaded_model_name}")
    st.sidebar.caption(f"Rules version: {RISK_RULES_VERSION}")
    with st.sidebar.expander("Quick reference"):
        st.markdown(
            "- **NPI Lookup** — search by ID, OIG check, browse filters\n"
            "- **Risk Dashboard** — population stats\n"
            "- **Analyst Queue** — review workflow\n"
            "- **Low** = 0–1 pts · **High** = ≥ 2 pts"
        )


@st.cache_data(show_spinner=False)
def count_csv_data_rows(csv_path: str) -> int:
    with open(csv_path, "rb") as f:
        return max(sum(1 for _ in f) - 1, 0)


@st.cache_data(show_spinner="Loading preview...")
def load_predictions_preview(csv_path: str, nrows: int = 100) -> pd.DataFrame:
    df = pd.read_csv(csv_path, nrows=nrows)
    if "prescriber_id" in df.columns:
        df["prescriber_id"] = df["prescriber_id"].astype(str)
    return df


@st.cache_data(show_spinner="Summarizing categories...")
def category_counts_from_csv(csv_path: str) -> pd.Series:
    """Memory-safe category counts for large prediction files (Render-friendly)."""
    totals: Dict[str, int] = {}
    for chunk in pd.read_csv(csv_path, usecols=["predicted_category"], chunksize=250_000):
        for cat, cnt in chunk["predicted_category"].value_counts().items():
            totals[str(cat)] = totals.get(str(cat), 0) + int(cnt)
    return pd.Series(totals).sort_index()


def lookup_prescriber_in_csv(csv_path: str, prescriber_id: str) -> pd.DataFrame:
    target = str(prescriber_id).strip()
    for chunk in pd.read_csv(csv_path, chunksize=250_000):
        if "prescriber_id" not in chunk.columns:
            break
        chunk["prescriber_id"] = chunk["prescriber_id"].astype(str)
        hit = chunk[chunk["prescriber_id"] == target]
        if not hit.empty:
            return hit
    return pd.DataFrame()


def render_explore_model_outputs(model_key: str, label: str) -> None:
    """Show precomputed predictions for one trained model."""
    path = _predictions_path(model_key)
    if not path:
        st.warning(f"No predictions file for {label}. Run the matching train script.")
        return
    try:
        row_count = count_csv_data_rows(path)
    except Exception as e:
        st.error(f"Could not read {PREDICTIONS_FILES[model_key]}: {e}")
        return
    st.caption(f"{row_count:,} rows · `{PREDICTIONS_FILES[model_key]}` (preview shows first 100)")
    lookup_npi = st.text_input(
        "Filter by prescriber_id (optional)",
        key=f"explore_npi_{model_key}",
        placeholder="e.g. 1003000126",
    )
    if lookup_npi.strip():
        with st.spinner("Searching..."):
            view_df = lookup_prescriber_in_csv(path, lookup_npi.strip())
        if view_df.empty:
            st.warning("No rows for that prescriber_id.")
        else:
            if "prescriber_id" in view_df.columns:
                view_df["prescriber_id"] = view_df["prescriber_id"].astype(str)
            st.dataframe(view_df, use_container_width=True)
    else:
        preview = load_predictions_preview(path)
        st.dataframe(preview, use_container_width=True)
        try:
            st.bar_chart(category_counts_from_csv(path))
        except Exception as e:
            st.warning(f"Category chart skipped: {e}")

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
render_sidebar_status()
render_analyst_logout_button()

tab_npi, tab_stats, tab_queue, tab1, tab2, tab3 = st.tabs(
    [
        "🔍 NPI Lookup",
        "📊 Risk Dashboard",
        "📋 Analyst Queue",
        "Single Prediction",
        "Batch Upload",
        "Explore Outputs",
    ]
)

# --- TAB 0: NPI LOOKUP ---
with tab_npi:
    st.header("NPI Lookup")
    st.caption(
        "Enter a prescriber NPI to see rule-based review priority, fired rules, and ML prediction. "
        "No manual feature entry required."
    )
    use_api = _api_ready()
    lookup_path = resolve_npi_lookup_path() if not use_api else None

    if use_api:
        st.success(f"Connected to Postgres API")
    elif lookup_path is None:
        st.warning(
            "No data source found. Start the API (`uvicorn api.main:app --port 8000`) or build "
            "`npi_risk_lookup.sqlite.gz` for file-based lookup."
        )
    else:
        st.info(f"Using local file index: `{os.path.basename(lookup_path)}`")

    if use_api or lookup_path is not None:
        def _npi_load_example(npi: str) -> None:
            st.session_state["npi_lookup_input"] = npi
            st.session_state["npi_run_lookup"] = True

        npi_input = st.text_input(
            "Prescriber NPI (National Provider Identifier)",
            placeholder="e.g. 1003000126",
            key="npi_lookup_input",
        )
        example_col1, example_col2 = st.columns(2)
        with example_col1:
            st.button(
                "Try example Low (1003000126)",
                key="npi_ex_low",
                on_click=_npi_load_example,
                args=("1003000126",),
            )
        with example_col2:
            st.button(
                "Try example High (1003000142)",
                key="npi_ex_high",
                on_click=_npi_load_example,
                args=("1003000142",),
            )
        run_lookup = st.button("🔎 Look up", type="primary", key="npi_lookup_btn") or st.session_state.pop(
            "npi_run_lookup", False
        )
        if run_lookup:
            npi = str(st.session_state.get("npi_lookup_input", npi_input or "")).strip()
            if not npi:
                st.info("Enter an NPI above.")
            else:
                with st.spinner("Searching..."):
                    if use_api:
                        row = lookup_prescriber_via_api(npi)
                    else:
                        row = lookup_npi_in_risk_file(lookup_path, npi)
                    ml_row = lookup_ml_prediction_for_npi(npi)
                if row is None:
                    st.warning(f"No prescriber found for NPI **{npi}**.")
                else:
                    render_npi_lookup_result(row, ml_row)

        if use_api:
            st.markdown("---")
            render_prescriber_browse()

# --- TAB: RISK DASHBOARD (Step 5) ---
with tab_stats:
    st.header("Risk Dashboard")
    st.caption("Pre-aggregated stats from Postgres — no full CSV scan.")
    render_risk_dashboard()

# --- TAB: ANALYST QUEUE (Step 6 + 7) ---
with tab_queue:
    st.header("Analyst Queue")
    st.caption("Track review status for High-priority prescribers. Login required when APP_PASSWORD is set.")
    render_analyst_queue()

# --- TAB 1: SINGLE PREDICTION (manual) ---
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

# --- TAB 2: BATCH PREDICTION ---
with tab2:
    st.header("Batch Prediction")
    st.write(
        "Upload a CSV with `prescriber_id` and model feature columns "
        f"({', '.join(ML_FEATURE_COLS)}). "
        "Try `Data/Model_Data/sample_batch_input.csv` from the repo."
    )
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

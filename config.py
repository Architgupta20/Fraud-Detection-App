"""
Project paths and shared configuration.

Set BASE_DIR via environment variable to override the repo root, e.g.:

    export BASE_DIR="$(pwd)"
"""

from __future__ import annotations

import os
from pathlib import Path


def get_base_dir() -> Path:
    env = os.environ.get("BASE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parent


BASE_DIR = get_base_dir()
DATA_DIR = BASE_DIR / "Data"
MODELS_DIR = BASE_DIR / "Models"
OUTPUTS_DIR = BASE_DIR / "Outputs"

# --- Data folders (one stage per folder — see Data/README.md) ---
ORIGINAL_DATA_DIR = DATA_DIR / "Original_Datasets"
CLEANED_DATA_DIR = DATA_DIR / "Cleaned_Datasets"
AGGREGATED_DATA_DIR = DATA_DIR / "Aggregated_Datasets"
ENRICHED_DATA_DIR = DATA_DIR / "Enriched_Datasets"
SCORED_DATA_DIR = DATA_DIR / "Scored_Datasets"
MODEL_DATA_DIR = DATA_DIR / "Model_Data"
SPARK_TEMP_DIR = DATA_DIR / "_Spark_Temp"

# Raw CMS downloads
PART_D_PRESCRIBERS_CSV = ORIGINAL_DATA_DIR / "part_d_prescribers.csv"
OPEN_PAYMENTS_CSV = ORIGINAL_DATA_DIR / "open_payments.csv"

# After clean stage
CLEAN_PRESCRIBERS_CSV = CLEANED_DATA_DIR / "clean_prescribers.csv"
CLEAN_PAYMENTS_CSV = CLEANED_DATA_DIR / "clean_payments.csv"

# After aggregate stage
PRESCRIBER_LEVEL_CSV = AGGREGATED_DATA_DIR / "prescriber_level_dataset.csv"

# After features stage
PRESCRIBER_LEVEL_ENRICHED_CSV = ENRICHED_DATA_DIR / "prescriber_level_enriched.csv"

# After score stage (main file for ML + app)
FRAUD_RISK_SCORED_CSV = SCORED_DATA_DIR / "fraud_risk_scored_prescribers.csv"

# Slim NPI lookup index (build via Scripts/build_npi_lookup_index.py)
NPI_LOOKUP_CSV = MODEL_DATA_DIR / "npi_risk_lookup.csv"
NPI_LOOKUP_SQLITE = MODEL_DATA_DIR / "npi_risk_lookup.sqlite"
NPI_LOOKUP_SQLITE_GZ = MODEL_DATA_DIR / "npi_risk_lookup.sqlite.gz"

# Risk rules spec: risk_rules.py + docs/RISK_RULES.md
try:
    from risk_rules import RULES_VERSION as RISK_RULES_VERSION
except ImportError:
    RISK_RULES_VERSION = "unknown"

# Model artifacts
GBT_SKLEARN_PKL = MODELS_DIR / "gbt_sklearn.pkl"
XGB_CALIBRATED_PKL = MODELS_DIR / "xgb_calibrated.pkl"
SPARK_PIPELINE_MODEL_DIR = MODELS_DIR / "spark_pipeline_model"


def data_path(*parts: str) -> Path:
    """Legacy helper — prefer stage folders above. Temp Spark files use _Spark_Temp."""
    return DATA_DIR.joinpath(*parts)


def model_data_path(*parts: str) -> Path:
    return MODEL_DATA_DIR.joinpath(*parts)


def spark_temp_path(*parts: str) -> Path:
    return SPARK_TEMP_DIR.joinpath(*parts)

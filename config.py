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
MODEL_DATA_DIR = DATA_DIR / "Model_Data"
OUTPUTS_DIR = BASE_DIR / "Outputs"

# Raw inputs
PART_D_PRESCRIBERS_CSV = DATA_DIR / "part_d_prescribers.csv"
OPEN_PAYMENTS_CSV = DATA_DIR / "open_payments.csv"

# Cleaned
CLEAN_PRESCRIBERS_CSV = DATA_DIR / "clean_prescribers.csv"
CLEAN_PAYMENTS_CSV = DATA_DIR / "clean_payments.csv"

# Aggregated
PRESCRIBER_LEVEL_CSV = DATA_DIR / "prescriber_level_dataset.csv"
MERGED_PAYMENT_LEVEL_CSV = DATA_DIR / "merged_payment_level_dataset.csv"
PRESCRIBER_LEVEL_ENRICHED_CSV = DATA_DIR / "prescriber_level_enriched.csv"
FRAUD_RISK_SCORED_CSV = DATA_DIR / "fraud_risk_scored_prescribers.csv"

# Model artifacts
GBT_SKLEARN_PKL = MODELS_DIR / "gbt_sklearn.pkl"
SPARK_PIPELINE_MODEL_DIR = MODELS_DIR / "spark_pipeline_model"


def data_path(*parts: str) -> Path:
    return DATA_DIR.joinpath(*parts)


def model_data_path(*parts: str) -> Path:
    return MODEL_DATA_DIR.joinpath(*parts)

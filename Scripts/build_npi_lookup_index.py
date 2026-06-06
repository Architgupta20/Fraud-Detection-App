#!/usr/bin/env python3
"""
Build a slim NPI lookup CSV from the full scored file (~100MB vs ~400MB).

Use for Streamlit / Docker / Render when the full scored file is not shipped.

    export BASE_DIR="$(pwd)"
    python Scripts/build_npi_lookup_index.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import FRAUD_RISK_SCORED_CSV, NPI_LOOKUP_CSV

# Columns needed for NPI Lookup tab + rule explanations
LOOKUP_COLUMNS = [
    "prescriber_id",
    "first_name",
    "last_name",
    "state",
    "city",
    "provider_type",
    "fraud_risk_category",
    "risk_points",
    "rules_fired",
    "rules_version",
    "total_claims",
    "total_drug_cost",
    "opioid_claims",
    "payment_to_drug_cost_ratio",
    "peer_deviation_score",
    "avg_risk_score",
    "payment_variability",
    "adjusted_risk_payment",
    "high_payment_flag",
    "high_opioid_flag",
    "elderly_focus_flag",
    "antibiotic_claim_ratio",
    "antibiotic_claims",
    "total_payment_amount",
    "opioid_volume_pct_flag",
    "peer_outlier_pct_flag",
    "payment_spiky_pct_flag",
    "total_payments_pct_flag",
]


def main() -> None:
    src = Path(FRAUD_RISK_SCORED_CSV)
    dest = Path(NPI_LOOKUP_CSV)
    if not src.exists():
        raise FileNotFoundError(f"Scored CSV not found: {src}. Run: python run_pipeline.py score")

    import pandas as pd

    print(f"Reading {src} (columns subset)...")
    df = pd.read_csv(src, usecols=lambda c: c in LOOKUP_COLUMNS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"Wrote {len(df):,} rows -> {dest} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build NPI lookup index for Streamlit (local + Render).

Creates a gzipped SQLite DB (target <95 MB for GitHub) with prescriber_id as
PRIMARY KEY (WITHOUT ROWID). Name/city/rules_version are omitted; the app
shows "—" for names and uses config.RISK_RULES_VERSION.

    export BASE_DIR="$(pwd)"
    python Scripts/build_npi_lookup_index.py
"""

from __future__ import annotations

import gzip
import shutil
import sqlite3
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import FRAUD_RISK_SCORED_CSV, NPI_LOOKUP_SQLITE, NPI_LOOKUP_SQLITE_GZ

# Slim columns — omit first_name, last_name, city, rules_version (UI handles these)
LOOKUP_COLUMNS = [
    "prescriber_id",
    "state",
    "provider_type",
    "fraud_risk_category",
    "risk_points",
    "rules_fired",
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

CREATE_TABLE_SQL = """
CREATE TABLE prescribers (
    prescriber_id TEXT NOT NULL PRIMARY KEY,
    state TEXT,
    provider_type TEXT,
    fraud_risk_category TEXT,
    risk_points INTEGER,
    rules_fired TEXT,
    total_claims INTEGER,
    total_drug_cost REAL,
    opioid_claims INTEGER,
    payment_to_drug_cost_ratio REAL,
    peer_deviation_score REAL,
    avg_risk_score REAL,
    payment_variability REAL,
    adjusted_risk_payment REAL,
    high_payment_flag INTEGER,
    high_opioid_flag INTEGER,
    elderly_focus_flag INTEGER,
    antibiotic_claim_ratio REAL,
    antibiotic_claims INTEGER,
    total_payment_amount REAL,
    opioid_volume_pct_flag INTEGER,
    peer_outlier_pct_flag INTEGER,
    payment_spiky_pct_flag INTEGER,
    total_payments_pct_flag INTEGER
) WITHOUT ROWID
"""

INSERT_SQL = """
INSERT INTO prescribers (
    prescriber_id, state, provider_type, fraud_risk_category, risk_points,
    rules_fired, total_claims, total_drug_cost, opioid_claims,
    payment_to_drug_cost_ratio, peer_deviation_score, avg_risk_score,
    payment_variability, adjusted_risk_payment, high_payment_flag,
    high_opioid_flag, elderly_focus_flag, antibiotic_claim_ratio,
    antibiotic_claims, total_payment_amount, opioid_volume_pct_flag,
    peer_outlier_pct_flag, payment_spiky_pct_flag, total_payments_pct_flag
) VALUES (
    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
)
"""

INT_COLS = {
    "risk_points",
    "total_claims",
    "opioid_claims",
    "high_payment_flag",
    "high_opioid_flag",
    "elderly_focus_flag",
    "antibiotic_claims",
    "opioid_volume_pct_flag",
    "peer_outlier_pct_flag",
    "payment_spiky_pct_flag",
    "total_payments_pct_flag",
}

REAL_COLS = {
    "total_drug_cost",
    "payment_to_drug_cost_ratio",
    "peer_deviation_score",
    "avg_risk_score",
    "payment_variability",
    "adjusted_risk_payment",
    "antibiotic_claim_ratio",
    "total_payment_amount",
}


def _cell(row, col):
    val = row[col]
    if val is None or (isinstance(val, float) and val != val):  # NaN
        return None
    if col in INT_COLS:
        return int(val)
    if col in REAL_COLS:
        return float(val)
    return str(val) if val is not None else None


def main() -> None:
    import pandas as pd

    src = Path(FRAUD_RISK_SCORED_CSV)
    db_path = Path(NPI_LOOKUP_SQLITE)
    gz_path = Path(NPI_LOOKUP_SQLITE_GZ)
    if not src.exists():
        raise FileNotFoundError(f"Scored CSV not found: {src}. Run: python run_pipeline.py score")

    print(f"Reading {src} ...")
    df = pd.read_csv(src, usecols=lambda c: c in LOOKUP_COLUMNS, dtype={"prescriber_id": str})
    df["prescriber_id"] = df["prescriber_id"].astype(str)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    print(f"Writing SQLite ({len(df):,} rows, WITHOUT ROWID) ...")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute(CREATE_TABLE_SQL)

    rows = [
        tuple(_cell(r, c) for c in LOOKUP_COLUMNS)
        for r in df.to_dict("records")
    ]
    conn.executemany(INSERT_SQL, rows)
    conn.commit()
    print("Running VACUUM ...")
    conn.execute("VACUUM")
    conn.close()

    print(f"Compressing -> {gz_path} ...")
    with open(db_path, "rb") as f_in, gzip.open(gz_path, "wb", compresslevel=9) as f_out:
        shutil.copyfileobj(f_in, f_out)

    db_mb = db_path.stat().st_size / (1024 * 1024)
    gz_mb = gz_path.stat().st_size / (1024 * 1024)
    print(f"Done. SQLite {db_mb:.1f} MB | gzip {gz_mb:.1f} MB (commit the .gz for Render)")
    if gz_mb >= 95:
        print(f"WARNING: gzip is {gz_mb:.1f} MB — exceeds 95 MB GitHub target.")


if __name__ == "__main__":
    main()

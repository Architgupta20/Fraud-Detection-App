#!/usr/bin/env python3
"""
Load scored prescribers CSV into Render Postgres (Phase 2 Step 2.3).

    export BASE_DIR="$(pwd)"
    python Scripts/load_prescribers_to_postgres.py

Uses DATABASE_URL from .env or environment (Render External Database URL).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import FRAUD_RISK_SCORED_CSV

COLUMNS = [
    "prescriber_id",
    "state",
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

BATCH_SIZE = 10_000


def _load_dotenv() -> None:
    env_path = _ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _clean(val, as_int: bool = False, as_float: bool = False):
    if val is None or (isinstance(val, float) and val != val):
        return None
    if as_int:
        return int(val)
    if as_float:
        return float(val)
    s = str(val).strip()
    return s if s and s.lower() != "nan" else None


def main() -> None:
    _load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL missing. Put it in .env (see .env.example).")

    src = Path(FRAUD_RISK_SCORED_CSV)
    if not src.exists():
        raise SystemExit(f"Scored CSV not found: {src}\nRun: python run_pipeline.py score")

    try:
        import pandas as pd
        import psycopg2
        from psycopg2.extras import execute_values
    except ImportError as exc:
        raise SystemExit("Run: pip install pandas psycopg2-binary") from exc

    placeholders = ", ".join(COLUMNS)
    insert_sql = f"""
        INSERT INTO prescribers ({placeholders})
        VALUES %s
        ON CONFLICT (prescriber_id) DO UPDATE SET
            state = EXCLUDED.state,
            provider_type = EXCLUDED.provider_type,
            fraud_risk_category = EXCLUDED.fraud_risk_category,
            risk_points = EXCLUDED.risk_points,
            rules_fired = EXCLUDED.rules_fired,
            rules_version = EXCLUDED.rules_version,
            total_claims = EXCLUDED.total_claims,
            total_drug_cost = EXCLUDED.total_drug_cost,
            opioid_claims = EXCLUDED.opioid_claims,
            payment_to_drug_cost_ratio = EXCLUDED.payment_to_drug_cost_ratio,
            peer_deviation_score = EXCLUDED.peer_deviation_score,
            avg_risk_score = EXCLUDED.avg_risk_score,
            payment_variability = EXCLUDED.payment_variability,
            adjusted_risk_payment = EXCLUDED.adjusted_risk_payment,
            high_payment_flag = EXCLUDED.high_payment_flag,
            high_opioid_flag = EXCLUDED.high_opioid_flag,
            elderly_focus_flag = EXCLUDED.elderly_focus_flag,
            antibiotic_claim_ratio = EXCLUDED.antibiotic_claim_ratio,
            antibiotic_claims = EXCLUDED.antibiotic_claims,
            total_payment_amount = EXCLUDED.total_payment_amount,
            opioid_volume_pct_flag = EXCLUDED.opioid_volume_pct_flag,
            peer_outlier_pct_flag = EXCLUDED.peer_outlier_pct_flag,
            payment_spiky_pct_flag = EXCLUDED.payment_spiky_pct_flag,
            total_payments_pct_flag = EXCLUDED.total_payments_pct_flag,
            loaded_at = NOW()
    """

    int_cols = {
        "risk_points", "total_claims", "opioid_claims", "high_payment_flag",
        "high_opioid_flag", "elderly_focus_flag", "antibiotic_claims",
        "opioid_volume_pct_flag", "peer_outlier_pct_flag",
        "payment_spiky_pct_flag", "total_payments_pct_flag", "antibiotic_claims",
    }
    float_cols = {
        "total_drug_cost", "payment_to_drug_cost_ratio", "peer_deviation_score",
        "avg_risk_score", "payment_variability", "adjusted_risk_payment",
        "antibiotic_claim_ratio", "total_payment_amount",
    }

    print(f"Reading {src} in batches of {BATCH_SIZE:,} ...")
    total = 0
    with psycopg2.connect(url) as conn:
        with conn.cursor() as cur:
            for chunk in pd.read_csv(
                src,
                usecols=lambda c: c in COLUMNS,
                dtype={"prescriber_id": str},
                chunksize=BATCH_SIZE,
            ):
                rows = []
                for rec in chunk.to_dict("records"):
                    rows.append(tuple(
                        _clean(rec.get(c), as_int=c in int_cols, as_float=c in float_cols)
                        for c in COLUMNS
                    ))
                execute_values(cur, insert_sql, rows, page_size=1000)
                conn.commit()
                total += len(rows)
                print(f"  loaded {total:,} rows ...")

    with psycopg2.connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM prescribers")
            count = cur.fetchone()[0]

    print(f"Done. Postgres has {count:,} prescribers.")


if __name__ == "__main__":
    main()

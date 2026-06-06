-- Phase 2 Step 2: prescribers table (scored Medicare Part D prescribers)
-- Run via: Scripts/init_postgres_schema.py

CREATE TABLE IF NOT EXISTS prescribers (
    prescriber_id TEXT NOT NULL PRIMARY KEY,
    state TEXT,
    provider_type TEXT,
    fraud_risk_category TEXT NOT NULL,
    risk_points INTEGER NOT NULL DEFAULT 0,
    rules_fired TEXT,
    rules_version TEXT,
    total_claims BIGINT,
    total_drug_cost DOUBLE PRECISION,
    opioid_claims BIGINT,
    payment_to_drug_cost_ratio DOUBLE PRECISION,
    peer_deviation_score DOUBLE PRECISION,
    avg_risk_score DOUBLE PRECISION,
    payment_variability DOUBLE PRECISION,
    adjusted_risk_payment DOUBLE PRECISION,
    high_payment_flag SMALLINT,
    high_opioid_flag SMALLINT,
    elderly_focus_flag SMALLINT,
    antibiotic_claim_ratio DOUBLE PRECISION,
    antibiotic_claims BIGINT,
    total_payment_amount DOUBLE PRECISION,
    opioid_volume_pct_flag SMALLINT,
    peer_outlier_pct_flag SMALLINT,
    payment_spiky_pct_flag SMALLINT,
    total_payments_pct_flag SMALLINT,
    loaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_prescribers_state ON prescribers (state);
CREATE INDEX IF NOT EXISTS idx_prescribers_category ON prescribers (fraud_risk_category);
CREATE INDEX IF NOT EXISTS idx_prescribers_state_category ON prescribers (state, fraud_risk_category);

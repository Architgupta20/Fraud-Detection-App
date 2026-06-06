-- Phase 2 Step 8: HHS OIG LEIE exclusion list (NPI-indexed subset)
-- Run via: Scripts/init_postgres_schema.py

CREATE TABLE IF NOT EXISTS oig_exclusions (
    npi TEXT NOT NULL PRIMARY KEY,
    last_name TEXT,
    first_name TEXT,
    business_name TEXT,
    specialty TEXT,
    state TEXT,
    exclusion_type TEXT,
    exclusion_date TEXT,
    reinstatement_date TEXT,
    loaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_oig_exclusions_state ON oig_exclusions (state);

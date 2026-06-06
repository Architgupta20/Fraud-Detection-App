-- Phase 2 Step 6: analyst review queue
-- Run via: Scripts/init_postgres_schema.py

CREATE TABLE IF NOT EXISTS reviews (
    prescriber_id TEXT NOT NULL PRIMARY KEY
        REFERENCES prescribers (prescriber_id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'reviewed', 'needs_followup')),
    note TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_reviews_status ON reviews (status);
CREATE INDEX IF NOT EXISTS idx_reviews_updated_at ON reviews (updated_at DESC);

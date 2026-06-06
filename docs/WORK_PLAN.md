# Work plan

## Phase 1 — Complete

- [x] PySpark ETL + `risk_rules.py` v2.1 (binary Low/High)
- [x] Full score + train (XGB + sklearn, `--strict-rules-version`)
- [x] Streamlit demo (disclaimer, Explore sub-tabs, Render deploy)
- [x] Live demo — https://fraud-detection-app-9pen.onrender.com/
- [x] README + screenshots; GitHub portfolio trimmed for deploy

---

## Phase 2 — Product roadmap

Goal: turn the demo into something an analyst can **use** — NPI lookup first, then database + API, then workflow.

### Step 1 — NPI lookup ⭐ **done**

**Done in app:** **NPI Lookup** tab — enter NPI → category, points, rules fired, ML prediction.

**Render fallback:** `npi_risk_lookup.sqlite.gz` when API unavailable.

---

### Step 5 — Pre-aggregated stats ✅ **done**

**API:** `GET /stats/summary`, `/stats/by-state`, `/stats/top-risk`

**App:** **Risk Dashboard** tab (charts + top prescribers from Postgres).

---

### Step 6 — Analyst queue ✅ **done**

**Table:** `reviews` (prescriber_id, status, note, updated_at)

**API:** `GET /reviews`, `PUT /reviews/{npi}`, `GET /reviews/export`

**App:** **Analyst Queue** tab — filter High risk, save status, export CSV.

---

### Step 7 — Auth ✅ **done**

**Streamlit:** `APP_PASSWORD` gates Analyst Queue tab.

**API:** `APP_API_KEY` required for review writes/export (`X-API-Key` header).

Set the **same** `APP_API_KEY` on both Render services.

---

### Step 2 — Postgres

**What:** Load scored prescriber data into Postgres instead of giant CSVs.

**Why:**

- Fast NPI lookup (index on `prescriber_id`)
- Filters: “All High risk in TX”
- History when rules are re-run
- Multi-user access

**Flow:**

1. Create table e.g. `prescribers` (NPI, category, points, rules_fired, state, provider_type, …)
2. One-time ETL script: scored CSV → Postgres
3. Index on `prescriber_id`

**Host (free/cheap):** Render Postgres, Supabase, or Neon.

---

### Step 3 — FastAPI backend

**What:** Small API, e.g.:

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Liveness |
| `GET /prescribers/{npi}` | One prescriber score + rules |
| `GET /prescribers?risk=High&state=TX&limit=50` | Filtered list |

**Order:** Postgres (Step 2) **then** FastAPI reads from DB (Step 3).

**Why:** Streamlit calls API instead of loading 75MB+ CSVs on every visit → fixes slow Explore on Render.

---

### Step 4 — Streamlit → API

**What:**

- Lookup tab → `GET /prescribers/{npi}`
- Explore / list → `GET /prescribers?...` with filters

**Why:** Faster, cleaner live app; same UI, better backend.

---

### Step 5 — Pre-aggregated stats

**What:** Precompute small summary tables (not full CSV scans on each visit):

- Count by risk category
- Count by state
- Top N highest risk

**How:** Script after each `run_pipeline.py score`, or `GET /stats/by-state` etc.

**Why:** Free Render stays responsive.

---

### Step 6 — Analyst queue

**What:** Workflow: filter High → mark Reviewed / Needs follow-up → export CSV.

**Needs:** Postgres table e.g. `reviews` (npi, status, note, updated_at).

**Why:** First **product** feature beyond lookup.

---

### Step 7 — Auth

**What:** Login before app or before changing review status.

**Why later:** Prove value with lookup + DB first. Demo can work without auth.

**Options:** Streamlit password, Supabase Auth, Google OAuth.

---

### Step 8 — OIG / exclusion list (validation)

**What:** Match prescribers against OIG exclusion list.

**Why last:** Bonus trust signal; core product is still rule-based prioritization.

---

## Suggested build order

```
Step 1 (NPI in Streamlit, local/sample)
    → Step 2 (Postgres)
    → Step 3 (FastAPI)
    → Step 4 (Streamlit → API)
    → Step 5 (stats)
    → Step 6 (queue)
    → Step 7 (auth)
    → Step 8 (OIG)
```

See [README.md](../README.md) for Phase 1 setup and live demo.

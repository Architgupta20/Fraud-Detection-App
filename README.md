# Prescriber Risk Prioritization Platform

Rule-based **Medicare Part D prescriber review priority** (Low / High) using CMS prescribing + Open Payments data, PySpark ETL, calibrated ML, **HHS OIG exclusion screening**, and a production-style **Postgres + FastAPI + Streamlit** stack on Render.

> **Disclaimer:** Labels are **not confirmed fraud** — they prioritize human review only. OIG matches indicate federal exclusion list presence, not automatic eligibility determinations.

**Repository:** [github.com/Architgupta20/Fraud-Detection-App](https://github.com/Architgupta20/Fraud-Detection-App)

## Live demo

**[https://fraud-detection-app-9pen.onrender.com/](https://fraud-detection-app-9pen.onrender.com/)**

| NPI Lookup | Explore outputs (~1.38M prescribers) |
|------------|--------------------------------------|
| ![Single prediction](docs/images/demo-single-prediction.png) | ![Explore outputs](docs/images/demo-explore-outputs.png) |

Render **free tier** sleeps after ~15 min idle; first visit may take 30–60s to wake.

### App tabs

| Tab | Purpose |
|-----|---------|
| **NPI Lookup** | Enter NPI → review priority, rules fired, ML prediction, **OIG LEIE check**, browse by state/risk |
| **Risk Dashboard** | Population stats, charts, top-risk prescribers, OIG overlap counts |
| **Analyst Queue** | Filter High-priority prescribers, mark reviewed / needs follow-up, export CSV |
| **Single / Batch** | Manual or CSV batch ML prediction |
| **Explore** | Precomputed XGB / sklearn prediction previews |

---

## Highlights

| | |
|--|--|
| **Scale** | ~1.38M prescribers after ETL |
| **Rules** | v2.1 additive points → **Low** (0–1 pts) / **High** (≥ 2 pts) |
| **ML** | 80/20 holdout; XGB 91.4% acc / sklearn 90.7% on validation |
| **OIG** | ~8.4k LEIE NPI exclusions loaded; cross-check on lookup |
| **Stack** | PySpark, pandas, XGBoost, sklearn, FastAPI, Postgres, Streamlit, Docker, Render |
| **Phase** | **Phase 1 + Phase 2 complete** (NPI → Postgres → API → dashboard → queue → auth → OIG) |

---

## Architecture

```
CMS CSVs (local)
    → run_pipeline.py (clean → aggregate → features → score)
    → fraud_risk_scored_prescribers.csv
         ├─→ load_prescribers_to_postgres.py → Render Postgres (prescribers)
         ├─→ load_oig_to_postgres.py        → Render Postgres (oig_exclusions)
         └─→ build_npi_lookup_index.py     → npi_risk_lookup.sqlite.gz (fallback)

Render Postgres
    → prescriber-risk-api (FastAPI, Dockerfile.api)
    → fraud-detection-app (Streamlit, Dockerfile)
         Tabs call API via API_BASE_URL
         Falls back to SQLite index if API unavailable
```

**Single source of truth for rules:** `risk_rules.py`. Details: [docs/RISK_RULES.md](docs/RISK_RULES.md).

---

## What is in this repo (Git)

| Included | Not in Git (local only) |
|----------|-------------------------|
| `risk_rules.py`, `run_pipeline.py`, `config.py` | CMS raw CSVs (~18 GB) |
| `api/` FastAPI service, `api_client.py`, `Dockerfile.api` | `.env` (DATABASE_URL, secrets) |
| `Models/train_*.py`, `gbt_sklearn.pkl` | `xgb_calibrated.pkl` |
| `Outputs/Reports/streamlit_app.py` | Full scored CSV (~400 MB) |
| `npi_risk_lookup.sqlite.gz` (~72 MB deploy index) | OIG LEIE CSV (~15 MB, downloaded by load script) |
| `fraud_detection_gbt_sklearn_predictions.csv` | XGB predictions CSV |
| Postgres + OIG scripts (`Scripts/load_*.py`, `Scripts/sql/`) | |

---

## Quick start (local)

```bash
git clone https://github.com/Architgupta20/Fraud-Detection-App.git
cd Fraud-Detection-App
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-spark.txt   # ETL only; Java 17+
```

1. **Data** — download CMS files per [docs/DATA.md](docs/DATA.md) into `Data/Original_Datasets/`
2. **ETL** — `export BASE_DIR="$(pwd)" && python run_pipeline.py all`
3. **Train** — `python Models/train_xgb.py --sample-frac 1.0 --strict-rules-version` (and sklearn)
4. **Postgres** — copy `.env.example` to `.env`, set `DATABASE_URL`, then:
   ```bash
   python Scripts/init_postgres_schema.py
   python Scripts/load_prescribers_to_postgres.py
   python Scripts/load_oig_to_postgres.py      # downloads HHS OIG LEIE CSV
   ```
5. **API + app** — two terminals:
   ```bash
   # Terminal 1
   uvicorn api.main:app --reload --port 8000

   # Terminal 2
   export API_BASE_URL=http://localhost:8000
   streamlit run Outputs/Reports/streamlit_app.py
   ```

Without Postgres/API, Streamlit still works via `npi_risk_lookup.sqlite.gz` after `python Scripts/build_npi_lookup_index.py`.

Preview CSVs safely: `python Scripts/inspect_csv.py scored --rows 5`

---

## API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness + DB connection |
| GET | `/prescribers/{npi}` | One prescriber (rules + metrics) |
| GET | `/prescribers?risk=High&state=TX&limit=50` | Filtered list |
| GET | `/stats/summary` | Counts by review priority |
| GET | `/stats/by-state` | Prescriber counts by state |
| GET | `/stats/top-risk` | Highest risk_points prescribers |
| GET | `/stats/oig-overlap` | OIG exclusions overlapping prescriber panel |
| GET | `/reviews` | Analyst queue (prescribers + review status) |
| PUT | `/reviews/{npi}` | Update review status (requires `X-API-Key`) |
| GET | `/reviews/export` | CSV export (requires `X-API-Key`) |
| GET | `/oig/check/{npi}` | OIG LEIE exclusion match for NPI |

Interactive docs: `http://localhost:8000/docs` when running locally.

---

## Model results (v2.1, full data)

| Model | Val accuracy | Macro-F1 | High recall |
|-------|--------------|----------|-------------|
| XGBoost (calibrated) | 91.4% | 0.897 | 85.9% |
| sklearn GBT | 90.7% | 0.888 | 85.0% |

Train/val split: **80% / 20%** by hashed `prescriber_id`.

---

## Deploy (Render)

Three Render resources: **Postgres** + **two web services**.

| Service | Dockerfile | Key env vars |
|---------|------------|--------------|
| **prescriber-risk-api** | `Dockerfile.api` | `DATABASE_URL` (Internal), `APP_API_KEY` |
| **fraud-detection-app** | `Dockerfile` | `BASE_DIR`, `MODEL_DATA_DIR`, `SKLEARN_MODEL_PATH`, `API_BASE_URL`, `APP_API_KEY`, `APP_PASSWORD` (optional) |

**API service:** Health check `/health`, Dockerfile path `Dockerfile.api`.

**After deploy — load data once from your Mac:**

```bash
export BASE_DIR="$(pwd)"
# DATABASE_URL = Postgres External URL from Render
python Scripts/init_postgres_schema.py
python Scripts/load_prescribers_to_postgres.py
python Scripts/load_oig_to_postgres.py
```

**Auth (optional):**

| Variable | Service | Purpose |
|----------|---------|---------|
| `APP_PASSWORD` | Streamlit | Login gate on Analyst Queue tab |
| `APP_API_KEY` | API + Streamlit (same value) | Protect review save / CSV export |

**Docker (Streamlit only, local)**

```bash
docker build -t prescriber-risk-app .
docker run --rm -p 8502:8501 prescriber-risk-app
```

Open **http://localhost:8502**.

---

## Repository layout

```
├── risk_rules.py              # Rules + ML features
├── run_pipeline.py            # PySpark ETL
├── api/                       # FastAPI (Postgres-backed)
├── api_client.py              # Streamlit → API client
├── Models/                    # train_xgb, train_sklearn, gbt_sklearn.pkl
├── Outputs/Reports/           # Streamlit app
├── Scripts/                   # NPI index, Postgres/OIG load, schema SQL
├── Data/Model_Data/           # predictions, sqlite.gz lookup index
├── docs/                      # DATA, RISK_RULES, images, WORK_PLAN
├── Dockerfile                 # Streamlit
├── Dockerfile.api             # FastAPI
├── render.yaml
└── requirements.txt
```

---

## Documentation

| Doc | Topic |
|-----|--------|
| [docs/DATA.md](docs/DATA.md) | CMS downloads |
| [docs/RISK_RULES.md](docs/RISK_RULES.md) | Scoring rules v2.1 |
| [docs/LABEL_LEAKAGE.md](docs/LABEL_LEAKAGE.md) | ML feature boundaries |
| [docs/SPARK.md](docs/SPARK.md) | PySpark 3.5.5 / Python 3.13 |
| [docs/WORK_PLAN.md](docs/WORK_PLAN.md) | Phase 2 plan (complete) |

---

## Phase 2 roadmap — complete

| Step | Feature | Status |
|------|---------|--------|
| 1 | NPI Lookup | Done |
| 2 | Postgres + load data | Done |
| 3 | FastAPI backend | Done |
| 4 | Streamlit → API | Done |
| 5 | Risk Dashboard (stats) | Done |
| 6 | Analyst review queue | Done |
| 7 | Auth (`APP_PASSWORD` / `APP_API_KEY`) | Done |
| 8 | OIG LEIE exclusion screening | Done |

---

## Ethics

Public CMS and OIG data — do not present model output or OIG matches as proof of fraud or program ineligibility without official verification.

## License

MIT — see [LICENSE](LICENSE).

## Author

LTI Mindtree internship — healthcare analytics / M.Tech review.

# Prescriber Risk Prioritization Platform

Analyze **Medicare Part D prescribers** together with **CMS Open Payments** to build risk profiles, apply transparent scoring rules, train machine learning models, and explore results in a **Streamlit** web app.

> **Disclaimer:** Labels are **rule-based review priority** (Low / High), not confirmed fraud. Use this system to **prioritize human review**, not as legal proof of wrongdoing.

**Repository:** [github.com/Architgupta20/Fraud-Detection-App](https://github.com/Architgupta20/Fraud-Detection-App)

## Live demo

**[Prescriber Risk Prioritization (Render)](https://fraud-detection-app-9pen.onrender.com/)**

| Single prediction | Explore model outputs (~1.38M rows) |
|-------------------|-------------------------------------|
| ![Single prediction](docs/images/demo-single-prediction.png) | ![Explore outputs](docs/images/demo-explore-outputs.png) |

First load on Render **free tier** can take **30–60+ seconds** (service wakes from sleep). After that, Single Prediction is usually responsive; **Explore** shows a 100-row preview plus chunked summaries of the full prediction files.

---

## What this project does

| Step | What happens |
|------|----------------|
| **Ingest** | Load public CMS prescribing + payment CSVs (~18 GB locally) |
| **Transform** | PySpark pipeline: clean → aggregate → engineer features |
| **Score** | Apply product rules from `risk_rules.py` → risk points + category |
| **Train** | Calibrated XGBoost + sklearn GBT on scored data (binary Low/High) |
| **Serve** | Streamlit demo for single lookup, batch CSV, and model outputs |

**Scale:** ~1.38M prescriber rows after aggregation.

---

## Who it is for

- Healthcare analytics and compliance teams triaging prescribers for review  
- Researchers studying payment–prescribing patterns  
- Internship / thesis demos with a real data pipeline at scale  

---

## Architecture

```
CMS CSVs (local, not in Git)
        │
        ▼
run_pipeline.py          ← PySpark ETL
  clean → aggregate → features → score
        │
        ▼
fraud_risk_scored_prescribers.csv
        │
        ├─► Models/train_xgb.py      ← production path (recommended)
        └─► Models/train_sklearn.py  ← lightweight deploy fallback
        │
        ▼
Outputs/Reports/streamlit_app.py
```

### Single source of truth: `risk_rules.py`

All **scoring rules**, **label definitions**, and **ML feature lists** live in one file:

- Pipeline scoring: `run_pipeline.py score` → `apply_risk_scoring_spark()`
- Training: `Models/ml_common.py` imports `ML_FEATURE_COLS`, `LABEL_COL`
- UI explanations: Streamlit reads the same rule IDs

After changing rules: re-score, then retrain. See [docs/RISK_RULES.md](docs/RISK_RULES.md).

---

## Repository layout

```
Project/
├── risk_rules.py           # Rules + ML feature config (edit here first)
├── run_pipeline.py         # PySpark ETL CLI
├── config.py               # Paths (BASE_DIR env override)
├── Data/                   # Stage folders — see Data/README.md
│   ├── Original_Datasets/  # CMS downloads (2 raw files)
│   ├── Cleaned_Datasets/
│   ├── Aggregated_Datasets/  # merged prescriber + payments
│   ├── Enriched_Datasets/
│   ├── Scored_Datasets/    # ML + app input
│   └── Model_Data/
├── Models/                 # Training scripts + saved .pkl models
├── Scripts/                # inspect_csv.py, migrate_data_layout.sh
├── Outputs/Reports/        # Streamlit app
├── docs/                   # DATA, RISK_RULES, LABEL_LEAKAGE, SPARK
├── Dockerfile              # Streamlit container (root)
└── render.yaml             # Render.com deploy config
```

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/Architgupta20/Fraud-Detection-App.git
cd Fraud-Detection-App

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-spark.txt   # PySpark 3.5.5 for ETL (Python 3.13+)
```

**ETL requirements:** Java **17+** (Java 21 works) and PySpark from `requirements-spark.txt`.  
On macOS with XGBoost: `brew install libomp` if import fails.

If you still have old **Low/Medium/High** scored or prediction files:

```bash
bash Scripts/remove_legacy_3class_artifacts.sh
```

### 2. Download data (local only)

Follow [docs/DATA.md](docs/DATA.md). Download CMS files into:

- `Data/Original_Datasets/part_d_prescribers.csv`
- `Data/Original_Datasets/open_payments.csv`

If you still have old flat `Data/*.csv` files, run: `bash Scripts/migrate_data_layout.sh`

**Do not open multi-GB CSVs in Excel** — use:

```bash
export BASE_DIR="$(pwd)"
python Scripts/inspect_csv.py scored --rows 5
```

### 3. Run the pipeline

```bash
export BASE_DIR="$(pwd)"
python run_pipeline.py all
# or: clean | aggregate | features | score
```

Output: `Data/Scored_Datasets/fraud_risk_scored_prescribers.csv`

### 4. Train models

Uses **100% of the scored CSV** (`--sample-frac 1.0`) with an **80% train / 20% validation** split by hashed `prescriber_id` (metrics are on the held-out 20%). Prediction CSVs are written for all ~1.38M prescribers after training.

```bash
export BASE_DIR="$(pwd)"
python Models/train_xgb.py --sample-frac 1.0 --strict-rules-version
python Models/train_sklearn.py --sample-frac 1.0 --strict-rules-version

# Quick smoke test on laptop
python Models/train_xgb.py --nrows 50000 --sample-frac 1.0 --strict-rules-version
```

**Latest full-data validation metrics (v2.1 labels, ~276k val rows):**

| Model | Accuracy | Macro-F1 | High recall | High precision |
|-------|----------|----------|-------------|----------------|
| XGBoost (calibrated) | 91.4% | 0.897 | 85.9% | 84.9% |
| sklearn GBT | 90.7% | 0.888 | 85.0% | 83.6% |

**Deploy artifacts in Git/Docker:** `Models/gbt_sklearn.pkl` and both prediction CSVs under `Data/Model_Data/`.  
**Local only:** full scored CSV (~400MB), enriched/raw CMS data, and `Models/xgb_calibrated.pkl` (optional; not required for Render).

### 5. Run the app

```bash
streamlit run Outputs/Reports/streamlit_app.py
```

Open http://localhost:8501

---

## Risk rules (v2.1)

Rules are **additive** — multiple signals can fire. Each adds points; **review category** is binary (option B):

| Category | Points | Meaning |
|----------|--------|---------|
| **Low** | 0–1 | Not on the priority review queue |
| **High** | ≥ 2 | Needs review (covers former Medium 2–3 and High 4+ tiers) |

Point tiers from v2 scoring (for interpreting severity): High signal strength ≥ 4 pts, medium 2–3, low 0–1 — see [docs/RISK_RULES.md](docs/RISK_RULES.md).

Scored CSV includes: `risk_points`, `rules_fired`, `rules_version` (`2.1.0`), `fraud_risk_category`.

**Example distribution** (full scored file, ~1.38M rows): Low ~976k, High ~405k.

Full rule table and training workflow: **[docs/RISK_RULES.md](docs/RISK_RULES.md)**

---

## Models

| Script | Model | Output |
|--------|-------|--------|
| `Models/train_xgb.py` | Calibrated XGBoost | `Models/xgb_calibrated.pkl`, `Data/Model_Data/fraud_detection_xgb_predictions.csv` |
| `Models/train_sklearn.py` | sklearn Gradient Boosting | `Models/gbt_sklearn.pkl`, `Data/Model_Data/fraud_detection_gbt_sklearn_predictions.csv` |

Training features: `risk_rules.ML_FEATURE_COLS` (7 columns) — see [docs/LABEL_LEAKAGE.md](docs/LABEL_LEAKAGE.md).  
Training **rejects** legacy `Medium` labels; re-run `score` if the scored CSV is not v2.1.

---

## Deploy online

### Docker (local)

```bash
docker build -t prescriber-risk-app .
docker run --rm -p 8501:8501 prescriber-risk-app
```

Open **http://localhost:8501** in your browser (not `http://0.0.0.0:8501` — that address is for the container only).

If port 8501 is busy (e.g. local Streamlit), map another host port:

```bash
docker run --rm -p 8502:8501 prescriber-risk-app
# → http://localhost:8502
```

The image includes `Models/gbt_sklearn.pkl` and `Data/Model_Data/` prediction CSVs from Git.

### Render

**Live:** https://fraud-detection-app-9pen.onrender.com/

Connect the GitHub repo; `render.yaml` uses the root `Dockerfile`. Env vars (also in `Dockerfile`):

- `BASE_DIR=/app`
- `MODEL_DATA_DIR=/app/Data/Model_Data`
- `SKLEARN_MODEL_PATH=/app/Models/gbt_sklearn.pkl`

**Free tier notes:** instances spin down after ~15 min idle; cold starts are slow. Upgrade plan or use a keep-alive ping for demos. Explore tab loads full prediction CSVs — consider sampling for faster UX in a later release.

> Ship **models + demo artifacts** in Git/Docker, not 18 GB of raw CMS data.

---

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/DATA.md](docs/DATA.md) | CMS download links, file sizes, safe CSV preview |
| [docs/RISK_RULES.md](docs/RISK_RULES.md) | Rule definitions, versioning, retrain steps |
| [docs/LABEL_LEAKAGE.md](docs/LABEL_LEAKAGE.md) | Which columns must not be ML features |
| [docs/SPARK.md](docs/SPARK.md) | PySpark version notes |
| [docs/WORK_PLAN.md](docs/WORK_PLAN.md) | Implementation checklist |

---

## Current limitations

- **No fraud ground truth** — labels are heuristics; frame as risk prioritization  
- **Large local data** — raw CMS files and full scored CSV stay on disk, not in Git  
- **Streamlit prototype** — no auth, API, or case-management workflow yet  
- **Render free tier** — cold starts and slow Explore tab (large prediction CSVs); Single Prediction uses sklearn bundle in Docker  

---

## Roadmap (product direction)

1. Faster Explore on Render (sample rows / pre-aggregated stats instead of full CSV load)  
2. Optional: commit `xgb_calibrated.pkl` for live XGB Single-tab on Render  
3. FastAPI + Postgres for NPI lookup and score history  
4. Auth and analyst queue (filter, export, assign)  
5. Similar-case search (embeddings) and optional LLM case summaries  
6. OIG / exclusion list for external validation  

---

## Ethics and privacy

Public CMS data may include provider names and locations. Do not present model output as proof of fraud. For public demos, consider NPI-only views and clear disclaimers.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Author

Developed as part of **LTI Mindtree internship** work (healthcare analytics / M.Tech review).

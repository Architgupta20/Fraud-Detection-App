# Healthcare Prescriber Fraud Risk Analytics

End-to-end analytics pipeline that combines **CMS Open Payments** and **Medicare Part D prescriber** data to score prescribers for potential fraud or abuse risk, train machine learning models, and serve results through a **Streamlit** demo (optional **Docker** deployment).

> **Important:** This project currently uses **rule-based risk labels**, not confirmed fraud outcomes. Treat outputs as **risk prioritization / anomaly screening**, not legal findings of fraud, until validated against external ground truth (e.g. OIG exclusions, DOJ settlements, payer SIU cases).

---

## Table of contents

- [Problem statement](#problem-statement)
- [Data sources](#data-sources)
- [Data guide](#data-guide)
- [Repository layout](#repository-layout)
- [Pipeline overview](#pipeline-overview)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running the pipeline](#running-the-pipeline)
- [Models & outputs](#models--outputs)
- [Streamlit demo](#streamlit-demo)
- [Docker](#docker)
- [Known limitations & remediation roadmap](#known-limitations--remediation-roadmap)
- [Ethics & privacy](#ethics--privacy)
- [License](#license)

---

## Problem statement

Identify prescribers who exhibit patterns associated with **suspicious financial relationships** (industry payments) and **high-risk prescribing** (e.g. opioids), then prioritize them for human review.

The workflow:

1. Ingest and clean public CMS datasets.
2. Build **prescriber-level** features (payments + claims).
3. Assign **Low / Medium / High** risk tiers using transparent rules.
4. Train **GBT / Random Forest** models (Spark ML + sklearn) to predict those tiers.
5. Explore results and demo predictions in Streamlit.

---

## Data sources

| Dataset | File (local) | Description |
|---------|----------------|-------------|
| Medicare Part D Prescriber | `Data/part_d_prescribers.csv` | Prescribing volume, costs, opioids, demographics |
| CMS Open Payments | `Data/open_payments.csv` | Payments from manufacturers/GPOs to providers |

**Scale (approximate):**

| Artifact | Rows |
|----------|------|
| Prescriber-level tables | ~1.38M |
| Payment-level merge | ~14.6M |
| On-disk `Data/` | ~18 GB |

Raw and processed CSVs are **not committed** to Git (see [.gitignore](.gitignore)). Place data locally under `Data/` or document download instructions in `docs/DATA.md` (recommended before publishing).

---

## Data guide

For direct CMS links, expected filenames, and local size guidance, see [`docs/DATA.md`](docs/DATA.md).

---

## Repository layout

```
Project/
├── Data/                    # Raw & processed CSVs (local only)
│   └── Model_Data/          # Predictions & confusion matrices (small; may be committed)
├── Scripts/                 # PySpark ETL, feature engineering, rule scoring
├── Models/                  # Training scripts + saved models
├── Outputs/
│   ├── EDA/                 # Exploratory analysis (Spark + matplotlib)
│   └── Reports/             # Streamlit app, Dockerfile, requirements
└── README.md
```

---

## Pipeline overview

```
part_d_prescribers.csv ──┐
                         ├──► clean ──► prescriber_level_dataset.csv
open_payments.csv ───────┘              │
                                        ▼
                              feature_engineering
                                        │
                                        ▼
                         prescriber_level_enriched.csv
                                        │
                                        ▼
                         fraud_risk_scoring (rules)
                                        │
                                        ▼
                         fraud_risk_scored_prescribers.csv
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
           Spark ML (GBT / RF)                    sklearn GBT
                    │                                       │
                    ▼                                       ▼
         Data/Model_Data/*_predictions.csv      gbt_sklearn.pkl
```

### Rule-based labels (`Scripts/fraud_risk_scoring.py`)

A prescriber receives a higher score when any of these hold (first matching rule wins in the `when` chain):

| Condition | Score |
|-----------|-------|
| `payment_to_drug_cost_ratio > 1` | 2 |
| `opioid_claims > 100` | 2 |
| `high_payment_flag == 1` | 2 |
| `high_opioid_flag == 1` | 2 |
| `peer_deviation_score > 5` | 2 |
| `elderly_focus_flag == 1` | 1 |
| Otherwise | 0 |

Mapped to categories: **High** (≥2), **Medium** (=1), **Low** (=0).

### Engineered features (representative)

`payment_per_claim`, `payment_to_drug_cost_ratio`, `opioid_claim_ratio`, `high_opioid_flag`, `high_payment_flag`, `payment_variability`, `peer_deviation_score`, `elderly_focus_flag`, `adjusted_risk_payment`, etc.

---

## Prerequisites

- **Python 3.10+** (3.12 used in development)
- **Java 17** (required for PySpark)
- **Apache Spark** via `pyspark` (local mode)
- **8 GB+ RAM** recommended for full datasets; use sampling for laptops

Optional: Docker for containerized Streamlit.

---

## Setup

```bash
cd "/path/to/LTI Internship/LTI/Project"

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Project paths resolve via `config.py` (repo root). Override if needed:

```bash
export BASE_DIR="$(pwd)"
```

Run the full ETL (when raw CMS files are under `Data/`):

```bash
python run_pipeline.py all
# or stage by stage: clean | aggregate | features | score
```

Spark saved models vs PySpark pin: see [docs/SPARK.md](docs/SPARK.md).  
Leakage controls and excluded rule-input features: [docs/LABEL_LEAKAGE.md](docs/LABEL_LEAKAGE.md).

---

## Running the pipeline

| Step | Command / script | Primary output |
|------|------------------|----------------|
| 1–6 | `python run_pipeline.py all` | `fraud_risk_scored_prescribers.csv` |
| — | `Scripts/load_*.py` | Schema exploration only |
| 7 | `Models/gbt_removing_leakage.py` (etc.) | `Data/Model_Data/*.csv` |
| 8 | `Models/train_sklearn.py` | `Models/gbt_sklearn.pkl` |

Example (sklearn, sampled):

```bash
export BASE_DIR="$(pwd)"
python Models/train_sklearn.py --sample-frac 0.2
```

Example (Spark fraud scoring):

```bash
spark-submit Scripts/fraud_risk_scoring.py
```

---

## Models & outputs

| Script | Model | Notes |
|--------|-------|-------|
| `Models/gbt_removing_leakage.py` | Spark GBT + One-vs-Rest | Excludes all rule-input features; reports macro-F1 + per-class metrics |
| `Models/gbt_tune_safe.py` | Spark GBT + TVS + resampling | Hyperparameter tuning; combined over/under-sample |
| `Models/rf_removing_leakage.py` | Spark Random Forest | Excludes all rule-input features; reports macro-F1 + per-class metrics |
| `Models/rf_tune_safe.py` | Spark RF + TVS + resampling | |
| `Models/train_sklearn.py` | sklearn `GradientBoostingClassifier` | Lightweight deployable model |

**Artifacts in `Data/Model_Data/`:**

- `fraud_detection_gbt_predictions.csv`
- `fraud_detection_gbt_combined_predictions.csv`
- `fraud_detection_rf_predictions.csv`
- `fraud_detection_rf_combined_predictions.csv`
- `fraud_detection_gbt_sklearn_predictions.csv`
- `confusion_matrix_*.png`

**Saved Spark pipeline:** `Models/spark_pipeline_model/` (used by Streamlit when Spark is enabled).

---

## Streamlit demo

```bash
export BASE_DIR="$(pwd)"
export MODEL_DATA_DIR="$BASE_DIR/Data/Model_Data"
streamlit run Outputs/Reports/streamlit_app.py
```

Defaults (Phase 4):

- **sklearn model** (`Models/gbt_sklearn.pkl`) when available — recommended for deploy
- CSV fallback includes `fraud_detection_gbt_sklearn_predictions.csv`
- Spark model is optional (heavy)
- Single prediction shows **rule signals** and **top model features** (“why flagged”)

Features:

- Single prescriber lookup
- Batch CSV upload
- Optional Spark model load (heavy) or CSV prediction fallback

---

## Docker

From project root (ensure `BASE_DIR` paths inside container match `/app`):

```bash
docker build -f Outputs/Reports/Dockerfile -t fraud-risk-app .
docker run -p 8501:8501 fraud-risk-app
```

Open http://localhost:8501

> **Warning:** A naive `docker build` that `COPY . /app` will bundle **all local CSVs** (~18 GB). Use a `.dockerignore` excluding `Data/` except `Model_Data/`, or mount volumes at runtime.

---

## Known limitations & remediation roadmap

### Critical issues

| Issue | Impact | Recommended fix |
|-------|--------|-----------------|
| **Circular / target leakage** | Rules use `high_payment_flag`, `high_opioid_flag`, `payment_to_drug_cost_ratio`, `opioid_claims`, `peer_deviation_score`, `elderly_focus_flag`; many of these are also model features. ML mostly **re-learns the rules**, so accuracy is misleading. | Remove **all** label-defining columns from `feature_cols`. Or change task to unsupervised anomaly detection. |
| **No fraud ground truth** | Labels are heuristic, not confirmed fraud. | Frame as *risk scoring*; add OIG LEIE / public enforcement cases for evaluation. |
| **Broken `train_sklearn.py` tail** | Lines after `main()` reference undefined `clf` → **ImportError** if the file is imported. | Delete or move evaluation code inside `main()`. |
| **`gbt_tune_safe.py` feature list bug** | When `USE_ELDERLY_FLAG=False`, `feature_cols = base_features + base_features` (duplicated). | Use `feature_cols = base_features + (["elderly_focus_flag"] if USE_ELDERLY_FLAG else [])`. |
| **Hard-coded paths** | Scripts point to `~/Desktop/LTI/Project` instead of current folder. | Central `config.py` with `BASE_DIR = os.getenv("BASE_DIR", Path(__file__).resolve().parents[1])`. |
| **Broken `.gitignore`** | Was a shell snippet, not ignore rules. | Fixed in repo; verify before `git add`. |
| **Spark version mismatch** | Saved model metadata shows Spark **4.0**; `requirements.txt` pins **3.4.1**. | Align versions and retrain, or ship sklearn-only in production. |

### High priority

| Issue | Fix |
|-------|-----|
| ETL logic mostly **commented out** | Extract runnable `run_etl.py` with CLI stages |
| `output.py` references missing `merged_important_dataset.csv` | Remove or implement |
| **PII in CSVs** (names) | Drop or hash `first_name`/`last_name` before public GitHub |
| **18 GB data** in repo risk | Keep `Data/` ignored; add `docs/DATA.md` with CMS download links |
| Streamlit ignores sklearn predictions | Add `fraud_detection_gbt_sklearn_predictions.csv` to fallback list; default to `joblib` model |
| Docker copies entire tree | Add `.dockerignore` |
| Class imbalance (Medium ~55%) | Report macro-F1, PR-AUC; tune thresholds on validation set |
| Sequential `when` in rules | Document that only first rule fires; consider `greatest()` of scores |

### Medium priority

| Issue | Fix |
|-------|-----|
| No `requirements.txt` at project root | Add consolidated deps + optional `requirements-spark.txt` |
| No tests | Add smoke tests for schema, row counts, label distribution |
| Empty `Outputs/*/artifacts/` | Save EDA plots to disk in `visuals.py` |
| Arbitrary thresholds (100 opioids, ratio > 1) | Calibrate on labeled subset or percentile-based flags |
| `train_sklearn` predicts on full data for export | Save separate `train`/`test` prediction files or only score holdout for metrics |

### Suggested refactor order

1. Fix paths + `config.py` + broken scripts (`train_sklearn`, `gbt_tune_safe`).
2. Redefine features to **exclude label leakage**; retrain and document new metrics.
3. Make ETL reproducible (uncomment → modules → CLI).
4. Add `.dockerignore`, `docs/DATA.md`, root `requirements.txt`.
5. Initialize Git; commit code + small `Model_Data/` only.
6. (Optional) Replace rule labels with external fraud labels or semi-supervised approach.

---

## Ethics & privacy

- Data are **public CMS files** but may contain provider names and locations.
- Do **not** present model output as proof of fraud without investigation.
- For open-source release: remove or pseudonymize direct identifiers; document intended use (research / prioritization only).

---

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).

---

## Author & context

Developed as part of **LTI Mindtree internship** work (healthcare analytics / M.Tech review).

For questions about pushing to GitHub, coordinate repo name, whether to include `Model_Data/`, and handling of local `Data/` separately from this codebase.

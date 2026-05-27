# Work plan — implementation checklist

Use this file as the **single source of truth** for fixes and improvements.  
Keep **this chat** for questions, explanations, and progress notes.  
Use a **new Cursor chat** (or Composer) with `@docs/WORK_PLAN.md` when you want to implement items.

**Last updated:** 2026-05-28  
**Project root:** `/Users/architgupta280/Desktop/LTI Internship/LTI/Project`

---

## How to work in two places

| Where | Purpose |
|-------|---------|
| **This chat** | Ask “what does X mean?”, examples, thesis framing, review decisions |
| **New chat + `@docs/WORK_PLAN.md`** | “Do Phase 1 item 3”, code changes, commits (when you say to push) |

**Prompt to paste in a new implementation chat:**

```text
Read @docs/WORK_PLAN.md and @README.md.
Start with Phase 1 (unchecked items only).
Use BASE_DIR = project root. Do not push to GitHub unless I ask.
After each phase, update checkboxes in WORK_PLAN.md and summarize what changed.
```

---

## Project framing (do not forget)

- **What it is:** Prescriber risk scoring from CMS Part D + Open Payments (rules + ML + Streamlit).
- **What it is not (yet):** Proven fraud detection without external labels (OIG, investigations).
- **Main scientific issue:** Rules create labels; many rule inputs are also ML features → model mostly copies rules.

---

## Phase 0 — Framing & success criteria

- [ ] One-sentence project definition agreed (thesis / PPT / GitHub)
- [ ] Use wording: **“risk scoring / audit prioritization”** not “confirmed fraud”
- [ ] Define “done”: reproducible pipeline, honest holdout metrics, Streamlit runs, no label leakage

---

## Phase 1 — Make it runnable (do first)

- [x] Add `config.py` with `BASE_DIR` from env or repo root
- [x] Replace all `/Users/.../Desktop/LTI/Project` paths (Scripts, Models, Outputs/EDA)
- [x] Fix `Models/train_sklearn.py` — remove module-level code after `main()` (lines ~181+)
- [x] Fix `Models/gbt_tune_safe.py` — `feature_cols = base_features + ([] or ["elderly_focus_flag"])`
- [x] Fix or remove `Scripts/output.py` (missing `merged_important_dataset.csv`)
- [x] Refactor ETL: uncomment logic → `run_pipeline.py` with stages (`clean`, `aggregate`, `features`, `score`)
- [x] Root `requirements.txt` (streamlit, pyspark, sklearn, pandas, joblib, matplotlib)
- [x] Align Spark version with saved `Models/spark_pipeline_model/` OR document sklearn-only path

**Done already:**

- [x] `README.md` written
- [x] `.gitignore` fixed (was invalid shell snippet)
- [x] `Data/Model_Data/.gitkeep`

---

## Phase 2 — Science & labels (credibility)

- [ ] Table: each rule in `fraud_risk_scoring.py` → columns that must **not** be features
- [ ] Retrain Spark + sklearn **without** leaky columns (`high_payment_flag`, `high_opioid_flag`, etc. if used in rules)
- [ ] Re-evaluate on holdout (keep hash split by `prescriber_id`)
- [ ] Report macro-F1, per-class precision/recall (expect metrics may drop — that is OK)
- [ ] Optional: join OIG LEIE (or similar) for real-label evaluation
- [ ] Optional: unsupervised anomaly model (Isolation Forest) for “patterns rules missed”
- [ ] Calibrate rule thresholds (percentiles / SME) instead of magic numbers only

---

## Phase 3 — GitHub readiness

- [ ] `docs/DATA.md` — CMS download links, file names, sizes
- [ ] Confirm `Data/*` stays ignored (~18 GB)
- [ ] Decide: commit `Data/Model_Data/` predictions or not
- [ ] Strip/hash `first_name`, `last_name` before public repo
- [ ] Add `.dockerignore` (exclude raw `Data/`)
- [ ] `git init` + first commit when user provides remote URL
- [ ] Add `LICENSE`

---

## Phase 4 — App & deployment

- [ ] Streamlit: default to `gbt_sklearn.pkl` (lightweight)
- [ ] Add `fraud_detection_gbt_sklearn_predictions.csv` to fallback list
- [ ] UI: show why flagged (rules fired + top features)
- [ ] Test Docker build without bundling full `Data/`

---

## Phase 5 — Docs & presentation

- [ ] Sync `MTECH_PPT_Review_3.pptx` with honest framing + architecture diagram
- [ ] Limitations slide: no ground truth, leakage fixed, public data only
- [ ] Optional: smoke tests, save EDA plots to `Outputs/EDA/artifacts/`

---

## Known bugs (quick reference)

| File | Issue |
|------|--------|
| `train_sklearn.py` | `clf` used at import time after `main` |
| `gbt_tune_safe.py` | Duplicates `feature_cols` when `USE_ELDERLY_FLAG=False` |
| All scripts | Wrong `BASE_DIR` path |
| `fraud_risk_scoring.py` | Sequential `when` — only first rule applies |
| Spark model metadata | Spark 4.0 saved vs pyspark 3.4.1 in requirements |

---

## Label leakage map (for Phase 2)

Rules in `Scripts/fraud_risk_scoring.py` use:

| Rule input | Also used as ML feature today? | Action |
|------------|-------------------------------|--------|
| `payment_to_drug_cost_ratio` | Yes | Exclude from features OR remove from rules |
| `opioid_claims` | Yes (`opioid_claims`) | Same |
| `high_payment_flag` | Yes | **Exclude** from features |
| `high_opioid_flag` | Yes | **Exclude** from features |
| `peer_deviation_score` | Partially removed in “no leak” scripts | Exclude everywhere |
| `elderly_focus_flag` | Yes in sklearn | Exclude |

---

## Progress log (fill in as you go)

| Date | Phase | What was done |
|------|-------|----------------|
| 2026-05-27 | Docs | README, WORK_PLAN, fixed .gitignore |
| 2026-05-28 | Phase 1 | config.py, run_pipeline.py, path fixes, train_sklearn/gbt_tune_safe bugs, requirements.txt, docs/SPARK.md |
| | | |
| | | |

---

## Questions for this chat only

Use the **original Q&A chat** for:

- “What does column X mean?”
- Examples from raw CSV (same NPI, etc.)
- How to explain to examiner / interviewer
- Whether a change is worth doing before deadline

Implementation details and PRs → **new chat + this file**.

# Work plan

**Project root:** `LTI Internship/LTI/Project`  
**Q&A / examples:** original chat  
**Implementation:** new chat with `@docs/WORK_PLAN.md`

---

## Phase 1 — Make it runnable

- [x] `config.py` + `BASE_DIR` env override
- [x] Scripts/Models/EDA use `config` (active code; old paths only in comments)
- [x] `Models/train_sklearn.py` — fixed (no import-time crash)
- [x] `Models/gbt_tune_safe.py` — fixed feature list duplication
- [x] `Scripts/output.py` — uses `MERGED_PAYMENT_LEVEL_CSV`
- [x] `run_pipeline.py` — stages: `clean`, `aggregate`, `features`, `score`, `all`
- [x] Root `requirements.txt` + `requirements-spark.txt`
- [x] `docs/SPARK.md` — sklearn-first vs Spark 4.0 saved model
- [x] `run_pipeline.py` — lazy PySpark import so `--help` works without pyspark

**How to run**

```bash
cd "/Users/architgupta280/Desktop/LTI Internship/LTI/Project"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # sklearn + streamlit
pip install -r requirements-spark.txt    # + pyspark for ETL

export BASE_DIR="$(pwd)"
python run_pipeline.py --help
python run_pipeline.py score             # re-score only (needs enriched CSV)
python Models/train_sklearn.py --nrows 10000
```

---

## Phase 2 — Science & labels (next)

- [ ] See `docs/LABEL_LEAKAGE.md`
- [ ] Retrain all models with non-leaky features only
- [ ] Report macro-F1 on holdout
- [ ] Optional: OIG / external labels

---

## Phase 3 — GitHub

- [ ] `docs/DATA.md` (exists — verify links)
- [ ] Push when user provides remote URL
- [ ] PII / `.dockerignore` review

---

## Phase 4 — App

- [ ] Streamlit sklearn-first (partially done — verify)
- [ ] Explain flags in UI

---

## Progress log

| Date | Note |
|------|------|
| 2026-05-27 | Phase 1 completed / verified |

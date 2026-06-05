# Work plan (simplified)

## Phase 1 — Done / in progress

- [x] `run_pipeline.py` ETL + `risk_rules.py` v2.1 (binary Low/High)
- [x] Full score + train (`train_xgb.py`, `train_sklearn.py`, `--strict-rules-version`)
- [x] Streamlit: rename, disclaimer, Explore sub-tabs (XGB / sklearn CSV)
- [x] Docker build + local run
- [x] GitHub: app code, `gbt_sklearn.pkl`, prediction CSVs
- [x] **Render deploy:** https://fraud-detection-app-9pen.onrender.com/
- [ ] Live smoke test (Single, Batch, Explore) on Render URL
- [ ] Landing screenshot/GIF in repo
- [ ] Optional: commit `xgb_calibrated.pkl` for live XGB Single-tab on Render

## Phase 2+ (product)

- [ ] FastAPI + Postgres for NPI lookup and score history
- [ ] Auth and analyst queue
- [ ] Slim Explore tab (sample rows / API) for faster Render UX

See [README.md](../README.md) and [docs/RISK_RULES.md](RISK_RULES.md).

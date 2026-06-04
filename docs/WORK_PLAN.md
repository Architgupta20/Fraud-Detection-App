# Work plan (simplified)

## Done

- [x] `run_pipeline.py` ETL + `risk_rules.py` v2
- [x] `Models/train_xgb.py`, `Models/train_sklearn.py`, `Models/ml_common.py`
- [x] Data stage folders under `Data/`
- [x] Streamlit app + Docker + GitHub

## Next (product)

- [ ] `python run_pipeline.py score` after rules v2
- [ ] Full train: `train_xgb.py` + `train_sklearn.py` with `--strict-rules-version`
- [ ] Wire XGB into Streamlit
- [ ] Deploy lean Docker image on Render

See [README.md](../README.md) and [docs/RISK_RULES.md](RISK_RULES.md).

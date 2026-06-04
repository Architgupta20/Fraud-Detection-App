# PySpark (ETL only)

This project uses **PySpark for the data pipeline** (`run_pipeline.py`), not for production ML.

## Install

```bash
pip install -r requirements-spark.txt
```

Requires **Java 17** on your machine.

## IDE import warnings

If VS Code shows `pyspark` as missing, install `requirements-spark.txt` into `.venv` and select `.venv` as the Python interpreter.

## Commands

```bash
export BASE_DIR="$(pwd)"
python run_pipeline.py all
# or: clean | aggregate | features | score
```

Production models: `Models/train_xgb.py` and `Models/train_sklearn.py` (pandas/sklearn, no Spark).

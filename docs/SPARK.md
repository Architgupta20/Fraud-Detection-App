# PySpark (ETL only)

This project uses **PySpark for the data pipeline** (`run_pipeline.py`), not for production ML.

## Install

```bash
pip install -r requirements-spark.txt
```

Requires **Java 17+** on your machine (Java 21 is fine).

Uses **PySpark 3.5.5** (see `requirements-spark.txt`) for **Python 3.13** compatibility. Older PySpark 3.4.x fails on 3.13 with `typing.io` import errors.

## IDE import warnings

If VS Code shows `pyspark` as missing, install `requirements-spark.txt` into `.venv` and select `.venv` as the Python interpreter.

## Commands

```bash
export BASE_DIR="$(pwd)"
python run_pipeline.py all
# or: clean | aggregate | features | score
```

Production models: `Models/train_xgb.py` and `Models/train_sklearn.py` (pandas/sklearn, no Spark).

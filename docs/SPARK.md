# Spark vs sklearn deployment

## Version mismatch

The saved Spark pipeline under `Models/spark_pipeline_model/` was trained with **Spark 4.0.0** (see `metadata/part-00000-*.txt`). This repo pins **PySpark 3.4.1** in `requirements.txt`.

Loading the saved model with PySpark 3.4.x may fail or behave unexpectedly.

## Recommended paths

| Use case | Approach |
|----------|----------|
| **Local demo / laptop** | sklearn only: `python Models/train_sklearn.py` → `Models/gbt_sklearn.pkl` |
| **Streamlit** | Leave “Load Spark model” **unchecked**; use CSV prediction fallback |
| **Retrain Spark models** | Install Spark 4.x + matching PySpark, re-run `Models/gbt_tune_safe.py` (etc.), save a new pipeline |
| **Full ETL** | `python run_pipeline.py all` (requires Java 17 + enough RAM for ~18 GB data) |

## Aligning versions (optional)

1. Install Java 17 and Apache Spark 4.0.x.
2. Update `requirements.txt`: `pyspark==4.0.0` (match your Spark install).
3. Retrain and overwrite `Models/spark_pipeline_model/`.
4. Re-test Streamlit with Spark loading enabled.

Until then, treat **sklearn + precomputed CSVs in `Data/Model_Data/`** as the supported production path.

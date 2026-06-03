#!/usr/bin/env python3
# train_sklearn.py
"""
Train a lightweight sklearn model for inference and produce a predictions CSV
suitable for your Streamlit app fallback.

- Trains GradientBoostingClassifier on selected features.
- Saves model pipeline (scaler + model + feature list) to Models/gbt_sklearn.pkl.
- Writes a predictions CSV aligned to the DataFrame used for prediction to
  Data/Model_Data/fraud_detection_gbt_sklearn_predictions.csv.

Usage examples:
    python train_sklearn.py
    python train_sklearn.py --sample-frac 0.5
    python train_sklearn.py --nrows 50000
"""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support

from config import FRAUD_RISK_SCORED_CSV, GBT_SKLEARN_PKL, model_data_path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ml_common import FEATURE_COLS, INV_LABEL_MAP, holdout_split, load_and_preprocess

INPUT_CSV = str(FRAUD_RISK_SCORED_CSV)
OUT_MODEL = str(GBT_SKLEARN_PKL)
OUT_PRED_CSV = str(model_data_path("fraud_detection_gbt_sklearn_predictions.csv"))
def main(args):
    df = load_and_preprocess(
        INPUT_CSV,
        nrows=args.nrows,
        sample_frac=args.sample_frac,
        random_state=args.random_state,
        strict_rules_version=args.strict_rules_version,
    )
    if df.empty:
        raise RuntimeError("No data after loading/preprocessing. Check CSV and parameters.")

    print(f"Data rows after preprocessing: {len(df)}")
    unique_labels = np.unique(df["label_num"].values)
    if len(unique_labels) < 2:
        raise RuntimeError(f"Need at least two classes to train. Found labels: {unique_labels}")

    train_df, val_df, X_train, y_train, X_val, y_val = holdout_split(df, test_size=args.test_size)
    print(f"Train rows: {len(X_train)}, Val rows: {len(X_val)}")

    # Scale
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)

    # Train model
    print("Training GradientBoostingClassifier...")
    clf = GradientBoostingClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        random_state=args.random_state
    )
    clf.fit(X_train_s, y_train)

    # Validate
    val_pred = clf.predict(X_val_s)
    acc = (val_pred == y_val).mean()
    macro_f1 = f1_score(y_val, val_pred, average="macro")
    per_p, per_r, per_f1, _ = precision_recall_fscore_support(
        y_val, val_pred, labels=[0, 1, 2], zero_division=0
    )
    print(f"Validation accuracy: {acc:.4f}")
    print(f"Validation macro-F1: {macro_f1:.4f}")
    print("Per-class metrics:")
    for lbl, p, r, f in zip(["Low", "Medium", "High"], per_p, per_r, per_f1):
        print(f"  {lbl:<6} Precision={p:.3f} Recall={r:.3f} F1={f:.3f}")
    print("\nDetailed classification report:")
    print(classification_report(y_val, val_pred, target_names=["Low", "Medium", "High"]))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_val, val_pred))

    # Save model object (scaler + model + feature list)
    model_obj = {"scaler": scaler, "model": clf, "feature_cols": FEATURE_COLS}
    os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
    joblib.dump(model_obj, OUT_MODEL)
    print("Saved sklearn model to:", OUT_MODEL)

    # -------------------------
    # Build predictions CSV aligned to the dataframe `df` used above
    # -------------------------
    print("Preparing predictions CSV (aligned to the processed dataframe)...")
    df_pred = df.reset_index(drop=True)  # aligned index 0..N-1

    X_full = df_pred[FEATURE_COLS].values
    print(f"DEBUG: df_pred rows = {df_pred.shape[0]}, X_full shape = {X_full.shape}")

    # Transform & predict (may raise if scaler dims mismatch)
    X_full_scaled = scaler.transform(X_full)
    probs = clf.predict_proba(X_full_scaled)
    preds = clf.predict(X_full_scaled)

    # sanity checks
    if len(preds) != len(df_pred):
        raise ValueError(f"Length mismatch: preds ({len(preds)}) != df_pred ({len(df_pred)}). Aborting.")
    if probs.shape[0] != len(df_pred):
        raise ValueError(f"Length mismatch: probs rows ({probs.shape[0]}) != df_pred ({len(df_pred)}). Aborting.")

    # prescriber id column if exists, else use index
    prescriber_series = df_pred.get("prescriber_id")
    if prescriber_series is None:
        prescriber_series = pd.Series(range(len(df_pred)), name="prescriber_id")
    else:
        prescriber_series = prescriber_series.astype(str).reset_index(drop=True)

    out_df = pd.DataFrame({
        "prescriber_id": prescriber_series,
        "prediction": preds,
        "predicted_category": pd.Series(preds).map(INV_LABEL_MAP),
        "p_low": probs[:, 0],
        "p_medium": probs[:, 1],
        "p_high": probs[:, 2]
    })

    os.makedirs(os.path.dirname(OUT_PRED_CSV), exist_ok=True)
    out_df.to_csv(OUT_PRED_CSV, index=False)
    print("Saved predictions CSV to:", OUT_PRED_CSV)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sklearn model for fraud detection and save predictions CSV.")
    parser.add_argument("--sample-frac", type=float, default=0.2,
                        help="Fraction of data to sample for training/prediction (0-1). Use 1.0 to use full file (may OOM). Default=0.2")
    parser.add_argument("--nrows", type=int, default=None, help="If set, only read this many rows from CSV (overrides sample_frac).")
    parser.add_argument("--n_estimators", type=int, default=100, help="Gradient boosting n_estimators")
    parser.add_argument("--max_depth", type=int, default=4, help="Gradient boosting max_depth")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Gradient boosting learning rate")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--strict-rules-version",
        action="store_true",
        help="Fail if scored CSV rules_version != risk_rules.RULES_VERSION",
    )
    args = parser.parse_args()

    main(args)


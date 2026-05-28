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
import hashlib
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support

from config import FRAUD_RISK_SCORED_CSV, GBT_SKLEARN_PKL, model_data_path

# ---------------------------
# Configurable paths & cols
# ---------------------------
INPUT_CSV = str(FRAUD_RISK_SCORED_CSV)
OUT_MODEL = str(GBT_SKLEARN_PKL)
OUT_PRED_CSV = str(model_data_path("fraud_detection_gbt_sklearn_predictions.csv"))

# Exclude all rule-input columns to reduce label leakage from rule-based targets.
FEATURE_COLS = [
    "total_claims",
    "total_drug_cost",
    "opioid_cost",
    "antibiotic_claims",
    "avg_risk_score",
    "payment_variability",
    "adjusted_risk_payment",
]

LABEL_COL = "fraud_risk_category"  # expected values: "Low","Medium","High"

# ---------------------------
# Helpers
# ---------------------------
def map_label(series):
    return series.map({"Low": 0, "Medium": 1, "High": 2})


def stable_prescriber_bucket(value, modulo=10000):
    text = str(value if pd.notnull(value) else "")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo

def load_and_preprocess(input_csv, nrows=None, sample_frac=None, random_state=42):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    print(f"Loading CSV: {input_csv}  (nrows={nrows}, sample_frac={sample_frac})")
    df = pd.read_csv(input_csv, nrows=nrows)
    if sample_frac is not None and 0.0 < sample_frac < 1.0:
        print(f"Sampling fraction {sample_frac} of {len(df)} rows...")
        df = df.sample(frac=sample_frac, random_state=random_state)
    # Drop rows missing the label
    df = df[df[LABEL_COL].notnull()].copy()
    # Ensure feature cols exist and are numeric
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    # Map label
    df["label_num"] = map_label(df[LABEL_COL]).fillna(0).astype(int)
    return df

# ---------------------------
# Main
# ---------------------------
def main(args):
    # Load / preprocess
    df = load_and_preprocess(INPUT_CSV, nrows=args.nrows, sample_frac=args.sample_frac, random_state=args.random_state)
    if df.empty:
        raise RuntimeError("No data after loading/preprocessing. Check CSV and parameters.")

    print(f"Data rows after preprocessing: {len(df)}")
    # Features / labels
    X = df[FEATURE_COLS].values
    y = df["label_num"].values

    # If labels have only one class, training won't work. Check:
    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        raise RuntimeError(f"Need at least two classes to train. Found labels: {unique_labels}")

    # Holdout split by prescriber hash to avoid leakage across train/validation.
    split_series = df.get("prescriber_id", pd.Series(range(len(df))))
    split_bucket = split_series.apply(stable_prescriber_bucket)
    val_mask = (split_bucket / 10000.0) >= (1.0 - args.test_size)
    if val_mask.all() or (~val_mask).all():
        raise RuntimeError("Holdout split failed: adjust test_size or check prescriber_id values.")
    train_mask = ~val_mask

    X_train, X_val = X[train_mask.values], X[val_mask.values]
    y_train, y_val = y[train_mask.values], y[val_mask.values]

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
        "predicted_category": pd.Series(preds).map({0: "Low", 1: "Medium", 2: "High"}),
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
    args = parser.parse_args()

    main(args)


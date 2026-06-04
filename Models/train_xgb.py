#!/usr/bin/env python3
"""
Train calibrated XGBoost for prescriber risk tiers.

Uses feature list and labels from risk_rules.py via ml_common.
Saves bundle to Models/xgb_calibrated.pkl and predictions CSV.

Usage:
    pip install xgboost
    python Models/train_xgb.py --sample-frac 0.2
    python Models/train_xgb.py --nrows 50000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import FRAUD_RISK_SCORED_CSV, XGB_CALIBRATED_PKL, model_data_path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ml_common import (
    CLASS_NAMES,
    FEATURE_COLS,
    INV_LABEL_MAP,
    NUM_CLASSES,
    holdout_split,
    load_and_preprocess,
)

INPUT_CSV = str(FRAUD_RISK_SCORED_CSV)
OUT_MODEL = str(XGB_CALIBRATED_PKL)
OUT_PRED_CSV = str(model_data_path("fraud_detection_xgb_predictions.csv"))


def print_metrics(y_true, y_pred, title: str) -> float:
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    label_ids = list(range(NUM_CLASSES))
    per_p, per_r, per_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, zero_division=0
    )
    print(f"\n===== {title} =====")
    print(f"Accuracy: {(y_pred == y_true).mean():.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("Per-class metrics:")
    for lbl, p, r, f in zip(CLASS_NAMES, per_p, per_r, per_f1):
        print(f"  {lbl:<6} Precision={p:.3f} Recall={r:.3f} F1={f:.3f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    return macro_f1


def main(args: argparse.Namespace) -> None:
    df = load_and_preprocess(
        INPUT_CSV,
        nrows=args.nrows,
        sample_frac=args.sample_frac,
        random_state=args.random_state,
        strict_rules_version=args.strict_rules_version,
    )
    if df.empty:
        raise RuntimeError("No data after loading/preprocessing.")

    unique_labels = np.unique(df["label_num"].values)
    if len(unique_labels) < 2:
        raise RuntimeError(f"Need at least two classes to train. Found: {unique_labels}")

    train_df, val_df, X_train, y_train, X_val, y_val = holdout_split(df, test_size=args.test_size)
    print(f"Train rows: {len(train_df)}, Val rows: {len(val_df)}")

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)

    base = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic" if NUM_CLASSES == 2 else "multi:softprob",
        eval_metric="logloss",
        random_state=args.random_state,
        n_jobs=-1,
    )

    if args.calibrate:
        print("Training XGBoost with probability calibration (sigmoid)...")
        model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        model.fit(X_train_s, y_train)
    else:
        print("Training XGBoost (no calibration)...")
        model = base
        model.fit(X_train_s, y_train)

    val_pred = model.predict(X_val_s)
    print_metrics(y_val, val_pred, "VALIDATION")

    bundle = {
        "scaler": scaler,
        "model": model,
        "feature_cols": FEATURE_COLS,
        "calibrated": args.calibrate,
        "label_map": INV_LABEL_MAP,
        "num_classes": NUM_CLASSES,
    }
    os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
    joblib.dump(bundle, OUT_MODEL)
    print(f"\nSaved model bundle: {OUT_MODEL}")

    # Predictions on full processed dataframe
    X_full = scaler.transform(df[FEATURE_COLS].values)
    probs = model.predict_proba(X_full)
    preds = model.predict(X_full)

    prescriber_series = df.get("prescriber_id")
    if prescriber_series is None:
        prescriber_series = pd.Series(range(len(df)), name="prescriber_id")
    else:
        prescriber_series = prescriber_series.astype(str).reset_index(drop=True)

    out_cols = {
        "prescriber_id": prescriber_series,
        "prediction": preds.astype(int),
        "predicted_category": pd.Series(preds).map(INV_LABEL_MAP),
    }
    for i, name in enumerate(CLASS_NAMES):
        out_cols[f"p_{name.lower()}"] = probs[:, i]
    out_df = pd.DataFrame(out_cols)
    os.makedirs(os.path.dirname(OUT_PRED_CSV), exist_ok=True)
    out_df.to_csv(OUT_PRED_CSV, index=False)
    print(f"Saved predictions: {OUT_PRED_CSV}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train calibrated XGBoost risk model.")
    parser.add_argument("--sample-frac", type=float, default=0.2)
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--calibrate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wrap XGBoost with CalibratedClassifierCV (default: on)",
    )
    parser.add_argument(
        "--strict-rules-version",
        action="store_true",
        help="Fail if scored CSV rules_version != risk_rules.RULES_VERSION",
    )
    main(parser.parse_args())

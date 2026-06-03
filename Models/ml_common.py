"""Shared helpers for sklearn/XGBoost training — imports config from risk_rules.py."""

from __future__ import annotations

import hashlib
import os
from typing import Tuple

import numpy as np
import pandas as pd

from risk_rules import (
    INV_LABEL_MAP,
    LABEL_COL,
    LABEL_MAP,
    ML_FEATURE_COLS,
    RULES_VERSION,
    check_scored_rules_version,
)

# Re-export for train scripts (single source: risk_rules.py)
FEATURE_COLS = ML_FEATURE_COLS


def map_label(series: pd.Series) -> pd.Series:
    return series.map(LABEL_MAP)


def stable_prescriber_bucket(value, modulo: int = 10000) -> int:
    text = str(value if pd.notnull(value) else "")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def load_and_preprocess(
    input_csv: str,
    nrows: int | None = None,
    sample_frac: float | None = None,
    random_state: int = 42,
    *,
    strict_rules_version: bool = False,
) -> pd.DataFrame:
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    print(f"Training with risk_rules.RULES_VERSION = {RULES_VERSION}")
    print(f"Loading CSV: {input_csv}  (nrows={nrows}, sample_frac={sample_frac})")
    df = pd.read_csv(input_csv, nrows=nrows)
    if "rules_version" in df.columns:
        versions = df["rules_version"].dropna().astype(str).unique()
        if len(versions) == 1:
            check_scored_rules_version(versions[0], strict=strict_rules_version)
        elif len(versions) > 1:
            print(f"WARNING: multiple rules_version values in sample: {versions}")
    else:
        check_scored_rules_version(None)
    if sample_frac is not None and 0.0 < sample_frac < 1.0:
        print(f"Sampling fraction {sample_frac} of {len(df)} rows...")
        df = df.sample(frac=sample_frac, random_state=random_state)
    df = df[df[LABEL_COL].notnull()].copy()
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df["label_num"] = map_label(df[LABEL_COL]).fillna(0).astype(int)
    return df


def holdout_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split by hashed prescriber_id into train / validation frames."""
    split_series = df.get("prescriber_id", pd.Series(range(len(df))))
    split_bucket = split_series.apply(stable_prescriber_bucket)
    val_mask = (split_bucket / 10000.0) >= (1.0 - test_size)
    if val_mask.all() or (~val_mask).all():
        raise RuntimeError("Holdout split failed: adjust test_size or check prescriber_id values.")
    train_df = df.loc[~val_mask].copy()
    val_df = df.loc[val_mask].copy()
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["label_num"].values
    X_val = val_df[FEATURE_COLS].values
    y_val = val_df["label_num"].values
    return train_df, val_df, X_train, y_train, X_val, y_val

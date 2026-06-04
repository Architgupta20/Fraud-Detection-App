#!/usr/bin/env python3
"""Preview large CSVs without loading the full file into memory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from config import (
    CLEAN_PAYMENTS_CSV,
    CLEAN_PRESCRIBERS_CSV,
    FRAUD_RISK_SCORED_CSV,
    OPEN_PAYMENTS_CSV,
    PART_D_PRESCRIBERS_CSV,
    PRESCRIBER_LEVEL_CSV,
    PRESCRIBER_LEVEL_ENRICHED_CSV,
)

ALIASES = {
    "raw-prescribers": PART_D_PRESCRIBERS_CSV,
    "raw-payments": OPEN_PAYMENTS_CSV,
    "clean-prescribers": CLEAN_PRESCRIBERS_CSV,
    "clean-payments": CLEAN_PAYMENTS_CSV,
    "prescriber-level": PRESCRIBER_LEVEL_CSV,
    "aggregated": PRESCRIBER_LEVEL_CSV,
    "enriched": PRESCRIBER_LEVEL_ENRICHED_CSV,
    "scored": FRAUD_RISK_SCORED_CSV,
}


def resolve_path(path_arg: str) -> Path:
    if path_arg in ALIASES:
        return ALIASES[path_arg]
    return Path(path_arg).expanduser()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safe preview for huge CMS CSVs (never opens the full file)."
    )
    parser.add_argument(
        "path",
        help="File path or alias: scored, enriched, raw-payments, raw-prescribers, ...",
    )
    parser.add_argument("--rows", type=int, default=5, help="Rows to print (default: 5)")
    parser.add_argument(
        "--count",
        action="store_true",
        help="Count total lines via streaming (slow on multi-GB files)",
    )
    args = parser.parse_args()

    path = resolve_path(args.path)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"File: {path}")
    print(f"Size: {size_mb:.1f} MB")

    if args.count:
        print("Counting lines (streaming)...")
        with path.open("r", encoding="utf-8", errors="replace") as f:
            n = sum(1 for _ in f) - 1
        print(f"Data rows (approx): {n:,}")

    df = pd.read_csv(path, nrows=args.rows, low_memory=False)
    print(f"\nColumns ({len(df.columns)}): {list(df.columns)}")
    print(f"\nFirst {len(df)} rows:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

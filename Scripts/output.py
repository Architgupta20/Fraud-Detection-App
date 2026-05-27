"""Preview first rows of the merged payment-level dataset."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from config import MERGED_PAYMENT_LEVEL_CSV

file_path = MERGED_PAYMENT_LEVEL_CSV

if not file_path.exists():
    raise FileNotFoundError(
        f"Expected {file_path}. Run: python run_pipeline.py aggregate"
    )

df = pd.read_csv(file_path, nrows=5, low_memory=False)

print("===== First 5 Records (Vertical Output) =====\n")
for idx, row in df.iterrows():
    print(f"-RECORD {idx}----------------------------")
    for col_name, val in row.items():
        print(f"{col_name:25} | {val}")
    print()

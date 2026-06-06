#!/usr/bin/env python3
"""
Download HHS OIG LEIE CSV and load NPI-indexed exclusions into Postgres.

    export BASE_DIR="$(pwd)"
    python Scripts/load_oig_to_postgres.py

Uses DATABASE_URL from .env. Source (public domain):
https://oig.hhs.gov/exclusions/downloadables/UPDATED.csv
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import OIG_LEIE_CSV

OIG_LEIE_URL = "https://oig.hhs.gov/exclusions/downloadables/UPDATED.csv"
BATCH_SIZE = 5_000
VALID_NPI = re.compile(r"^\d{10}$")


def _load_dotenv() -> None:
    env_path = _ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _valid_npi(raw: object) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip().replace(".0", "")
    if not VALID_NPI.match(text) or text == "0000000000":
        return None
    return text


def _download_csv(dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {OIG_LEIE_URL} ...")
    urllib.request.urlretrieve(OIG_LEIE_URL, dest)
    print(f"Saved {dest} ({dest.stat().st_size / (1024 * 1024):.1f} MB)")


def _clean_str(raw: object) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, float) and raw != raw:
        return None
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def main() -> None:
    _load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set.")

    csv_path = Path(OIG_LEIE_CSV)
    if not csv_path.exists():
        _download_csv(csv_path)

    try:
        import pandas as pd
        import psycopg2
        from psycopg2.extras import execute_values
    except ImportError as exc:
        raise SystemExit("pip install pandas psycopg2-binary") from exc

    print(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    df.columns = [c.upper() for c in df.columns]

    rows: list[tuple] = []
    seen: set[str] = set()
    for rec in df.to_dict("records"):
        npi = _valid_npi(rec.get("NPI"))
        if not npi or npi in seen:
            continue
        seen.add(npi)
        rows.append(
            (
                npi,
                _clean_str(rec.get("LASTNAME")),
                _clean_str(rec.get("FIRSTNAME")),
                _clean_str(rec.get("BUSNAME")),
                _clean_str(rec.get("SPECIALTY")),
                _clean_str(rec.get("STATE")),
                _clean_str(rec.get("EXCLTYPE")),
                _clean_str(rec.get("EXCLDATE")),
                _clean_str(rec.get("REINDATE")),
            )
        )

    insert_sql = """
        INSERT INTO oig_exclusions (
            npi, last_name, first_name, business_name, specialty, state,
            exclusion_type, exclusion_date, reinstatement_date, loaded_at
        ) VALUES %s
        ON CONFLICT (npi) DO UPDATE SET
            last_name = EXCLUDED.last_name,
            first_name = EXCLUDED.first_name,
            business_name = EXCLUDED.business_name,
            specialty = EXCLUDED.specialty,
            state = EXCLUDED.state,
            exclusion_type = EXCLUDED.exclusion_type,
            exclusion_date = EXCLUDED.exclusion_date,
            reinstatement_date = EXCLUDED.reinstatement_date,
            loaded_at = NOW()
    """
    template = "(%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())"

    print(f"Loading {len(rows):,} NPI exclusions into Postgres ...")
    with psycopg2.connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE oig_exclusions")
            for i in range(0, len(rows), BATCH_SIZE):
                batch = rows[i : i + BATCH_SIZE]
                execute_values(cur, insert_sql, batch, template=template, page_size=1000)
                conn.commit()
                print(f"  {min(i + BATCH_SIZE, len(rows)):,} / {len(rows):,}")

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM oig_exclusions")
            oig_count = cur.fetchone()[0]
            cur.execute(
                """
                SELECT COUNT(*)
                FROM prescribers p
                INNER JOIN oig_exclusions o ON p.prescriber_id = o.npi
                """
            )
            overlap = cur.fetchone()[0]

    print(f"Done. oig_exclusions: {oig_count:,} rows | overlap with prescribers: {overlap:,}")


if __name__ == "__main__":
    main()

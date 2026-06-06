#!/usr/bin/env python3
"""
Create Postgres schema for Phase 2 (Step 2.2).

Requires DATABASE_URL in the environment, e.g. from Render External Database URL:

    export BASE_DIR="$(pwd)"
    export DATABASE_URL="postgresql://user:pass@host/dbname"
    python Scripts/init_postgres_schema.py

Optional: put DATABASE_URL in a local .env file (gitignored).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

SQL_DIR = Path(__file__).resolve().parent / "sql"


def _apply_sql_files(cur) -> None:
    files = sorted(SQL_DIR.glob("*.sql"))
    if not files:
        raise FileNotFoundError(f"No SQL files in {SQL_DIR}")
    for sql_file in files:
        print(f"Applying {sql_file.name} ...")
        cur.execute(sql_file.read_text())


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


def main() -> None:
    _load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit(
            "DATABASE_URL is not set. Add it to .env or export it "
            "(Render → Postgres → External Database URL)."
        )

    try:
        import psycopg2
    except ImportError as exc:
        raise SystemExit("Install psycopg2: pip install psycopg2-binary") from exc

    print(f"Applying schema from {SQL_DIR} ...")
    with psycopg2.connect(url) as conn:
        with conn.cursor() as cur:
            _apply_sql_files(cur)
        conn.commit()

    with psycopg2.connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                ORDER BY table_name
                """
            )
            tables = [r[0] for r in cur.fetchall()]
            cur.execute("SELECT COUNT(*) FROM prescribers")
            prescribers = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM reviews")
            reviews = cur.fetchone()[0]

    print(f"Tables: {', '.join(tables)}")
    print(f"prescribers: {prescribers:,} rows | reviews: {reviews:,} rows")


if __name__ == "__main__":
    main()

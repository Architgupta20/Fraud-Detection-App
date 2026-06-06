"""Postgres connection pool for the API."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

from api.config import DATABASE_URL

_pool: pool.SimpleConnectionPool | None = None


def init_pool(minconn: int = 1, maxconn: int = 5) -> None:
    global _pool
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")
    if _pool is None:
        _pool = pool.SimpleConnectionPool(minconn, maxconn, dsn=DATABASE_URL)


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None


@contextmanager
def get_cursor() -> Generator[Any, None, None]:
    if _pool is None:
        init_pool()
    assert _pool is not None
    conn = _pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)


def row_to_dict(row: dict | None) -> dict | None:
    if row is None:
        return None
    out: dict = {}
    for key, val in dict(row).items():
        if hasattr(val, "isoformat"):
            out[key] = val.isoformat()
        elif isinstance(val, float) and val == val:
            out[key] = val
        else:
            out[key] = val
    return out

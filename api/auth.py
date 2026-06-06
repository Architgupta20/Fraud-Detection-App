"""Optional API key auth for write endpoints."""

from __future__ import annotations

import os

from fastapi import Header, HTTPException

APP_API_KEY = os.environ.get("APP_API_KEY", "")


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    if not APP_API_KEY:
        return
    if x_api_key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

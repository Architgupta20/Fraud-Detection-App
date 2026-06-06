"""HTTP client for the Prescriber Risk API (Streamlit Step 4)."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
_TIMEOUT = float(os.getenv("API_TIMEOUT_SECONDS", "30"))


def api_health() -> Dict[str, Any]:
    """Return health JSON or {"status": "error", "database": "error"}."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {"status": "error", "database": "error"}


def api_is_ready() -> bool:
    data = api_health()
    return data.get("status") == "ok" and data.get("database") == "connected"


def fetch_prescriber(npi: str) -> Optional[Dict[str, Any]]:
    target = str(npi).strip()
    if not target:
        return None
    resp = requests.get(f"{API_BASE_URL}/prescribers/{target}", timeout=_TIMEOUT)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def fetch_prescribers(
    *,
    risk: Optional[str] = None,
    state: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {"limit": limit, "offset": offset}
    if risk and risk != "All":
        params["risk"] = risk
    if state and state.strip():
        params["state"] = state.strip().upper()
    resp = requests.get(f"{API_BASE_URL}/prescribers", params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()

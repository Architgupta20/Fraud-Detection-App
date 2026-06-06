"""HTTP client for the Prescriber Risk API (Streamlit Step 4+)."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
_TIMEOUT = float(os.getenv("API_TIMEOUT_SECONDS", "30"))


def _headers(api_key: Optional[str] = None) -> Dict[str, str]:
    key = api_key if api_key is not None else os.getenv("APP_API_KEY", "")
    if key:
        return {"X-API-Key": key}
    return {}


def api_health() -> Dict[str, Any]:
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


def fetch_stats_summary() -> Dict[str, Any]:
    resp = requests.get(f"{API_BASE_URL}/stats/summary", timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def fetch_stats_by_state(*, risk: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
    params: Dict[str, Any] = {"limit": limit}
    if risk and risk != "All":
        params["risk"] = risk
    resp = requests.get(f"{API_BASE_URL}/stats/by-state", params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def fetch_top_risk(*, limit: int = 10, state: Optional[str] = None) -> Dict[str, Any]:
    params: Dict[str, Any] = {"limit": limit}
    if state and state != "All":
        params["state"] = state
    resp = requests.get(f"{API_BASE_URL}/stats/top-risk", params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def fetch_reviews(
    *,
    status: Optional[str] = None,
    risk: str = "High",
    state: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {"risk": risk, "limit": limit, "offset": offset}
    if status and status != "All":
        params["status"] = status
    if state and state != "All":
        params["state"] = state
    resp = requests.get(f"{API_BASE_URL}/reviews", params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def upsert_review(
    npi: str,
    *,
    status: str,
    note: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    payload = {"status": status, "note": note or None}
    resp = requests.put(
        f"{API_BASE_URL}/reviews/{npi.strip()}",
        json=payload,
        headers=_headers(api_key),
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def export_reviews_csv(
    *,
    status: Optional[str] = None,
    risk: str = "High",
    state: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    params: Dict[str, Any] = {"risk": risk}
    if status and status != "All":
        params["status"] = status
    if state and state != "All":
        params["state"] = state
    resp = requests.get(
        f"{API_BASE_URL}/reviews/export",
        params=params,
        headers=_headers(api_key),
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.text

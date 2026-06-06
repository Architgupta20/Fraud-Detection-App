"""
Prescriber Risk API (Phase 2 Step 3).

Local run:
    export BASE_DIR="$(pwd)"
    uvicorn api.main:app --reload --port 8000

Docs: http://localhost:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query

from api.config import DATABASE_URL
from api.db import close_pool, get_cursor, init_pool, row_to_dict
from api.schemas import HealthResponse, Prescriber, PrescriberListResponse, PrescriberSummary

PRESCRIBER_SELECT = """
    SELECT prescriber_id, state, provider_type, fraud_risk_category, risk_points,
           rules_fired, rules_version, total_claims, total_drug_cost, opioid_claims,
           payment_to_drug_cost_ratio, peer_deviation_score, avg_risk_score,
           payment_variability, adjusted_risk_payment, high_payment_flag,
           high_opioid_flag, elderly_focus_flag, antibiotic_claim_ratio,
           antibiotic_claims, total_payment_amount, opioid_volume_pct_flag,
           peer_outlier_pct_flag, payment_spiky_pct_flag, total_payments_pct_flag,
           loaded_at
    FROM prescribers
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    if DATABASE_URL:
        init_pool()
    yield
    close_pool()


app = FastAPI(
    title="Prescriber Risk API",
    description="Rule-based review priority and prescriber metrics from Postgres.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if not DATABASE_URL:
        return HealthResponse(database="not_configured")
    try:
        with get_cursor() as cur:
            cur.execute("SELECT 1")
        return HealthResponse(database="connected")
    except Exception:
        return HealthResponse(database="error")


@app.get("/prescribers/{npi}", response_model=Prescriber)
def get_prescriber(npi: str) -> Prescriber:
    target = npi.strip()
    if not target:
        raise HTTPException(status_code=400, detail="NPI is required")
    with get_cursor() as cur:
        cur.execute(f"{PRESCRIBER_SELECT} WHERE prescriber_id = %s", (target,))
        row = row_to_dict(cur.fetchone())
    if row is None:
        raise HTTPException(status_code=404, detail=f"No prescriber found for NPI {target}")
    return Prescriber(**row)


@app.get("/prescribers", response_model=PrescriberListResponse)
def list_prescribers(
    risk: Optional[Literal["Low", "High"]] = Query(None, description="Filter by review priority"),
    state: Optional[str] = Query(None, min_length=2, max_length=2, description="Two-letter state code"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> PrescriberListResponse:
    clauses: list[str] = []
    params: list[object] = []
    if risk:
        clauses.append("fraud_risk_category = %s")
        params.append(risk)
    if state:
        clauses.append("state = %s")
        params.append(state.upper())

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    with get_cursor() as cur:
        cur.execute(f"SELECT COUNT(*) AS n FROM prescribers {where}", params)
        total = int(cur.fetchone()["n"])

        cur.execute(
            f"""
            SELECT prescriber_id, state, provider_type, fraud_risk_category,
                   risk_points, rules_fired
            FROM prescribers
            {where}
            ORDER BY risk_points DESC, prescriber_id
            LIMIT %s OFFSET %s
            """,
            [*params, limit, offset],
        )
        items = [PrescriberSummary(**row_to_dict(r)) for r in cur.fetchall()]

    return PrescriberListResponse(total=total, limit=limit, offset=offset, items=items)

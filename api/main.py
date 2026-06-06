"""
Prescriber Risk API (Phase 2 Step 3+).

Local run:
    export BASE_DIR="$(pwd)"
    uvicorn api.main:app --reload --port 8000

Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import csv
import io
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import Response

from api.auth import require_api_key
from api.config import DATABASE_URL
from api.db import close_pool, get_cursor, init_pool, row_to_dict
from api.schemas import (
    CategoryCount,
    HealthResponse,
    Prescriber,
    PrescriberListResponse,
    PrescriberSummary,
    ReviewListResponse,
    ReviewQueueItem,
    ReviewStatus,
    ReviewUpsert,
    StateCount,
    StatsByStateResponse,
    StatsSummaryResponse,
    TopRiskItem,
    TopRiskResponse,
)

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

REVIEW_QUEUE_SELECT = """
    SELECT p.prescriber_id, p.state, p.provider_type, p.fraud_risk_category,
           p.risk_points, p.rules_fired,
           COALESCE(r.status, 'pending') AS review_status,
           r.note, r.updated_at
    FROM prescribers p
    LEFT JOIN reviews r ON p.prescriber_id = r.prescriber_id
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
    version="0.2.0",
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


@app.get("/stats/summary", response_model=StatsSummaryResponse)
def stats_summary() -> StatsSummaryResponse:
    with get_cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM prescribers")
        total = int(cur.fetchone()["n"])
        cur.execute(
            """
            SELECT fraud_risk_category AS category, COUNT(*) AS count
            FROM prescribers
            GROUP BY fraud_risk_category
            ORDER BY fraud_risk_category
            """
        )
        by_category = [CategoryCount(**row_to_dict(r)) for r in cur.fetchall()]
    return StatsSummaryResponse(total_prescribers=total, by_category=by_category)


@app.get("/stats/by-state", response_model=StatsByStateResponse)
def stats_by_state(
    risk: Optional[Literal["Low", "High"]] = Query(None),
    limit: int = Query(20, ge=1, le=100),
) -> StatsByStateResponse:
    clauses = ["state IS NOT NULL", "state <> ''"]
    params: list[object] = []
    if risk:
        clauses.append("fraud_risk_category = %s")
        params.append(risk)
    where = "WHERE " + " AND ".join(clauses)
    with get_cursor() as cur:
        cur.execute(
            f"""
            SELECT state, COUNT(*) AS count
            FROM prescribers
            {where}
            GROUP BY state
            ORDER BY count DESC, state
            LIMIT %s
            """,
            [*params, limit],
        )
        items = [StateCount(**row_to_dict(r)) for r in cur.fetchall()]
    return StatsByStateResponse(risk=risk, items=items)


@app.get("/stats/top-risk", response_model=TopRiskResponse)
def stats_top_risk(
    limit: int = Query(10, ge=1, le=100),
    state: Optional[str] = Query(None, min_length=2, max_length=2),
) -> TopRiskResponse:
    clauses: list[str] = []
    params: list[object] = []
    if state:
        clauses.append("state = %s")
        params.append(state.upper())
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with get_cursor() as cur:
        cur.execute(
            f"""
            SELECT prescriber_id, state, provider_type, fraud_risk_category,
                   risk_points, rules_fired
            FROM prescribers
            {where}
            ORDER BY risk_points DESC, prescriber_id
            LIMIT %s
            """,
            [*params, limit],
        )
        items = [TopRiskItem(**row_to_dict(r)) for r in cur.fetchall()]
    return TopRiskResponse(limit=limit, items=items)


@app.get("/reviews", response_model=ReviewListResponse)
def list_reviews(
    status: Optional[ReviewStatus] = Query(None, description="Review status filter"),
    risk: Literal["Low", "High"] = Query("High", description="Prescriber risk category"),
    state: Optional[str] = Query(None, min_length=2, max_length=2),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> ReviewListResponse:
    clauses = ["p.fraud_risk_category = %s"]
    params: list[object] = [risk]
    if state:
        clauses.append("p.state = %s")
        params.append(state.upper())
    if status:
        clauses.append("COALESCE(r.status, 'pending') = %s")
        params.append(status)

    where = "WHERE " + " AND ".join(clauses)
    with get_cursor() as cur:
        cur.execute(
            f"""
            SELECT COUNT(*) AS n
            FROM prescribers p
            LEFT JOIN reviews r ON p.prescriber_id = r.prescriber_id
            {where}
            """,
            params,
        )
        total = int(cur.fetchone()["n"])
        cur.execute(
            f"""
            {REVIEW_QUEUE_SELECT}
            {where}
            ORDER BY p.risk_points DESC, p.prescriber_id
            LIMIT %s OFFSET %s
            """,
            [*params, limit, offset],
        )
        items = [ReviewQueueItem(**row_to_dict(r)) for r in cur.fetchall()]
    return ReviewListResponse(total=total, limit=limit, offset=offset, items=items)


@app.put("/reviews/{npi}", response_model=ReviewQueueItem, dependencies=[Depends(require_api_key)])
def upsert_review(npi: str, body: ReviewUpsert) -> ReviewQueueItem:
    target = npi.strip()
    if not target:
        raise HTTPException(status_code=400, detail="NPI is required")
    with get_cursor() as cur:
        cur.execute("SELECT 1 FROM prescribers WHERE prescriber_id = %s", (target,))
        if cur.fetchone() is None:
            raise HTTPException(status_code=404, detail=f"No prescriber found for NPI {target}")
        cur.execute(
            """
            INSERT INTO reviews (prescriber_id, status, note, updated_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (prescriber_id) DO UPDATE SET
                status = EXCLUDED.status,
                note = EXCLUDED.note,
                updated_at = NOW()
            """,
            (target, body.status, body.note),
        )
        cur.execute(
            f"{REVIEW_QUEUE_SELECT} WHERE p.prescriber_id = %s",
            (target,),
        )
        row = row_to_dict(cur.fetchone())
    return ReviewQueueItem(**row)


@app.get("/reviews/export")
def export_reviews(
    status: Optional[ReviewStatus] = Query(None),
    risk: Literal["Low", "High"] = Query("High"),
    state: Optional[str] = Query(None, min_length=2, max_length=2),
    _: None = Depends(require_api_key),
) -> Response:
    clauses = ["p.fraud_risk_category = %s"]
    params: list[object] = [risk]
    if state:
        clauses.append("p.state = %s")
        params.append(state.upper())
    if status:
        clauses.append("COALESCE(r.status, 'pending') = %s")
        params.append(status)
    where = "WHERE " + " AND ".join(clauses)

    with get_cursor() as cur:
        cur.execute(
            f"""
            {REVIEW_QUEUE_SELECT}
            {where}
            ORDER BY p.risk_points DESC, p.prescriber_id
            LIMIT 10000
            """,
            params,
        )
        rows = [row_to_dict(r) for r in cur.fetchall()]

    buf = io.StringIO()
    fieldnames = [
        "prescriber_id", "state", "provider_type", "fraud_risk_category",
        "risk_points", "rules_fired", "review_status", "note", "updated_at",
    ]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k) for k in fieldnames})

    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=review_queue_export.csv"},
    )

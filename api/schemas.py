"""Response models for the prescriber API."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Prescriber(BaseModel):
    prescriber_id: str
    state: Optional[str] = None
    provider_type: Optional[str] = None
    fraud_risk_category: str
    risk_points: int
    rules_fired: Optional[str] = None
    rules_version: Optional[str] = None
    total_claims: Optional[int] = None
    total_drug_cost: Optional[float] = None
    opioid_claims: Optional[int] = None
    payment_to_drug_cost_ratio: Optional[float] = None
    peer_deviation_score: Optional[float] = None
    avg_risk_score: Optional[float] = None
    payment_variability: Optional[float] = None
    adjusted_risk_payment: Optional[float] = None
    high_payment_flag: Optional[int] = None
    high_opioid_flag: Optional[int] = None
    elderly_focus_flag: Optional[int] = None
    antibiotic_claim_ratio: Optional[float] = None
    antibiotic_claims: Optional[int] = None
    total_payment_amount: Optional[float] = None
    opioid_volume_pct_flag: Optional[int] = None
    peer_outlier_pct_flag: Optional[int] = None
    payment_spiky_pct_flag: Optional[int] = None
    total_payments_pct_flag: Optional[int] = None
    loaded_at: Optional[str] = None


class PrescriberSummary(BaseModel):
    prescriber_id: str
    state: Optional[str] = None
    provider_type: Optional[str] = None
    fraud_risk_category: str
    risk_points: int
    rules_fired: Optional[str] = None


class PrescriberListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: List[PrescriberSummary]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    database: Literal["connected", "not_configured", "error"] = "connected"


class CategoryCount(BaseModel):
    category: str
    count: int


class StateCount(BaseModel):
    state: str
    count: int


class StatsSummaryResponse(BaseModel):
    total_prescribers: int
    by_category: List[CategoryCount]


class StatsByStateResponse(BaseModel):
    risk: Optional[str] = None
    items: List[StateCount]


class TopRiskItem(BaseModel):
    prescriber_id: str
    state: Optional[str] = None
    provider_type: Optional[str] = None
    fraud_risk_category: str
    risk_points: int
    rules_fired: Optional[str] = None


class TopRiskResponse(BaseModel):
    limit: int
    items: List[TopRiskItem]


ReviewStatus = Literal["pending", "reviewed", "needs_followup"]


class ReviewUpsert(BaseModel):
    status: ReviewStatus
    note: Optional[str] = None


class ReviewQueueItem(BaseModel):
    prescriber_id: str
    state: Optional[str] = None
    provider_type: Optional[str] = None
    fraud_risk_category: str
    risk_points: int
    rules_fired: Optional[str] = None
    review_status: ReviewStatus
    note: Optional[str] = None
    updated_at: Optional[str] = None


class ReviewListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: List[ReviewQueueItem]




from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.services.company_analysis_service import analyze_company


router = APIRouter()


# -----------------------------
# Request Model
# -----------------------------
class AnalyzeCompanyRequest(BaseModel):
    """Request payload for company ESG analysis."""

    electricity_kwh: float
    diesel_liters: float
    coal_kg: float
    waste_tons: float
    csr_spending: float
    employee_count: int


# -----------------------------
# Response Model
# -----------------------------
class AnalyzeCompanyResponse(BaseModel):
    """Response schema for unified company ESG analysis."""

    carbon: Dict[str, Any]
    esg: Dict[str, Any]
    anomaly: Dict[str, Any]
    compliance: str
    report: str


# -----------------------------
# API Info
# -----------------------------
@router.get("")
async def api_info() -> Dict[str, Any]:
    """API information endpoint."""
    return {
        "message": "ESG & Carbon Intelligence API",
        "endpoints": {
            "health": "/health",
            "analyze_company": "/api/v1/analyze-company",
            "docs": "/docs",
        },
    }


# -----------------------------
# Unified ESG Endpoint
# -----------------------------
@router.post(
    "/analyze-company",
    response_model=AnalyzeCompanyResponse,
    status_code=status.HTTP_200_OK,
    summary="Run end-to-end ESG & carbon analysis for a company.",
)
async def analyze_company_endpoint(
    payload: AnalyzeCompanyRequest,
) -> AnalyzeCompanyResponse:
    """
    Unified ESG & carbon intelligence endpoint.

    This endpoint:
      - Validates input automatically using Pydantic.
      - Delegates all computation to the company_analysis_service.
      - Returns a structured, enterprise-ready ESG intelligence payload.
    """

    try:
        result = analyze_company(
            electricity_kwh=payload.electricity_kwh,
            diesel_liters=payload.diesel_liters,
            coal_kg=payload.coal_kg,
            waste_tons=payload.waste_tons,
            csr_spending=payload.csr_spending,
            employee_count=payload.employee_count,
        )

        return AnalyzeCompanyResponse(**result)

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while performing ESG analysis.",
        )
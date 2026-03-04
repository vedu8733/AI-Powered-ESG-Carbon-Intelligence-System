"""
Company-level ESG analysis orchestration service.

This module composes lower-level services (carbon, ESG score, anomaly
detection, compliance RAG, and reporting) into a single, cohesive
analysis pipeline. It contains no FastAPI or transport-layer logic.
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.agents.compliance_agent import ComplianceAgent
from app.agents.report_agent import ReportAgent
from app.services.anomaly_service import detect_anomalies
from app.services.carbon_service import calculate_emissions
from app.services.esg_service import calculate_esg_score


_COMPLIANCE_DOCS: List[str] = [
    (
        "Companies should align their ESG disclosures with recognized "
        "standards such as the GRI, TCFD, and ISSB, providing transparent "
        "reporting on carbon emissions, climate risk, and social impact."
    ),
    (
        "Regulators increasingly expect robust governance over ESG data, "
        "including board-level oversight, internal controls, and clear "
        "responsibilities for sustainability reporting."
    ),
    (
        "High emissions intensity, inadequate waste management, and limited "
        "CSR investment can indicate elevated compliance and reputation risk "
        "under emerging climate and sustainability regulations."
    ),
]


def analyze_company(
    electricity_kwh: float,
    diesel_liters: float,
    coal_kg: float,
    waste_tons: float,
    csr_spending: float,
    employee_count: int,
) -> Dict[str, Any]:
    """
    Run full ESG and carbon intelligence analysis for a single company.

    This orchestration function:
      1. Calculates carbon emissions.
      2. Calculates ESG scores.
      3. Detects anomalies in input data.
      4. Runs a compliance-oriented RAG analysis.
      5. Generates an executive ESG report.

    Args:
        electricity_kwh: Annual electricity consumption in kWh.
        diesel_liters: Annual diesel consumption in liters.
        coal_kg: Annual coal consumption in kg.
        waste_tons: Waste generated in metric tons.
        csr_spending: CSR spending in currency units.
        employee_count: Number of employees.

    Returns:
        Dictionary with keys:
            - carbon
            - esg
            - anomaly
            - compliance
            - report

    Raises:
        ValueError: When underlying services raise validation errors.
        RuntimeError: When report or compliance generation fails.
    """

    # 1. Carbon analysis
    carbon_result = calculate_emissions(
        electricity_kwh=electricity_kwh,
        diesel_liters=diesel_liters,
        coal_kg=coal_kg,
        waste_tons=waste_tons,
    )

    # 2. ESG scoring (uses total emissions from carbon_result)
    total_emissions_tons = float(carbon_result.get("total_emissions_tons", 0.0))
    esg_result = calculate_esg_score(
        total_emissions_tons=total_emissions_tons,
        csr_spending=csr_spending,
        waste_tons=waste_tons,
        employee_count=employee_count,
    )

    # 3. Anomaly detection
    anomaly_result = detect_anomalies(
        electricity_kwh=electricity_kwh,
        diesel_liters=diesel_liters,
        coal_kg=coal_kg,
        waste_tons=waste_tons,
        csr_spending=csr_spending,
        employee_count=employee_count,
    )

    # 4. Compliance analysis via RAG-backed agent
    compliance_agent = ComplianceAgent(documents=_COMPLIANCE_DOCS)
    compliance_question = (
        "Provide an ESG and sustainability compliance assessment for a company "
        "with the following profile. Focus on key regulatory expectations, "
        "disclosure obligations, and potential areas of concern.\n\n"
        f"- Total emissions (tons CO2e): {total_emissions_tons}\n"
        f"- Waste generated (tons): {waste_tons}\n"
        f"- CSR spending: {csr_spending}\n"
        f"- Employee count: {employee_count}\n"
        f"- Anomaly status: {anomaly_result.get('status')}"
    )
    compliance_analysis = compliance_agent.analyze(compliance_question)

    # 5. Executive ESG report generation
    report_agent = ReportAgent()
    report = report_agent.generate_report(
        carbon_result=carbon_result,
        esg_result=esg_result,
        anomaly_result=anomaly_result,
        compliance_analysis=compliance_analysis,
    )

    return {
        "carbon": carbon_result,
        "esg": esg_result,
        "anomaly": anomaly_result,
        "compliance": compliance_analysis,
        "report": report,
    }



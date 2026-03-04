"""
AI Report Agent for ESG & Carbon Intelligence System.

This agent synthesizes:
- Carbon emission results
- ESG scoring
- Anomaly findings
- Compliance analysis (RAG)

into a structured executive-level ESG report.
"""

from __future__ import annotations

from typing import Dict, Any

from app.services.llm_service import generate_text, LLMServiceError


class ReportAgent:
    """
    High-level AI report generator for ESG intelligence.

    Generates a structured executive report in markdown format.
    """

    def generate_report(
        self,
        carbon_result: Dict[str, Any],
        esg_result: Dict[str, Any],
        anomaly_result: Dict[str, Any],
        compliance_analysis: str,
    ) -> str:
        """
        Generate full ESG intelligence report.

        Returns:
            Markdown formatted report string.
        """

        prompt = f"""
You are an ESG sustainability consultant.

Using the following structured analysis, generate a professional
corporate ESG report in clear markdown format.

CARBON ANALYSIS:
Total Emissions (tons): {carbon_result.get("total_emissions_tons")}
Breakdown: {carbon_result.get("breakdown_tons")}

ESG SCORE:
Environmental: {esg_result.get("environmental_score")}
Social: {esg_result.get("social_score")}
Governance: {esg_result.get("governance_score")}
Final ESG Score: {esg_result.get("final_esg_score")}
Risk Level: {esg_result.get("risk_level")}

ANOMALY ANALYSIS:
Status: {anomaly_result.get("status")}
Anomalies: {anomaly_result.get("anomalies_detected")}

COMPLIANCE ANALYSIS:
{compliance_analysis}

Structure the report with:

1. Executive Summary
2. Carbon Footprint Assessment
3. ESG Performance Evaluation
4. Compliance Review
5. Risk & Data Integrity Observations
6. Strategic Recommendations
7. Carbon Reduction Opportunities

Tone:
- Professional
- Corporate
- Clear and concise
- No emojis
"""

        try:
            report = generate_text(prompt)
        except LLMServiceError as exc:
            raise RuntimeError(f"Report generation failed: {exc}") from exc

        return report
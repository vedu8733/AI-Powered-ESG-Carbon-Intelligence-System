"""
ESG scoring business logic for the ESG & Carbon Intelligence System.

This module implements a transparent, deterministic scoring model
without any machine learning. It is intentionally simple and
explainable so that assumptions can be reviewed and adjusted.
"""

from __future__ import annotations

from typing import Dict


# Benchmarks and normalization constants (domain-agnostic defaults).
# These can be tuned per sector if needed, but are kept simple here.
EMISSIONS_BENCHMARK_TONS = 1_000.0  # Higher total emissions than this strongly penalize score.
WASTE_BENCHMARK_TONS = 100.0  # Waste above this level drives down the environmental score.
CSR_BENCHMARK_PER_EMPLOYEE = 200.0  # Approx. "good" CSR spend per employee in monetary units.


def _validate_non_negative_float(value: float, name: str) -> float:
    """Validate that a value is a number and non-negative."""
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return float(value)


def _validate_positive_int(value: int, name: str) -> int:
    """Validate that a value is an integer and strictly positive."""
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {type[value].__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def _clamp(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    """Clamp a numeric value to the inclusive range [minimum, maximum]."""
    return max(minimum, min(maximum, value))


def _inverse_linear_score(value: float, benchmark: float) -> float:
    """
    Simple inverse linear score in [0, 100].

    - value <= 0         -> score = 100
    - value >= benchmark -> score = 0
    - 0 < value < benchmark -> linearly scaled between 100 and 0
    """
    if value <= 0:
        return 100.0
    if benchmark <= 0:
        # Degenerate case; fall back to neutral.
        return 50.0

    ratio = value / benchmark
    raw_score = 100.0 * (1.0 - ratio)
    return _clamp(raw_score)


def _classify_dimension(score: float) -> str:
    """Classify a single ESG dimension into qualitative bands."""
    if score >= 75.0:
        return "strong"
    if score >= 50.0:
        return "moderate"
    return "weak"


def _determine_risk_level(final_score: float) -> str:
    """Map the final ESG score to a qualitative risk level."""
    if final_score >= 75.0:
        return "Low"
    if final_score >= 50.0:
        return "Medium"
    return "High"


def calculate_esg_score(
    total_emissions_tons: float,
    csr_spending: float,
    waste_tons: float,
    employee_count: int,
) -> Dict[str, object]:
    """
    Calculate an explainable ESG score based on simple, transparent rules.

    The score is composed as:
      - Environmental: 40%
      - Social: 30%
      - Governance: 30%

    All sub-scores are expressed on a 0–100 scale.

    Args:
        total_emissions_tons: Total GHG emissions in metric tons CO2e.
        csr_spending: Total CSR expenditure in arbitrary currency units.
        waste_tons: Waste generated in metric tons.
        employee_count: Number of employees (must be > 0).

    Returns:
        A dictionary with:
            - environmental_score (float)
            - social_score (float)
            - governance_score (float)
            - final_esg_score (float)
            - risk_level ("Low", "Medium", "High")
            - explanation (short, professional narrative)

    Raises:
        ValueError: When any input value is invalid.
    """

    total_emissions_tons = _validate_non_negative_float(
        total_emissions_tons, "total_emissions_tons"
    )
    csr_spending = _validate_non_negative_float(csr_spending, "csr_spending")
    waste_tons = _validate_non_negative_float(waste_tons, "waste_tons")
    employee_count = _validate_positive_int(employee_count, "employee_count")

    # --- Environmental (40%) ---
    # Inverse scaling of emissions: fewer emissions -> higher score.
    env_from_emissions = _inverse_linear_score(
        total_emissions_tons, EMISSIONS_BENCHMARK_TONS
    )

    # Penalize high waste: more waste -> lower score.
    env_from_waste = _inverse_linear_score(waste_tons, WASTE_BENCHMARK_TONS)

    # Combine with a modest tilt towards emissions.
    environmental_score = _clamp(0.7 * env_from_emissions + 0.3 * env_from_waste)

    # --- Social (30%) ---
    # CSR spending per employee, normalized against a benchmark.
    csr_per_employee = csr_spending / float(employee_count)
    if CSR_BENCHMARK_PER_EMPLOYEE > 0:
        social_raw = 100.0 * (csr_per_employee / CSR_BENCHMARK_PER_EMPLOYEE)
    else:
        social_raw = 50.0
    social_score = _clamp(social_raw)

    # --- Governance (30%) ---
    # Deterministic proxy using consistency between environmental
    # performance and CSR spending level.
    # - Start from the average of environmental and social.
    # - Add/subtract a small adjustment based on simple rules.
    base_governance = (environmental_score + social_score) / 2.0

    high_csr = csr_per_employee >= CSR_BENCHMARK_PER_EMPLOYEE
    low_emissions = total_emissions_tons <= EMISSIONS_BENCHMARK_TONS
    low_csr = csr_per_employee < 0.5 * CSR_BENCHMARK_PER_EMPLOYEE
    very_high_emissions = total_emissions_tons > 2 * EMISSIONS_BENCHMARK_TONS

    governance_adjustment = 0.0
    if high_csr and low_emissions:
        governance_adjustment += 10.0
    elif low_csr and very_high_emissions:
        governance_adjustment -= 10.0

    governance_score = _clamp(base_governance + governance_adjustment)

    # --- Final aggregation ---
    final_esg_score = _clamp(
        0.4 * environmental_score
        + 0.3 * social_score
        + 0.3 * governance_score
    )

    risk_level = _determine_risk_level(final_esg_score)

    # --- Explanation ---
    env_label = _classify_dimension(environmental_score)
    soc_label = _classify_dimension(social_score)
    gov_label = _classify_dimension(governance_score)

    explanation = (
        "ESG score is derived from inverse-scaled emissions and waste for the "
        f"environmental pillar ({env_label}), CSR spending per employee for the "
        f"social pillar ({soc_label}), and a governance proxy based on the "
        f"consistency between environmental performance and CSR investment "
        f"({gov_label}). The model is rule-based, deterministic, and designed "
        "for transparent review rather than predictive accuracy."
    )

    return {
        "environmental_score": environmental_score,
        "social_score": social_score,
        "governance_score": governance_score,
        "final_esg_score": final_esg_score,
        "risk_level": risk_level,
        "explanation": explanation,
    }


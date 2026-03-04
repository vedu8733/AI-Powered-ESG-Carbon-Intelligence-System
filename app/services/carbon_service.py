"""
Carbon-related business logic for the ESG & Carbon Intelligence System.

This module contains pure computation utilities with no API or I/O logic.
"""

from __future__ import annotations

from typing import Dict


# Emission factors (kg CO2 per activity unit)
ELECTRICITY_EMISSION_FACTOR = 0.82  # kg CO2 / kWh
DIESEL_EMISSION_FACTOR = 2.68  # kg CO2 / liter
COAL_EMISSION_FACTOR = 2.42  # kg CO2 / kg
WASTE_EMISSION_FACTOR = 100.0  # kg CO2 / ton


def _validate_non_negative_float(value: float, name: str) -> float:
    """
    Validate that value is a number and non-negative.

    Raises:
        ValueError: If the value is not a number or is negative.
    """
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return float(value)


def calculate_emissions(
    electricity_kwh: float,
    diesel_liters: float,
    coal_kg: float,
    waste_tons: float,
) -> Dict[str, object]:
    """
    Calculate carbon emissions and related metrics.

    Args:
        electricity_kwh: Electricity consumption in kWh.
        diesel_liters: Diesel consumption in liters.
        coal_kg: Coal consumption in kilograms.
        waste_tons: Waste generated in metric tons.

    Returns:
        A structured dictionary including:
            - total_emissions_tons: Total emissions in metric tons CO2e.
            - breakdown_tons: Emissions per source in metric tons CO2e.
            - emission_intensity: Relative contribution per source (0–1).
            - units: Metadata about units used.

    Raises:
        ValueError: If any input is invalid.
    """

    electricity_kwh = _validate_non_negative_float(electricity_kwh, "electricity_kwh")
    diesel_liters = _validate_non_negative_float(diesel_liters, "diesel_liters")
    coal_kg = _validate_non_negative_float(coal_kg, "coal_kg")
    waste_tons = _validate_non_negative_float(waste_tons, "waste_tons")

    # Emissions in kg CO2e
    electricity_kg = electricity_kwh * ELECTRICITY_EMISSION_FACTOR
    diesel_kg = diesel_liters * DIESEL_EMISSION_FACTOR
    coal_kg_emissions = coal_kg * COAL_EMISSION_FACTOR
    waste_kg = waste_tons * WASTE_EMISSION_FACTOR

    total_kg = electricity_kg + diesel_kg + coal_kg_emissions + waste_kg
    total_tons = total_kg / 1000.0

    breakdown_tons = {
        "electricity": electricity_kg / 1000.0,
        "diesel": diesel_kg / 1000.0,
        "coal": coal_kg_emissions / 1000.0,
        "waste": waste_kg / 1000.0,
    }

    if total_tons > 0:
        emission_intensity = {
            source: emissions_tons / total_tons
            for source, emissions_tons in breakdown_tons.items()
        }
    else:
        emission_intensity = {source: 0.0 for source in breakdown_tons}

    return {
        "total_emissions_tons": total_tons,
        "breakdown_tons": breakdown_tons,
        "emission_intensity": emission_intensity,
        "units": {
            "inputs": {
                "electricity_kwh": "kWh",
                "diesel_liters": "liter",
                "coal_kg": "kg",
                "waste_tons": "ton",
            },
            "emissions": "metric tons CO2e",
        },
    }


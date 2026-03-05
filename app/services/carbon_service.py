

from __future__ import annotations

from typing import Dict


# Emission factors (kg CO2 per activity unit) -- carboncalculation
ELECTRICITY_EMISSION_FACTOR = 0.82  # kg CO2 / kWh
DIESEL_EMISSION_FACTOR = 2.68  # kg CO2 / liter
COAL_EMISSION_FACTOR = 2.42  # kg CO2 / kg
WASTE_EMISSION_FACTOR = 100.0  # kg CO2 / ton


def _validate_non_negative_float(value: float, name: str) -> float:
   
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


"""
Anomaly detection business logic for ESG & Carbon Intelligence.

This module implements deterministic, rule-based and statistical checks
without any machine learning. It is designed to be transparent and
lightweight for production use.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def _validate_numeric(value, name: str) -> float:
    """Ensure that a value is numeric (int or float)."""
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric, got {type(value).__name__}")
    return float(value)


def _validate_int(value, name: str) -> int:
    """Ensure that a value is an integer."""
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    return value


def _logical_upper_bounds() -> Dict[str, float]:
    """
    Upper bounds used for simple sanity checks.

    These values are intentionally conservative and can be tuned for
    specific sectors if needed.
    """
    return {
        "electricity_kwh": 1e8,  # extremely high annual consumption
        "diesel_liters": 1e7,
        "coal_kg": 1e8,
        "waste_tons": 1e6,
        "csr_spending": 1e9,
        "employee_count": 1e6,
    }


def _zscore_parameters() -> Dict[str, Dict[str, float]]:
    """
    Reference means and standard deviations for Z-score checks.

    The values are generic baselines chosen for demonstrative purposes
    and can be revisited as domain data becomes available.
    """
    return {
        "electricity_kwh": {"mean": 50_000.0, "std": 25_000.0},
        "diesel_liters": {"mean": 10_000.0, "std": 5_000.0},
        "coal_kg": {"mean": 20_000.0, "std": 10_000.0},
        "waste_tons": {"mean": 1_000.0, "std": 500.0},
        "csr_spending": {"mean": 500_000.0, "std": 250_000.0},
        "employee_count": {"mean": 5_000.0, "std": 2_500.0},
    }


def _zscore(value: float, mean: float, std: float) -> float:
    """Compute a Z-score using NumPy, guarding against zero std."""
    arr = np.array([value, mean], dtype=float)
    if std <= 0:
        # Fallback: no dispersion info; treat as non-extreme.
        return 0.0
    return float((arr[0] - arr[1]) / std)


def detect_anomalies(
    electricity_kwh: float,
    diesel_liters: float,
    coal_kg: float,
    waste_tons: float,
    csr_spending: float,
    employee_count: int,
) -> Dict[str, object]:
    """
    Detect basic anomalies on operational and ESG-related inputs.

    The detection covers:
      1. Negative values.
      2. Unrealistic extreme values (simple upper bounds).
      3. Statistical outliers using Z-scores against fixed baselines.

    Returns a deterministic, explainable result without any ML.

    Raises:
        ValueError: When types are incorrect (non-numeric or non-integer).
    """

    # Type validation (keeps negative values as anomalies, not errors).
    electricity_kwh = _validate_numeric(electricity_kwh, "electricity_kwh")
    diesel_liters = _validate_numeric(diesel_liters, "diesel_liters")
    coal_kg = _validate_numeric(coal_kg, "coal_kg")
    waste_tons = _validate_numeric(waste_tons, "waste_tons")
    csr_spending = _validate_numeric(csr_spending, "csr_spending")
    employee_count = _validate_int(employee_count, "employee_count")

    values = {
        "electricity_kwh": electricity_kwh,
        "diesel_liters": diesel_liters,
        "coal_kg": coal_kg,
        "waste_tons": waste_tons,
        "csr_spending": csr_spending,
        "employee_count": float(employee_count),
    }

    anomalies: List[str] = []

    # 1. Negative values
    for name, val in values.items():
        if val < 0:
            anomalies.append(f"{name} is negative ({val}).")

    # 2. Unrealistic extreme values via upper bounds.
    upper_bounds = _logical_upper_bounds()
    for name, val in values.items():
        upper = upper_bounds.get(name)
        if upper is not None and val > upper:
            anomalies.append(
                f"{name}={val} exceeds logical upper bound of {upper}."
            )

    # 3. Statistical outliers using Z-score against fixed baselines.
    z_params = _zscore_parameters()
    for name, val in values.items():
        params = z_params.get(name)
        if not params:
            continue
        z = _zscore(val, params["mean"], params["std"])
        if abs(z) >= 3.0:
            anomalies.append(
                f"{name} appears as a statistical outlier (z-score={z:.2f})."
            )

    anomaly_count = len(anomalies)

    # Confidence score: start from 100 and decrease with anomaly count.
    # Each anomaly reduces confidence by a fixed step, with a floor at 0.
    confidence_penalty_per_anomaly = 10.0
    confidence_score = max(0.0, 100.0 - anomaly_count * confidence_penalty_per_anomaly)

    status = "Normal" if anomaly_count == 0 else "Suspicious"

    return {
        "anomalies_detected": anomalies,
        "anomaly_count": anomaly_count,
        "confidence_score": confidence_score,
        "status": status,
    }


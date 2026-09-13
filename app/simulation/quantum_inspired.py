"""Quantum-inspired probability analytics for the Monte Carlo UI.

This is NOT a quantum computer implementation and does not claim QAE speedup.
It uses classical Monte Carlo outcomes and converts a normalized probability
distribution into amplitude-like weights (alpha_i = sqrt(p_i)) so the frontend
can visualize a quantum-inspired future-state representation.
"""
from dataclasses import dataclass
from math import sqrt
from typing import Iterable


@dataclass(frozen=True)
class QuantumInspiredEstimate:
    sample_count: int
    success_probability: float
    counterattack_probability: float
    reserve_breach_probability: float
    expected_utility: float
    amplitude_success: float
    amplitude_counterattack: float
    amplitude_reserve_breach: float
    estimator: str = "CLASSICAL_MC_WITH_AMPLITUDE_INSPIRED_ENCODING"
    quantum_hardware_used: bool = False


def _amplitude(p: float) -> float:
    return round(sqrt(max(0.0, min(1.0, p))), 6)


def estimate(paths: Iterable[dict]) -> QuantumInspiredEstimate:
    rows = list(paths)
    if not rows:
        raise ValueError("At least one Monte Carlo path is required")
    n = len(rows)
    success = sum(bool(x.get("passed", False)) for x in rows) / n
    counter = sum(bool(x.get("counterattacked", False)) for x in rows) / n
    breach = sum(bool(x.get("reserve_breached", False)) for x in rows) / n
    utility = sum(float(x.get("utility", 0.0)) for x in rows) / n
    return QuantumInspiredEstimate(
        sample_count=n,
        success_probability=round(success, 4),
        counterattack_probability=round(counter, 4),
        reserve_breach_probability=round(breach, 4),
        expected_utility=round(utility, 5),
        amplitude_success=_amplitude(success),
        amplitude_counterattack=_amplitude(counter),
        amplitude_reserve_breach=_amplitude(breach),
    )

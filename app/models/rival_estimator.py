"""Infers rival (target car) tyre/energy proxies with uncertainty using only
observable stint/lap-trend heuristics. No proprietary ERS data assumed."""

from dataclasses import dataclass
from app.schemas.telemetry import DecisionRequest


@dataclass
class RivalEstimate:
    tyre_grip_estimate: float
    tyre_uncertainty: float
    defensive_likelihood: float  # heuristic proxy for counterattack propensity


def estimate(req: DecisionRequest) -> RivalEstimate:
    target = req.target

    # Older stints and negative (slowing) sector deltas suggest degraded tyres.
    stint_penalty = min(0.35, target.stint_age_laps * 0.015)
    trend_penalty = max(0.0, -target.recent_sector_delta_s) * 0.5
    grip = max(0.35, 0.9 - stint_penalty - trend_penalty)

    # Uncertainty grows with stint age since degradation curves get noisier.
    uncertainty = min(0.5, 0.1 + target.stint_age_laps * 0.01)

    # A rival with fresher tyres / recent positive sector deltas is more likely
    # to counterattack immediately after being passed.
    defensive_likelihood = max(0.1, min(0.9, 0.5 - target.recent_sector_delta_s * 2))

    return RivalEstimate(
        tyre_grip_estimate=round(grip, 3),
        tyre_uncertainty=round(uncertainty, 3),
        defensive_likelihood=round(defensive_likelihood, 3),
    )

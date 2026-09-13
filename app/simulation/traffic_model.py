"""Evolves the ego-target gap tick by tick under a chosen action, with a
stochastic rival response, so the rollout simulator can detect *when* (and
whether) a pass actually happens within the horizon, and whether the rival
counterattacks afterward -- not just estimate a single aggregate
probability."""

import random
from dataclasses import dataclass
from typing import Optional
from app.simulation.energy_model import speed_boost_kph
from app.simulation.tracks import Sector
from app.schemas.common import ActionType


@dataclass
class TrafficStepResult:
    gap_s: float
    relative_speed_kph: float
    passed: bool


def step(
    gap_s: float,
    relative_speed_kph: float,
    action: ActionType,
    sector: Sector,
    dt_s: float,
    rng: random.Random,
    defensive_likelihood: float,
    noise_std_kph: float = 4.0,
) -> TrafficStepResult:
    """One tick of gap evolution. Positive relative_speed_kph = closing.
    DRS sectors amplify closing speed since the tow effect compounds with
    deployment. Rival defends probabilistically based on defensive_likelihood,
    damping our closing speed when they react."""
    boost = speed_boost_kph(action)
    drs_multiplier = 1.35 if (sector.kind == "STRAIGHT" and sector.has_drs) else 1.0

    effective_relative_speed = (relative_speed_kph + boost) * drs_multiplier
    effective_relative_speed += rng.gauss(0, noise_std_kph)

    # Rival defensive reaction damps closing speed once the gap is already tight.
    if gap_s < 0.8 and rng.random() < defensive_likelihood:
        effective_relative_speed *= 0.6

    # Convert relative speed (kph) into gap change (s) for this tick.
    # gap_s decreases as we close; ~ (kph / 3.6) gives m/s closing rate,
    # and gap in seconds shrinks roughly proportional to closing_rate * dt / typical_speed.
    closing_ms = effective_relative_speed / 3.6
    gap_change_s = -(closing_ms * dt_s) / 80.0  # 80 m/s ~ nominal reference speed for scaling

    new_gap = gap_s + gap_change_s
    passed = new_gap <= 0.0
    new_gap = max(0.0, new_gap) if not passed else 0.0

    return TrafficStepResult(
        gap_s=round(new_gap, 4),
        relative_speed_kph=round(effective_relative_speed, 2),
        passed=passed,
    )


def post_pass_counterattack_check(
    rng: random.Random,
    defensive_likelihood: float,
    post_pass_traffic_gap_s: float,
    ticks_remaining: int,
    dt_s: float,
) -> bool:
    """After a successful pass, simulate whether the rival re-passes us
    (counterattack) before the horizon ends -- driven by their defensive
    likelihood and how much traffic/tow they have to work with."""
    if ticks_remaining <= 0:
        return False
    tow_available = max(0.0, 1.0 - post_pass_traffic_gap_s / 1.5)
    counterattack_prob_per_tick = 0.02 + 0.08 * defensive_likelihood * tow_available
    for _ in range(ticks_remaining):
        if rng.random() < counterattack_prob_per_tick * dt_s:
            return True
    return False

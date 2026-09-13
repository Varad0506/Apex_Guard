"""Battery SOC dynamics per simulation tick. Deliberately simple:
constant-rate charge/discharge modulated by track segment and action,
not FIA-grade electrical modeling -- credible relative comparisons between
actions are the goal, not exact physics.

Efficiency curve + soft power limit (mild, not a full thermal/ROM model):
real hybrid systems can't sustain full deploy power all the way down to an
empty battery, and regen efficiency tails off near full charge -- both are
smooth SOC-dependent tapers here, not hard cliffs, so a policy trained
against this doesn't learn to ride the SOC=0 edge at full power and get
surprised by a harder physical/thermal model later."""

from dataclasses import dataclass
from app.schemas.common import ActionType
from app.simulation.tracks import Sector

# Base rates in SOC-percent-per-second at full commitment, before the
# efficiency curve below is applied.
_DEPLOY_RATE_PCT_PER_S = {
    ActionType.HARVEST: 0.0,
    ActionType.HOLD: 0.0,
    ActionType.PARTIAL_DEPLOY: 1.1,
    ActionType.FULL_DEPLOY: 2.6,
}

_HARVEST_RATE_PCT_PER_S = {
    ActionType.HARVEST: 1.8,
    ActionType.HOLD: 0.3,   # small passive regen even when not actively harvesting
    ActionType.PARTIAL_DEPLOY: 0.1,
    ActionType.FULL_DEPLOY: 0.0,
}

# Speed delta (kph) contributed by deploying energy, used by the traffic model.
_SPEED_BOOST_KPH = {
    ActionType.HARVEST: -3.0,
    ActionType.HOLD: 0.0,
    ActionType.PARTIAL_DEPLOY: 9.0,
    ActionType.FULL_DEPLOY: 16.0,
}

# Below this SOC, deploy power tapers smoothly toward 0 instead of running
# at full rate right up until the hard 0% floor.
_DEPLOY_TAPER_START_PCT = 12.0
# Above this SOC, harvest/regen efficiency tapers smoothly toward 0 --
# a nearly-full pack can't absorb regen as effectively as a partly-depleted one.
_HARVEST_TAPER_START_PCT = 92.0


def _deploy_efficiency(soc_pct: float) -> float:
    """1.0 down to _DEPLOY_TAPER_START_PCT, smoothly falling to ~0.15 at
    soc_pct=0 (never fully 0, so PARTIAL/FULL_DEPLOY still nudge speed at
    very low SOC rather than behaving like HOLD)."""
    if soc_pct >= _DEPLOY_TAPER_START_PCT:
        return 1.0
    frac = max(0.0, soc_pct) / _DEPLOY_TAPER_START_PCT  # 0 at empty, 1 at taper start
    smooth = frac * frac * (3 - 2 * frac)  # smoothstep, avoids a kinked derivative
    return 0.15 + 0.85 * smooth


def _harvest_efficiency(soc_pct: float) -> float:
    """1.0 below _HARVEST_TAPER_START_PCT, smoothly falling to ~0.2 at
    soc_pct=100 (never fully 0, so trickle regen keeps working near full)."""
    if soc_pct <= _HARVEST_TAPER_START_PCT:
        return 1.0
    span = 100.0 - _HARVEST_TAPER_START_PCT
    frac = min(100.0, soc_pct - _HARVEST_TAPER_START_PCT) / span  # 0 at taper start, 1 at full
    smooth = frac * frac * (3 - 2 * frac)
    return 1.0 - 0.8 * smooth


@dataclass
class EnergyStepResult:
    soc_pct: float
    budget_kj_used: float


def speed_boost_kph(action: ActionType) -> float:
    return _SPEED_BOOST_KPH[action]


def step(
    soc_pct: float,
    action: ActionType,
    dt_s: float,
    sector: Sector,
    energy_recovery_factor: float,
    deploy_rate_multiplier: float = 1.0,
    harvest_rate_multiplier: float = 1.0,
) -> EnergyStepResult:
    """Advance SOC by one tick. Harvesting is stronger under braking
    (BRAKING_ZONE sectors), scaled by the track's energy_recovery_factor.

    deploy_rate_multiplier / harvest_rate_multiplier default to 1.0 (fully
    backward compatible with callers that don't pass them) and are meant to
    come from powertrain_modes.PowertrainModeTracker -- they layer discrete
    operating-mode behavior (override boost, super-clipping, lift-and-coast,
    the LOW_DERATE hard floor) on top of the continuous efficiency curve
    already applied below. The two are complementary, not redundant: the
    efficiency curve models "the same rate, less effective near the edges,"
    while the mode multipliers model "a qualitatively different rate,
    because the system is doing something tactically different."""
    harvest_rate = (
        _HARVEST_RATE_PCT_PER_S[action] * energy_recovery_factor
        * _harvest_efficiency(soc_pct) * harvest_rate_multiplier
    )
    if sector.kind == "BRAKING_ZONE":
        harvest_rate *= 1.6
    elif sector.kind == "STRAIGHT":
        harvest_rate *= 0.5

    deploy_rate = _DEPLOY_RATE_PCT_PER_S[action] * _deploy_efficiency(soc_pct) * deploy_rate_multiplier

    delta_pct = (harvest_rate - deploy_rate) * dt_s
    new_soc = max(0.0, min(100.0, soc_pct + delta_pct))

    # Nominal energy budget cost (kJ), roughly proportional to actual
    # (post-efficiency) deploy rate -- a throttled-back deploy at low SOC
    # should also cost less nominal budget, not the full-power figure.
    budget_kj_used = deploy_rate * dt_s * 12.0  # ~12kJ per %/s of deploy, nominal

    return EnergyStepResult(soc_pct=round(new_soc, 3), budget_kj_used=round(budget_kj_used, 2))

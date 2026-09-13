"""Discrete powertrain operating modes, replacing the flat constant-rate
energy accounting in energy_model.py's original version with something
closer to how a real hybrid powertrain actually behaves: the battery isn't
a linear tank, it has qualitatively different regimes near empty and near
full, and the driver/system behaves tactically differently in each.

Operating modes (SoC-driven):
    HIGH          -- plenty of charge, full deploy available on demand
    MEDIUM        -- normal operating range
    LOW_HARVEST   -- deliberately conserving: throttle-managed energy
                     building, not yet critical but actively recovering
    LOW_DERATE    -- SoC floor reached: full throttle commanded but the
                     battery physically cannot deliver more power. This is
                     qualitatively different from LOW_HARVEST -- it's not a
                     choice, it's a hard physical ceiling being hit.

Tactical modes (throttle/brake-driven, independent of SoC band):
    NORMAL           -- standard driving
    LIFT_AND_COAST   -- zero throttle well before a braking zone, maximizing
                        coast-down energy recovery at the cost of lap time
    SUPER_CLIPPING   -- full throttle commanded but engine power is partly
                        redirected to the battery rather than the wheels
                        (only meaningful near the top of the SoC band, where
                        deploy would otherwise be wasted/capped anyway)
    OVERRIDE_BOOST   -- transient high-power deployment beyond the nominal
                        rate, time-limited (can't be sustained -- see
                        MAX_OVERRIDE_TICKS below)

NOTE on a fixed bug: the original version of classify_tactical_mode()
required override_ticks_remaining > 0 to *enter* OVERRIDE_BOOST, but
override_ticks_remaining was only ever set to a nonzero value *inside*
PowertrainModeTracker.step()'s `if tactical_mode == OVERRIDE_BOOST` branch
-- i.e. entry was gated on a state that only gets set once you've already
entered. OVERRIDE_BOOST could never actually trigger. Fixed below: entry
only depends on the eligibility conditions (full deploy, full throttle, not
on cooldown); override_ticks_remaining now solely governs *duration* once
active, via PowertrainModeTracker.step()'s bookkeeping.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from app.schemas.common import ActionType


class PowertrainMode(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW_HARVEST = "LOW_HARVEST"
    LOW_DERATE = "LOW_DERATE"


class TacticalMode(str, Enum):
    NORMAL = "NORMAL"
    LIFT_AND_COAST = "LIFT_AND_COAST"
    SUPER_CLIPPING = "SUPER_CLIPPING"
    OVERRIDE_BOOST = "OVERRIDE_BOOST"


# SoC band boundaries (percent). Two bands near the bottom (LOW_HARVEST vs
# LOW_DERATE) is the key structural change from the old model: hitting the
# reserve floor and choosing to conserve above it are different states with
# different available actions, not points on the same linear curve.
_HIGH_THRESHOLD = 65.0
_MEDIUM_THRESHOLD = 30.0
_LOW_HARVEST_THRESHOLD = 15.0  # below this and still not at the hard floor -> LOW_HARVEST
# below the rules-supplied minimum_reserve_soc_pct -> LOW_DERATE (hard floor, not a soft band)

MAX_OVERRIDE_TICKS = 4   # override boost can't sustain longer than this many consecutive ticks
OVERRIDE_COOLDOWN_TICKS = 12  # ticks before override can be triggered again after use


@dataclass
class PowertrainState:
    mode: PowertrainMode
    tactical_mode: TacticalMode
    # Mode-dependent rate multipliers applied on top of energy_model's base
    # rates -- this is how the discrete mode actually changes behavior,
    # not just a label.
    deploy_rate_multiplier: float
    harvest_rate_multiplier: float
    override_ticks_remaining: int
    override_cooldown_remaining: int


def classify_powertrain_mode(soc_pct: float, minimum_reserve_soc_pct: float) -> PowertrainMode:
    """SoC-driven operating mode. minimum_reserve_soc_pct is the hard
    rules-supplied floor (from RulesState) -- LOW_DERATE means "at or below
    the floor," which the rule engine should already prevent reaching via
    deploy actions, but the mode classification itself doesn't assume
    that; it reports what it sees."""
    if soc_pct <= minimum_reserve_soc_pct:
        return PowertrainMode.LOW_DERATE
    if soc_pct < _LOW_HARVEST_THRESHOLD + minimum_reserve_soc_pct:
        return PowertrainMode.LOW_HARVEST
    if soc_pct < _HIGH_THRESHOLD:
        return PowertrainMode.MEDIUM
    return PowertrainMode.HIGH


def classify_tactical_mode(
    throttle_pct: float,
    brake_pct: float,
    action: ActionType,
    powertrain_mode: PowertrainMode,
    override_ticks_remaining: int,
    override_cooldown_remaining: int,
) -> TacticalMode:
    """Throttle/brake-driven tactical sub-mode, from directly observable
    telemetry (EgoState.throttle_pct / brake_pct are already real fields
    in the schema -- this uses what's there, not new sensors)."""
    # Lift-and-coast: zero throttle, not braking -- classic coast-down for
    # max regen, typically commanded ahead of a braking zone.
    if throttle_pct < 5.0 and brake_pct < 5.0:
        return TacticalMode.LIFT_AND_COAST

    # Override boost: only reachable via FULL_DEPLOY action, only if not on
    # cooldown. Entry depends only on eligibility here -- override_ticks_remaining
    # is purely a *duration* counter owned by PowertrainModeTracker.step(),
    # not an entry gate (see module docstring for the bug this fixes).
    if action == ActionType.FULL_DEPLOY and throttle_pct >= 98.0 and override_cooldown_remaining <= 0:
        return TacticalMode.OVERRIDE_BOOST

    # Super-clipping: full throttle commanded, but the powertrain is at or
    # near the SoC ceiling -- HIGH mode -- so continuing to "deploy" isn't
    # meaningful (nothing left to gain from more charge); the modeled
    # behavior is to redirect toward harvest instead. This is genuinely
    # different from LOW_DERATE's full-throttle-but-empty case -- here it's
    # full-throttle-but-*full*, at the other end of the band.
    if throttle_pct >= 98.0 and powertrain_mode == PowertrainMode.HIGH and action in (ActionType.HOLD, ActionType.HARVEST):
        return TacticalMode.SUPER_CLIPPING

    return TacticalMode.NORMAL


# Rate multipliers per (powertrain_mode, tactical_mode) combination. This
# is the actual behavioral payoff of the mode system -- these numbers
# modulate energy_model.py's base rates.
_MODE_RATE_TABLE = {
    # (powertrain_mode, tactical_mode): (deploy_multiplier, harvest_multiplier)
    (PowertrainMode.HIGH, TacticalMode.NORMAL): (1.0, 1.0),
    (PowertrainMode.HIGH, TacticalMode.SUPER_CLIPPING): (0.15, 1.6),  # most "deploy" intent redirected to harvest
    (PowertrainMode.HIGH, TacticalMode.LIFT_AND_COAST): (0.0, 1.3),
    (PowertrainMode.HIGH, TacticalMode.OVERRIDE_BOOST): (1.0, 1.0),  # no benefit to overriding when already full

    (PowertrainMode.MEDIUM, TacticalMode.NORMAL): (1.0, 1.0),
    (PowertrainMode.MEDIUM, TacticalMode.LIFT_AND_COAST): (0.0, 1.25),
    (PowertrainMode.MEDIUM, TacticalMode.OVERRIDE_BOOST): (1.45, 0.9),  # transient boost above nominal
    (PowertrainMode.MEDIUM, TacticalMode.SUPER_CLIPPING): (1.0, 1.0),  # not applicable outside HIGH; no-op

    (PowertrainMode.LOW_HARVEST, TacticalMode.NORMAL): (0.75, 1.15),  # system biases toward conserving even in "normal"
    (PowertrainMode.LOW_HARVEST, TacticalMode.LIFT_AND_COAST): (0.0, 1.4),
    (PowertrainMode.LOW_HARVEST, TacticalMode.OVERRIDE_BOOST): (1.2, 0.7),  # still possible but costly
    (PowertrainMode.LOW_HARVEST, TacticalMode.SUPER_CLIPPING): (0.75, 1.15),  # no-op outside HIGH

    (PowertrainMode.LOW_DERATE, TacticalMode.NORMAL): (0.0, 1.0),   # the whole point: can't deploy at the floor
    (PowertrainMode.LOW_DERATE, TacticalMode.LIFT_AND_COAST): (0.0, 1.2),
    (PowertrainMode.LOW_DERATE, TacticalMode.OVERRIDE_BOOST): (0.0, 1.0),  # override physically unavailable at the floor
    (PowertrainMode.LOW_DERATE, TacticalMode.SUPER_CLIPPING): (0.0, 1.0),  # no-op outside HIGH
}


def throttle_brake_proxy(sector_kind: str, action: ActionType) -> tuple[float, float]:
    """Shared heuristic for callers that simulate a hypothetical future
    (rollout Monte Carlo paths, RL training episodes) and therefore have
    no real EgoState.throttle_pct/brake_pct to read -- only sector kind
    and the chosen action. A live decision tick should use the real
    telemetry fields directly instead of this. HARVEST on a straight is
    modeled as a deliberate lift-and-coast; other actions on a straight
    assume the driver is at or near full throttle, consistent with a
    genuine DRS-zone attack window."""
    if sector_kind == "BRAKING_ZONE":
        return 0.0, 90.0
    if sector_kind == "CORNER":
        return 40.0, 0.0
    # STRAIGHT
    if action == ActionType.HARVEST:
        return 0.0, 0.0  # lift-and-coast
    if action == ActionType.HOLD:
        return 60.0, 0.0
    return 100.0, 0.0  # PARTIAL_DEPLOY / FULL_DEPLOY -- committed on the straight


class PowertrainModeTracker:
    """Stateful per-session tracker for override boost's self-limiting
    (ticks-remaining) and cooldown behavior -- this needs to persist across
    ticks within a rollout/session, unlike the stateless classify_*
    functions above."""

    def __init__(self):
        self.override_ticks_remaining = 0
        self.override_cooldown_remaining = 0

    def step(
        self,
        soc_pct: float,
        minimum_reserve_soc_pct: float,
        throttle_pct: float,
        brake_pct: float,
        action: ActionType,
    ) -> PowertrainState:
        powertrain_mode = classify_powertrain_mode(soc_pct, minimum_reserve_soc_pct)
        tactical_mode = classify_tactical_mode(
            throttle_pct, brake_pct, action, powertrain_mode,
            self.override_ticks_remaining, self.override_cooldown_remaining,
        )

        if tactical_mode == TacticalMode.OVERRIDE_BOOST:
            if self.override_ticks_remaining == 0:
                self.override_ticks_remaining = MAX_OVERRIDE_TICKS
            self.override_ticks_remaining -= 1
            if self.override_ticks_remaining <= 0:
                self.override_cooldown_remaining = OVERRIDE_COOLDOWN_TICKS
        elif self.override_cooldown_remaining > 0:
            self.override_cooldown_remaining -= 1

        deploy_mult, harvest_mult = _MODE_RATE_TABLE[(powertrain_mode, tactical_mode)]

        return PowertrainState(
            mode=powertrain_mode,
            tactical_mode=tactical_mode,
            deploy_rate_multiplier=deploy_mult,
            harvest_rate_multiplier=harvest_mult,
            override_ticks_remaining=self.override_ticks_remaining,
            override_cooldown_remaining=self.override_cooldown_remaining,
        )

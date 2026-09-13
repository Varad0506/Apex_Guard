"""Validates PowertrainModeTracker's stateful plumbing over synthetic
multi-tick sequences: mode classification boundaries, and specifically the
OVERRIDE_BOOST entry/duration/cooldown bug this module fixes (see
powertrain_modes.py's module docstring)."""

from app.schemas.common import ActionType
from app.simulation.powertrain_modes import (
    PowertrainMode,
    TacticalMode,
    PowertrainModeTracker,
    classify_powertrain_mode,
)


def test_soc_band_boundaries():
    assert classify_powertrain_mode(80.0, minimum_reserve_soc_pct=10.0) == PowertrainMode.HIGH
    assert classify_powertrain_mode(50.0, minimum_reserve_soc_pct=10.0) == PowertrainMode.MEDIUM
    assert classify_powertrain_mode(20.0, minimum_reserve_soc_pct=10.0) == PowertrainMode.LOW_HARVEST
    assert classify_powertrain_mode(10.0, minimum_reserve_soc_pct=10.0) == PowertrainMode.LOW_DERATE
    assert classify_powertrain_mode(5.0, minimum_reserve_soc_pct=10.0) == PowertrainMode.LOW_DERATE


def test_override_boost_actually_triggers_on_first_eligible_tick():
    """Regression test for the fixed bootstrapping bug: a fresh tracker,
    on its very first tick, with full-deploy + full-throttle + no
    cooldown, must enter OVERRIDE_BOOST immediately -- not stay NORMAL
    because override_ticks_remaining started at 0."""
    tracker = PowertrainModeTracker()
    state = tracker.step(
        soc_pct=50.0, minimum_reserve_soc_pct=10.0,
        throttle_pct=100.0, brake_pct=0.0, action=ActionType.FULL_DEPLOY,
    )
    assert state.tactical_mode == TacticalMode.OVERRIDE_BOOST
    assert state.deploy_rate_multiplier == 1.45  # MEDIUM/OVERRIDE_BOOST multiplier


def test_override_boost_self_limits_then_cools_down():
    """Synthetic multi-tick sequence: sustained full-deploy/full-throttle
    demand for far longer than MAX_OVERRIDE_TICKS. Override must cap out
    at MAX_OVERRIDE_TICKS consecutive ticks, then force a cooldown window
    of OVERRIDE_COOLDOWN_TICKS before it can trigger again, even though
    the driver keeps demanding it every tick."""
    tracker = PowertrainModeTracker()
    modes = []
    for _ in range(30):  # far more ticks than MAX_OVERRIDE_TICKS + cooldown needs to prove the cap
        state = tracker.step(
            soc_pct=50.0, minimum_reserve_soc_pct=10.0,
            throttle_pct=100.0, brake_pct=0.0, action=ActionType.FULL_DEPLOY,
        )
        modes.append(state.tactical_mode)

    # First MAX_OVERRIDE_TICKS ticks are OVERRIDE_BOOST...
    from app.simulation.powertrain_modes import MAX_OVERRIDE_TICKS, OVERRIDE_COOLDOWN_TICKS
    assert modes[:MAX_OVERRIDE_TICKS] == [TacticalMode.OVERRIDE_BOOST] * MAX_OVERRIDE_TICKS
    # ...then it must NOT still be boosting on the very next tick (the cap actually caps).
    assert modes[MAX_OVERRIDE_TICKS] != TacticalMode.OVERRIDE_BOOST
    # Nor at any point during the cooldown window.
    cooldown_window = modes[MAX_OVERRIDE_TICKS:MAX_OVERRIDE_TICKS + OVERRIDE_COOLDOWN_TICKS]
    assert TacticalMode.OVERRIDE_BOOST not in cooldown_window
    # It should be available again once the cooldown has fully elapsed.
    assert modes[MAX_OVERRIDE_TICKS + OVERRIDE_COOLDOWN_TICKS] == TacticalMode.OVERRIDE_BOOST


def test_lift_and_coast_and_derate_are_mutually_exclusive_with_override():
    tracker = PowertrainModeTracker()
    # Coasting: zero throttle, zero brake -- LIFT_AND_COAST regardless of action.
    state = tracker.step(
        soc_pct=50.0, minimum_reserve_soc_pct=10.0,
        throttle_pct=0.0, brake_pct=0.0, action=ActionType.HOLD,
    )
    assert state.tactical_mode == TacticalMode.LIFT_AND_COAST
    assert state.deploy_rate_multiplier == 0.0

    # At the SoC floor, deploy must be fully zeroed regardless of tactical mode.
    tracker2 = PowertrainModeTracker()
    state2 = tracker2.step(
        soc_pct=8.0, minimum_reserve_soc_pct=10.0,
        throttle_pct=100.0, brake_pct=0.0, action=ActionType.FULL_DEPLOY,
    )
    assert state2.mode == PowertrainMode.LOW_DERATE
    assert state2.deploy_rate_multiplier == 0.0


def test_super_clipping_only_applies_near_full_soc():
    # HIGH SoC + full throttle + HARVEST action -> SUPER_CLIPPING, redirecting to harvest.
    tracker = PowertrainModeTracker()
    state = tracker.step(
        soc_pct=90.0, minimum_reserve_soc_pct=10.0,
        throttle_pct=100.0, brake_pct=0.0, action=ActionType.HARVEST,
    )
    assert state.tactical_mode == TacticalMode.SUPER_CLIPPING
    assert state.harvest_rate_multiplier > 1.0

    # Same throttle/action but MEDIUM SoC -- must NOT super-clip (not applicable outside HIGH).
    tracker2 = PowertrainModeTracker()
    state2 = tracker2.step(
        soc_pct=50.0, minimum_reserve_soc_pct=10.0,
        throttle_pct=100.0, brake_pct=0.0, action=ActionType.HARVEST,
    )
    assert state2.tactical_mode != TacticalMode.SUPER_CLIPPING

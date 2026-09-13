"""Phase 0: the real lap-time + energy simulator. Multi-tick, stochastic,
Monte Carlo -- this is what the build guide means by "roll forward N ticks
in the simulator; score by net time gained + remaining energy + rule
headroom." Everything else (PPO training, GNN ablation testing) depends on
this existing, per the guide's "this is non-negotiable" framing.

Design choices, deliberately kept simple per the guide's own advice ("Keep it
simple and deterministic at first... you need credible relative comparisons
between actions, not FIA-grade vehicle physics"):

- Fixed tick size (dt_s), horizon in seconds, not laps.
- Track geometry for the horizon window is built live from the request's
  observed segment/straight/braking-zone fields (not from the standalone
  synth_* track profiles in tracks.py, which are for offline generalization
  testing against a policy/verifier -- see scripts/evaluate_rollout.py).
- Rival response is a simple probabilistic defensive reaction, not a
  separate learned rival policy (that's the Phase-2-style "rival behavior
  prediction" power-up from the guide, not built here).
- Running N stochastic paths per candidate action gives an *empirical*
  pass probability and a genuine confidence/uncertainty measure, instead of
  a single point estimate -- this is the guide's "Uncertainty handling"
  power-up, implemented as a side effect of doing the simulation properly.
"""

import hashlib
import random
import statistics
from dataclasses import dataclass, field
from typing import List

from app.schemas.common import ActionType
from app.schemas.telemetry import DecisionRequest
from app.engine.rule_engine import action_energy_cost
from app.engine.traffic_engine import TrafficAssessment
from app.models.rival_estimator import RivalEstimate
from app.models.opponent_belief import OpponentBeliefState, forecast_step
from app.simulation import energy_model, traffic_model
from app.simulation.powertrain_modes import PowertrainModeTracker, throttle_brake_proxy
from app.simulation.rollout import RolloutOutcome
from app.simulation.tracks import Sector

DEFAULT_HORIZON_S = 12.0
DEFAULT_DT_S = 0.5
DEFAULT_N_PATHS = 30

# Nominal per-pass time gain by action, same scale as the Phase 1 heuristic,
# used to convert "passed at tick T" into a time-delta estimate: passing
# earlier in the horizon is worth more than passing right at the edge of it.
_ACTION_BASE_TIME_GAIN = {
    ActionType.HARVEST: -0.11,
    ActionType.HOLD: 0.02,
    ActionType.PARTIAL_DEPLOY: 0.22,
    ActionType.FULL_DEPLOY: 0.29,
}


@dataclass
class PathResult:
    passed: bool
    pass_tick_s: float
    final_soc_pct: float
    budget_kj_used: float
    reserve_breached: bool
    counterattacked: bool


def _stable_seed(request_id: str, action_value: str) -> int:
    """Deterministic seed independent of Python's per-process hash
    randomization (PYTHONHASHSEED), so the same request always simulates
    identically -- required for the audit log / replay JSON to actually
    reproduce a decision."""
    digest = hashlib.sha256(f"{request_id}:{action_value}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _build_local_sectors(req: DecisionRequest, horizon_s: float) -> List[Sector]:
    """Builds a plausible sector sequence for the rollout horizon from the
    live telemetry's current segment, cycling through a generic
    straight/braking/corner pattern for anything beyond the currently
    observed segment."""
    track = req.track
    first_kind = "STRAIGHT" if "STRAIGHT" in track.segment_type else "CORNER"
    first_length = track.straight_remaining_m if first_kind == "STRAIGHT" else 150.0

    sectors = [
        Sector(kind=first_kind, length_m=max(50.0, first_length), has_drs=track.drs_available),
        Sector(kind="BRAKING_ZONE", length_m=max(60.0, track.braking_zone_m)),
        Sector(kind="CORNER", length_m=150.0),
    ]

    # Repeat a generic straight/braking/corner pattern to cover the horizon.
    # Duration accounting happens in _sector_schedule; this just needs to be
    # long enough in *segment count* to cover DEFAULT horizons comfortably.
    while len(sectors) < 12:
        sectors.append(Sector(kind="STRAIGHT", length_m=400.0, has_drs=track.drs_available))
        sectors.append(Sector(kind="BRAKING_ZONE", length_m=max(60.0, track.braking_zone_m)))
        sectors.append(Sector(kind="CORNER", length_m=150.0))

    return sectors


def _sector_at(sectors: List[Sector], elapsed_s: float, ref_speed_ms: float = 80.0) -> Sector:
    """Maps elapsed simulation time to a sector, using a nominal reference
    speed to convert sector length into duration. Cycles the sector list if
    the horizon runs past it."""
    t = 0.0
    idx = 0
    n = len(sectors)
    # Reference speed varies a bit by sector kind for a touch more realism.
    speed_by_kind = {"STRAIGHT": 90.0, "BRAKING_ZONE": 50.0, "CORNER": 40.0}
    guard = 0
    while guard < 500:
        sector = sectors[idx % n]
        duration = sector.length_m / speed_by_kind.get(sector.kind, ref_speed_ms)
        if t + duration > elapsed_s:
            return sector
        t += duration
        idx += 1
        guard += 1
    return sectors[-1]


def _simulate_one_path(
    req: DecisionRequest,
    action: ActionType,
    traffic: TrafficAssessment,
    rival: RivalEstimate,
    sectors: List[Sector],
    horizon_s: float,
    dt_s: float,
    rng: random.Random,
    opponent_belief_state: OpponentBeliefState = None,
) -> PathResult:
    gap_s = req.target.gap_s
    relative_speed = req.target.relative_speed_kph
    soc = req.ego.soc_pct
    budget_used = 0.0
    reserve_breached = False
    passed = False
    pass_tick_s = horizon_s
    energy_recovery_factor = 0.55  # nominal, live telemetry doesn't expose this directly
    powertrain_tracker = PowertrainModeTracker()

    # Sample hidden opponent state from the filtered posterior for this path.
    # The state is then re-forecast every physics tick, so opponent behavior is
    # not frozen for the whole Monte Carlo horizon.
    belief = opponent_belief_state
    defensive_likelihood = rival.defensive_likelihood
    if belief is not None:
        tactical_states = list(belief.tactical_belief)
        tactical_state = rng.choices(tactical_states, weights=[belief.tactical_belief[k] for k in tactical_states], k=1)[0]
        low_drag = rng.random() < belief.low_drag_probability
        override_available = rng.random() < belief.override_available_probability
        if tactical_state == "DEFENDING":
            defensive_likelihood += 0.14
        elif tactical_state == "HARVESTING":
            defensive_likelihood -= 0.08
        elif tactical_state == "ATTACKING":
            defensive_likelihood += 0.05
        if low_drag:
            defensive_likelihood += 0.06
        if override_available:
            defensive_likelihood += 0.08
        defensive_likelihood = max(0.0, min(1.0, defensive_likelihood))

    t = 0.0
    while t < horizon_s:
        sector = _sector_at(sectors, t)

        throttle_pct, brake_pct = throttle_brake_proxy(sector.kind, action)
        powertrain_state = powertrain_tracker.step(
            soc, req.rules.minimum_reserve_soc_pct, throttle_pct, brake_pct, action,
        )
        e_result = energy_model.step(
            soc, action, dt_s, sector, energy_recovery_factor,
            deploy_rate_multiplier=powertrain_state.deploy_rate_multiplier,
            harvest_rate_multiplier=powertrain_state.harvest_rate_multiplier,
        )
        soc = e_result.soc_pct
        budget_used += e_result.budget_kj_used
        if soc < req.rules.minimum_reserve_soc_pct:
            reserve_breached = True

        if not passed:
            t_result = traffic_model.step(
                gap_s, relative_speed, action, sector, dt_s, rng, defensive_likelihood
            )
            gap_s = t_result.gap_s
            relative_speed = t_result.relative_speed_kph
            if t_result.passed:
                passed = True
                pass_tick_s = t

        if belief is not None:
            belief = forecast_step(belief, gap_s, relative_speed, sector.kind, dt_s, rng)
            tactical = belief.tactical_belief
            defensive_likelihood = rival.defensive_likelihood
            defensive_likelihood += 0.10 * tactical.get("DEFENDING", 0.0)
            defensive_likelihood += 0.05 * tactical.get("ATTACKING", 0.0)
            defensive_likelihood += 0.05 * belief.low_drag_probability
            defensive_likelihood += 0.07 * belief.override_available_probability
            defensive_likelihood = max(0.0, min(1.0, defensive_likelihood))

        t += dt_s

    counterattacked = False
    if passed:
        ticks_remaining = int(round((horizon_s - pass_tick_s) / dt_s))
        counterattacked = traffic_model.post_pass_counterattack_check(
            rng, rival.defensive_likelihood, req.traffic.post_pass_traffic_gap_s, ticks_remaining, dt_s
        )

    return PathResult(
        passed=passed,
        pass_tick_s=pass_tick_s,
        final_soc_pct=soc,
        budget_kj_used=budget_used,
        reserve_breached=reserve_breached,
        counterattacked=counterattacked,
    )


def _run_verified_paths(
    req: DecisionRequest,
    action: ActionType,
    traffic: TrafficAssessment,
    rival: RivalEstimate,
    horizon_s: float,
    dt_s: float,
    n_paths: int,
    seed: int = None,
    opponent_belief_state: OpponentBeliefState = None,
) -> list[PathResult]:
    """Run and return the actual stochastic paths used by the verifier.

    Keeping the path list available lets experimental analytics (for example
    the quantum-inspired frontend view) inspect the *same* paths rather than
    reconstructing synthetic samples from aggregate probabilities.
    """
    rng = random.Random(seed if seed is not None else _stable_seed(req.request_id, action.value))
    sectors = _build_local_sectors(req, horizon_s)
    return [
        _simulate_one_path(req, action, traffic, rival, sectors, horizon_s, dt_s, rng, opponent_belief_state)
        for _ in range(max(1, n_paths))
    ]


def rollout_verified(
    req: DecisionRequest,
    action: ActionType,
    traffic: TrafficAssessment,
    rival: RivalEstimate,
    horizon_s: float = DEFAULT_HORIZON_S,
    dt_s: float = DEFAULT_DT_S,
    n_paths: int = DEFAULT_N_PATHS,
    seed: int = None,
    opponent_belief_state: OpponentBeliefState = None,
) -> RolloutOutcome:
    """Runs n_paths stochastic Monte Carlo rollouts of a single candidate
    action and aggregates them into the same RolloutOutcome shape the
    analytical rollout produces, plus a genuine confidence figure derived
    from the spread across paths (not a fixed placeholder)."""
    paths = _run_verified_paths(
        req, action, traffic, rival, horizon_s, dt_s, n_paths, seed, opponent_belief_state
    )

    pass_probability = sum(p.passed for p in paths) / len(paths)
    projected_soc_pct = statistics.mean(p.final_soc_pct for p in paths)
    budget_used_mean = statistics.mean(p.budget_kj_used for p in paths)
    reserve_breach_fraction = sum(p.reserve_breached for p in paths) / len(paths)

    passed_paths = [p for p in paths if p.passed]
    if passed_paths:
        # Earlier passes are worth more: scale base time gain by how much of
        # the horizon was left when the pass happened.
        avg_gain_fraction = statistics.mean(
            max(0.0, (horizon_s - p.pass_tick_s) / horizon_s) for p in passed_paths
        )
        immediate_time_delta_s = _ACTION_BASE_TIME_GAIN[action] * (0.5 + 0.5 * avg_gain_fraction)
        counterattack_fraction = sum(p.counterattacked for p in passed_paths) / len(passed_paths)
    else:
        immediate_time_delta_s = min(0.0, _ACTION_BASE_TIME_GAIN[action]) * 0.3
        counterattack_fraction = 0.0

    # Confidence: high when paths agree (pass probability near 0 or 1),
    # lower when the outcome is genuinely uncertain (near 0.5). This is the
    # "78% confidence this deploy gains time" power-up from the guide.
    confidence = round(1.0 - 2.0 * min(pass_probability, 1 - pass_probability), 3)
    confidence = max(0.35, confidence)

    energy_penalty = budget_used_mean / 3500.0  # normalize against a nominal per-stint budget
    risk_penalty = traffic.counterattack_risk * 0.15 + counterattack_fraction * 0.18
    reserve_penalty = reserve_breach_fraction * 0.6

    expected_net_value = (
        pass_probability * immediate_time_delta_s
        - energy_penalty
        - risk_penalty
        - reserve_penalty
    )

    return RolloutOutcome(
        action=action,
        legal=True,
        pass_probability=round(pass_probability, 3),
        immediate_time_delta_s=round(immediate_time_delta_s, 3),
        projected_soc_pct=round(projected_soc_pct, 2),
        expected_net_value=round(expected_net_value, 4),
        confidence=confidence,
        counterattack_rate=round(counterattack_fraction, 3),
        energy_used_kj=round(budget_used_mean, 3),
        reserve_breach_rate=round(reserve_breach_fraction, 3),
    )

"""Deterministic hard rule mask. This must run before any learned component
and can never be overridden by a model."""

from typing import List
from app.schemas.common import ActionType
from app.schemas.telemetry import DecisionRequest


def allowed_actions(req: DecisionRequest) -> List[ActionType]:
    """Return the legal action set for the current state. HOLD is (almost)
    always legal; everything else is gated by SOC, budget and segment rules."""
    ego = req.ego
    rules = req.rules

    legal: List[ActionType] = [ActionType.HOLD]

    # HARVEST is only legal where the simulator/track profile allows recovery.
    # For the deterministic MVP we allow harvest whenever SOC has headroom
    # to receive more charge and the car isn't already at max reserve.
    if ego.soc_pct < 100:
        legal.append(ActionType.HARVEST)

    # PARTIAL_DEPLOY blocked if remaining SOC would breach hard minimum,
    # or if the deployment budget is insufficient.
    partial_cost_kj = 120.0  # nominal energy cost of a partial deploy burst
    partial_soc_cost_pct = 8.0
    if (
        ego.soc_pct - partial_soc_cost_pct >= rules.minimum_reserve_soc_pct
        and rules.deployment_budget_remaining_kj >= partial_cost_kj
    ):
        legal.append(ActionType.PARTIAL_DEPLOY)

    # FULL_DEPLOY blocked when: SOC after projected action < minimum reserve,
    # budget exhausted, full deploy disallowed on segment, telemetry stale,
    # or verifier latency budget would be exceeded (checked upstream).
    full_cost_kj = 260.0
    full_soc_cost_pct = 22.0
    if (
        rules.full_deploy_allowed
        and ego.soc_pct - full_soc_cost_pct >= rules.minimum_reserve_soc_pct
        and rules.deployment_budget_remaining_kj >= full_cost_kj
        and req.telemetry_age_ms <= 500
    ):
        legal.append(ActionType.FULL_DEPLOY)

    return legal


def action_energy_cost(action: ActionType) -> tuple[float, float]:
    """Returns (soc_cost_pct, budget_cost_kj) for a given action. Positive
    soc_cost_pct means SOC is consumed; HARVEST returns a negative cost
    (SOC gain)."""
    costs = {
        ActionType.HOLD: (0.0, 0.0),
        ActionType.HARVEST: (-4.0, 0.0),
        ActionType.PARTIAL_DEPLOY: (8.0, 120.0),
        ActionType.FULL_DEPLOY: (22.0, 260.0),
    }
    return costs[action]

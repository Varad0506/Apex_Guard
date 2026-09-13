"""Simple, transparent analytical rollout. Projects immediate time delta,
SOC, and net value for a single candidate action. This is deliberately not
FIA-grade physics -- it needs to produce credible *relative* comparisons
between the four actions, not exact vehicle dynamics."""

from dataclasses import dataclass
from app.schemas.common import ActionType
from app.schemas.telemetry import DecisionRequest
from app.engine.rule_engine import action_energy_cost
from app.engine.traffic_engine import TrafficAssessment

# Nominal immediate time gain per action if the pass succeeds (seconds).
_ACTION_TIME_GAIN = {
    ActionType.HARVEST: -0.11,
    ActionType.HOLD: 0.02,
    ActionType.PARTIAL_DEPLOY: 0.22,
    ActionType.FULL_DEPLOY: 0.29,
}


@dataclass
class RolloutOutcome:
    action: ActionType
    legal: bool
    pass_probability: float
    immediate_time_delta_s: float
    projected_soc_pct: float
    expected_net_value: float
    confidence: float = 0.75  # placeholder
    counterattack_rate: float = 0.0
    energy_used_kj: float = 0.0
    reserve_breach_rate: float = 0.0
                               # analytical path has no sampled counterattack. for the single-shot analytical path;
                               # the Monte Carlo engine computes this for real.


def rollout(
    req: DecisionRequest,
    action: ActionType,
    pass_probability: float,
    traffic: TrafficAssessment,
) -> RolloutOutcome:
    soc_cost_pct, _budget_cost_kj = action_energy_cost(action)
    projected_soc = max(0.0, min(100.0, req.ego.soc_pct - soc_cost_pct))

    immediate_time_delta = _ACTION_TIME_GAIN[action]

    # Net value blends: (pass_probability * time_gain) against energy spent
    # and downstream counterattack/post-pass traffic risk. Bigger deployments
    # leave less energy in reserve for the next defensive lap, so they carry
    # proportionally more exposure to counterattack/traffic risk.
    energy_penalty = soc_cost_pct / 100.0
    full_gain = _ACTION_TIME_GAIN[ActionType.FULL_DEPLOY]
    exposure_factor = 0.3 + 0.7 * min(1.0, max(0.0, soc_cost_pct) / 22.0)
    risk_penalty = (traffic.counterattack_risk * 0.24 + traffic.post_pass_traffic_risk * 0.14) * exposure_factor

    # Reserve breach guard: if projected SOC dips below the rules minimum,
    # heavily penalize net value so it gets rejected by the ranker.
    reserve_breach_penalty = 0.0
    if projected_soc < req.rules.minimum_reserve_soc_pct:
        reserve_breach_penalty = 0.5

    expected_net_value = (
        pass_probability * immediate_time_delta
        - energy_penalty * 0.2
        - risk_penalty
        - reserve_breach_penalty
    )

    return RolloutOutcome(
        action=action,
        legal=True,
        pass_probability=pass_probability,
        immediate_time_delta_s=round(immediate_time_delta, 3),
        projected_soc_pct=round(projected_soc, 2),
        expected_net_value=round(expected_net_value, 4),
    )

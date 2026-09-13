from app.schemas.common import ActionType
from app.simulation.rollout import RolloutOutcome

_ACTION_PHRASING = {
    ActionType.HARVEST: "harvesting energy",
    ActionType.HOLD: "holding position without deploying",
    ActionType.PARTIAL_DEPLOY: "a partial deployment",
    ActionType.FULL_DEPLOY: "a full deployment",
}


def build(selected: RolloutOutcome, all_outcomes: list[RolloutOutcome]) -> str:
    phrase = _ACTION_PHRASING[selected.action]
    runner_up = max(
        (o for o in all_outcomes if o.action != selected.action),
        key=lambda o: o.expected_net_value,
        default=None,
    )
    text = (
        f"{phrase.capitalize()} maximizes expected net race-time value "
        f"(pass probability {selected.pass_probability:.0%}, "
        f"projected SOC {selected.projected_soc_pct:.0f}%)."
    )
    if runner_up is not None:
        text += (
            f" This outperforms {_ACTION_PHRASING[runner_up.action]} "
            f"({runner_up.expected_net_value:+.3f} vs {selected.expected_net_value:+.3f} net value)."
        )
    return text

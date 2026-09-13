from typing import List, Optional
from app.simulation.rollout import RolloutOutcome


def select_highest_legal_net_value(outcomes: List[RolloutOutcome]) -> Optional[RolloutOutcome]:
    """Picks the outcome with the highest expected_net_value among legal
    candidates. Returns None if no outcomes are provided."""
    legal_outcomes = [o for o in outcomes if o.legal]
    if not legal_outcomes:
        return None
    return max(legal_outcomes, key=lambda o: o.expected_net_value)

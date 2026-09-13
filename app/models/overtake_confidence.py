"""Confidence-aware logistic intelligence for ApexGuard.

The calibrated logistic overtake model is promoted from a baseline classifier
into an uncertainty and predicate layer.  It exposes:
  * action-distribution confidence (max calibrated probability),
  * normalized predictive entropy,
  * human-readable probabilistic predicates implemented as sigmoid mappers.

The layer is advisory.  It never bypasses the legal mask, Monte Carlo verifier,
or deterministic safety verifier.
"""
from __future__ import annotations

import math
from typing import Dict, List

from app.schemas.common import ActionType
from app.schemas.telemetry import DecisionRequest
from app.models.rival_estimator import RivalEstimate
from app.engine.traffic_engine import TrafficAssessment
from app.models import overtake_probability
from app.models.opponent_belief import OpponentBeliefState


def _sigmoid(x: float) -> float:
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _entropy(probabilities: List[float]) -> float:
    return -sum(p * math.log(max(p, 1e-9)) for p in probabilities)


def logistic_predicates(
    req: DecisionRequest,
    rival: RivalEstimate,
    traffic: TrafficAssessment,
    belief: OpponentBeliefState | None = None,
) -> Dict[str, float]:
    """Map telemetry/evidence into auditable probabilistic predicates.

    These are intentionally lightweight sigmoid predicate mappers rather than
    a second black-box model.  Thresholds are advisory and must be validated
    against benchmark data before being presented as safety guarantees.
    """
    t = req.target
    tr = req.traffic
    track = req.track
    reserve_margin = req.ego.soc_pct - req.rules.minimum_reserve_soc_pct

    corridor_open = _sigmoid(
        3.0 * (1.15 - t.gap_s)
        + 0.045 * t.relative_speed_kph
        + 1.2 * (1.0 - track.overtake_difficulty)
        + 0.8 * min(1.0, track.straight_remaining_m / 500.0)
        - 1.5 * max(0.0, 1.0 - tr.rear_gap_s / 1.5)
        - 1.0
    )
    closing_battle = _sigmoid(
        0.12 * t.relative_speed_kph - 2.0 * t.gap_s + 2.5 * (1.0 if t.gap_s < 1.0 else 0.0) - 0.8
    )
    rival_harvesting = _sigmoid(
        2.8 * max(0.0, t.recent_sector_delta_s)
        + 0.9 * (1.0 if t.stint_age_laps > 12 else 0.0)
        + 1.4 * (belief.harvesting_probability if belief else 0.0)
        - 0.9
    )
    low_drag = _sigmoid(
        2.4 * (traffic.tow_strength - 0.45)
        + 1.8 * (1.0 if track.drs_available else 0.0)
        + 1.0 * (belief.low_drag_probability if belief else 0.0)
        - 1.1
    )
    counter_harvest_trap = _sigmoid(
        2.5 * closing_battle
        + 2.2 * rival_harvesting
        + 1.6 * low_drag
        + 1.2 * traffic.counterattack_risk
        - 3.4
    )
    reserve_feasible = _sigmoid(0.22 * reserve_margin - 1.0)

    return {
        "corridor_open": round(corridor_open, 4),
        "closing_battle": round(closing_battle, 4),
        "rival_harvesting": round(rival_harvesting, 4),
        "low_drag": round(low_drag, 4),
        "counter_harvest_trap": round(counter_harvest_trap, 4),
        "reserve_feasible": round(reserve_feasible, 4),
    }


def assess(
    req: DecisionRequest,
    rival: RivalEstimate,
    traffic: TrafficAssessment,
    legal_actions: List[ActionType],
    belief: OpponentBeliefState | None = None,
) -> dict:
    """Return calibrated LR confidence + entropy + predicate probabilities."""
    probabilities = overtake_probability.predict_for_actions(req, rival, traffic, legal_actions)
    values = [float(probabilities[a]) for a in legal_actions]
    if not values:
        values = [0.0]

    # The classifier predicts independent pass probabilities per action, so
    # max probability is the cleanest confidence signal. Entropy is computed
    # over a normalized action distribution to quantify ambiguity between
    # candidate actions, not as a claim of calibrated Bayesian uncertainty.
    total = sum(max(0.0, p) for p in values)
    if total <= 1e-9:
        action_dist = [1.0 / len(values)] * len(values)
    else:
        action_dist = [max(0.0, p) / total for p in values]
    entropy = _entropy(action_dist)
    max_entropy = math.log(len(action_dist)) if len(action_dist) > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy else 0.0
    confidence = max(values)

    predicates = logistic_predicates(req, rival, traffic, belief)
    return {
        "overtake_probability": round(confidence, 4),
        "confidence": round(confidence, 4),
        "entropy": round(entropy, 4),
        "normalized_entropy": round(normalized_entropy, 4),
        "uncertainty": round(normalized_entropy, 4),
        "predicates": predicates,
    }

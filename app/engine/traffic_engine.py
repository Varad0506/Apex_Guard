"""Local battle traffic assessment.

The deterministic analytical model is retained as the fail-safe baseline.
When the trained TrafficGAT artifact is present, ``evaluate`` uses the GNN
prediction; any missing artifact or inference failure automatically falls
back to the baseline.  ``evaluate_baseline`` is intentionally exposed for
ablation/benchmark scripts so baseline-vs-GNN comparisons never recurse into
or accidentally use the GNN.
"""

from dataclasses import dataclass

from app.schemas.telemetry import DecisionRequest
from app.models.rival_estimator import RivalEstimate
from app.models.battle_graph import build_battle_graph
from app.models import traffic_graph
from app.models.opponent_belief import OpponentBeliefState


@dataclass
class TrafficAssessment:
    tow_strength: float
    counterattack_risk: float
    post_pass_traffic_risk: float


def evaluate_baseline(req: DecisionRequest, rival: RivalEstimate, belief: OpponentBeliefState | None = None) -> TrafficAssessment:
    """Deterministic hand-engineered traffic model used as the safety fallback."""
    traffic = req.traffic
    target = req.target

    # Tow strength: closer gaps and higher relative speed give a stronger slipstream.
    tow_strength = max(0.0, min(1.0, (1.2 - target.gap_s) * 0.6 + target.relative_speed_kph / 60.0))

    # Counterattack risk: tight rear gap + rival's defensive likelihood + nearby traffic density.
    rear_pressure = max(0.0, 1.0 - traffic.rear_gap_s / 1.5)
    density_pressure = min(1.0, traffic.cars_within_3s / 5.0)
    counterattack_risk = max(
        0.0,
        min(1.0, 0.45 * rear_pressure + 0.35 * rival.defensive_likelihood + 0.2 * density_pressure),
    )
    # Hidden Override availability is an additional counterattack channel.
    # Keep this additive term modest so the analytical baseline remains a
    # conservative fail-safe rather than becoming belief-model dependent.
    if belief is not None:
        counterattack_risk = max(
            0.0,
            min(1.0, counterattack_risk + 0.12 * belief.override_available_probability),
        )

    # Post-pass traffic risk: how likely we get swallowed by traffic right after passing.
    post_pass_risk = max(0.0, min(1.0, 1.0 - traffic.post_pass_traffic_gap_s / 1.5))

    return TrafficAssessment(
        tow_strength=round(tow_strength, 3),
        counterattack_risk=round(counterattack_risk, 3),
        post_pass_traffic_risk=round(post_pass_risk, 3),
    )


def evaluate_with_source(req: DecisionRequest, rival: RivalEstimate, belief: OpponentBeliefState | None = None) -> tuple[TrafficAssessment, str]:
    """Return the active traffic assessment and its source (``gnn``/``baseline``).

    GNN inference is deliberately fail-safe: it cannot crash the decision
    pipeline or bypass the deterministic model if the artifact is unavailable
    or malformed.
    """
    baseline = evaluate_baseline(req, rival, belief)

    if not traffic_graph.model_is_available():
        return baseline, "baseline"

    try:
        graph = build_battle_graph(req, rival)
        prediction = traffic_graph.predict(graph)
        traffic = TrafficAssessment(
            tow_strength=round(max(0.0, min(1.0, prediction["tow_strength"])), 3),
            counterattack_risk=round(max(0.0, min(1.0, prediction["counterattack_risk"])), 3),
            post_pass_traffic_risk=round(max(0.0, min(1.0, prediction["post_pass_traffic_risk"])), 3),
        )
        if belief is not None:
            traffic.counterattack_risk = round(
                max(0.0, min(1.0, 0.88 * traffic.counterattack_risk + 0.12 * belief.override_available_probability)),
                3,
            )
        return traffic, "gnn"
    except Exception:
        return baseline, "baseline"


def evaluate(req: DecisionRequest, rival: RivalEstimate) -> TrafficAssessment:
    """Backward-compatible traffic API returning only the active assessment."""
    traffic, _ = evaluate_with_source(req, rival)
    return traffic

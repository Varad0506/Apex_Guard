import json
from pathlib import Path

from app.models.opponent_belief import infer
from app.models.rival_estimator import estimate
from app.schemas.telemetry import DecisionRequest

SCENARIO_DIR = Path(__file__).resolve().parents[1] / "app" / "data" / "scenarios"


def request(name):
    return DecisionRequest(**json.loads((SCENARIO_DIR / f"{name}.json").read_text()))


def test_belief_distributions_are_normalized():
    req = request("isolated_pass")
    belief = infer(req, estimate(req))
    for distribution in (belief.soc_belief, belief.override_belief, belief.tactical_belief, belief.aero_belief):
        assert abs(sum(distribution.values()) - 1.0) < 1e-6
        assert all(0.0 <= value <= 1.0 for value in distribution.values())
    assert 0.0 <= belief.counter_harvest_trap_probability <= 1.0
    assert 0.0 <= belief.confidence <= 1.0


def test_trap_signal_rises_for_healthy_slowing_rival_on_straight():
    data = json.loads((SCENARIO_DIR / "isolated_pass.json").read_text())
    data["target"]["relative_speed_kph"] = -6.0
    data["target"]["recent_sector_delta_s"] = 0.0
    data["target"]["stint_age_laps"] = 2
    data["target"]["gap_s"] = 0.8
    data["track"]["segment_type"] = "DRS_STRAIGHT"
    req = DecisionRequest(**data)
    belief = infer(req, estimate(req))

    assert belief.harvesting_probability > 0.20
    assert belief.low_drag_probability > 0.50
    assert belief.override_available_probability > 0.45
    assert belief.counter_harvest_trap_probability > 0.08


def test_decision_response_exposes_belief_state():
    from app.engine.decision_engine import decide
    req = request("isolated_pass")
    response = decide(req)
    assert "soc_belief" in response.opponent_belief
    assert "override_belief" in response.opponent_belief
    assert "tactical_belief" in response.opponent_belief
    assert "aero_belief" in response.opponent_belief
    assert "counter_harvest_trap_probability" in response.opponent_belief


def test_temporal_filter_accumulates_consistent_trap_evidence():
    from app.models.opponent_belief import reset_temporal, update_temporal

    data = json.loads((SCENARIO_DIR / "isolated_pass.json").read_text())
    data["battle_id"] = "test-battle-temporal"
    data["target"]["relative_speed_kph"] = -6.0
    data["target"]["recent_sector_delta_s"] = 0.0
    data["target"]["stint_age_laps"] = 2
    data["target"]["gap_s"] = 0.8
    data["track"]["segment_type"] = "DRS_STRAIGHT"

    reset_temporal(data["battle_id"])
    beliefs = []
    for i in range(5):
        tick = dict(data)
        tick["request_id"] = f"temporal-{i}"
        tick["timestamp_ms"] = data["timestamp_ms"] + i * 1000
        belief = update_temporal(DecisionRequest(**tick), estimate(DecisionRequest(**tick)), data["battle_id"])
        beliefs.append(belief)

    assert beliefs[0].temporal is False
    assert beliefs[-1].temporal is True
    assert beliefs[-1].update_count == 5
    assert beliefs[-1].counter_harvest_trap_probability >= beliefs[0].counter_harvest_trap_probability
    assert beliefs[-1].confidence >= 0.0
    reset_temporal(data["battle_id"])


def test_forecast_belief_evolves_inside_simulation_horizon():
    from app.models.opponent_belief import forecast_step
    data = json.loads((SCENARIO_DIR / "isolated_pass.json").read_text())
    req = DecisionRequest(**data)
    rival = estimate(req)
    belief = infer(req, rival)
    values = []
    for _ in range(8):
        belief = forecast_step(belief, 0.8, -5.0, "DRS_STRAIGHT", 0.5)
        values.append(belief.counter_harvest_trap_probability)
    assert all(0.0 <= v <= 1.0 for v in values)
    assert belief.update_count == 8
    assert len(set(round(v, 6) for v in values)) > 1


def test_verified_rollout_uses_temporal_belief_without_breaking_bounds():
    from app.engine import traffic_engine
    from app.simulation.engine import rollout_verified
    from app.models.opponent_belief import infer
    from app.schemas.common import ActionType
    data = json.loads((SCENARIO_DIR / "rear_drs_threat.json").read_text())
    req = DecisionRequest(**data)
    rival = estimate(req)
    traffic = traffic_engine.evaluate(req, rival)
    belief = infer(req, rival)
    outcome = rollout_verified(req, ActionType.PARTIAL_DEPLOY, traffic, rival, n_paths=12, opponent_belief_state=belief)
    assert 0.0 <= outcome.pass_probability <= 1.0
    assert 0.0 <= outcome.confidence <= 1.0
    assert 0.0 <= outcome.projected_soc_pct <= 100.0

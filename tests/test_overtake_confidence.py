from app.models import overtake_confidence, rival_estimator
from app.engine.traffic_engine import evaluate_baseline
from app.schemas.common import ActionType, DecisionMode
from app.schemas.telemetry import DecisionRequest


def _req():
    return DecisionRequest.model_validate({
        "request_id": "confidence-test",
        "battle_id": "confidence-battle",
        "timestamp_ms": 1778600000000,
        "telemetry_age_ms": 30,
        "ego": {"soc_pct": 62, "speed_kph": 290, "throttle_pct": 98, "brake_pct": 0, "tyre_grip_estimate": .84, "laps_remaining": 12, "current_mode": "HOLD"},
        "target": {"gap_s": .65, "relative_speed_kph": 12, "stint_age_laps": 14, "recent_sector_delta_s": .10},
        "traffic": {"rear_gap_s": 1.6, "cars_within_3s": 2, "post_pass_traffic_gap_s": 1.5},
        "track": {"track_id": "test", "segment_type": "DRS_STRAIGHT", "drs_available": True, "straight_remaining_m": 500, "braking_zone_m": 130, "overtake_difficulty": .35},
        "rules": {"deployment_budget_remaining_kj": 1500, "minimum_reserve_soc_pct": 15, "full_deploy_allowed": True},
        "decision_mode": DecisionMode.VERIFIED,
    })


def test_confidence_has_entropy_and_predicates():
    req = _req()
    rival = rival_estimator.estimate(req)
    traffic = evaluate_baseline(req, rival)
    result = overtake_confidence.assess(req, rival, traffic, list(ActionType))
    assert 0 <= result["confidence"] <= 1
    assert 0 <= result["normalized_entropy"] <= 1
    assert set(("corridor_open", "closing_battle", "rival_harvesting", "low_drag", "counter_harvest_trap", "reserve_feasible")) <= set(result["predicates"])

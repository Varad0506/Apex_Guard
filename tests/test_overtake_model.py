import json
from pathlib import Path
import pytest
from app.schemas.telemetry import DecisionRequest
from app.schemas.common import ActionType
from app.engine import rule_engine, traffic_engine
from app.models import rival_estimator, overtake_probability

SCENARIOS_DIR = Path(__file__).resolve().parents[1] / "app" / "data" / "scenarios"
MODEL_PATH = Path(__file__).resolve().parents[1] / "app" / "data" / "artifacts" / "overtake_model.joblib"


def load_scenario(name):
    with (SCENARIOS_DIR / f"{name}.json").open() as f:
        return DecisionRequest(**json.load(f))


def test_probabilities_bounded_regardless_of_model_availability():
    req = load_scenario("rear_drs_threat")
    legal = rule_engine.allowed_actions(req)
    rival = rival_estimator.estimate(req)
    traffic = traffic_engine.evaluate(req, rival)
    probs = overtake_probability.predict_for_actions(req, rival, traffic, legal)
    assert set(probs.keys()) == set(legal)
    for p in probs.values():
        assert 0.0 <= p <= 1.0


def test_more_aggressive_action_has_higher_pass_probability():
    # True for both the heuristic and the trained model given our synthetic
    # ground truth: more energy deployed -> higher pass probability, all
    # else equal.
    req = load_scenario("isolated_pass")
    legal = rule_engine.allowed_actions(req)
    rival = rival_estimator.estimate(req)
    traffic = traffic_engine.evaluate(req, rival)
    probs = overtake_probability.predict_for_actions(req, rival, traffic, legal)
    assert probs[ActionType.FULL_DEPLOY] > probs[ActionType.HOLD]


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="trained model artifact not present")
def test_trained_model_reports_available():
    assert overtake_probability.model_is_available() is True


def test_health_ready_reports_overtake_model_mode():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    resp = client.get("/v1/health/ready")
    assert resp.status_code == 200
    assert resp.json()["overtake_model"] in ("trained-v1", "heuristic-v0")

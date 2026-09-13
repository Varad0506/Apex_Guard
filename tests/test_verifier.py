import json
from pathlib import Path
from app.schemas.telemetry import DecisionRequest
from app.schemas.common import ActionType
from app.engine import rule_engine, traffic_engine, ranker
from app.models import rival_estimator, overtake_probability
from app.simulation.rollout import rollout

SCENARIOS_DIR = Path(__file__).resolve().parents[1] / "app" / "data" / "scenarios"


def load_scenario(name):
    with (SCENARIOS_DIR / f"{name}.json").open() as f:
        return DecisionRequest(**json.load(f))


def test_ranker_rejects_reserve_breaching_full_deploy():
    req = load_scenario("rear_drs_threat")
    legal = rule_engine.allowed_actions(req)
    rival = rival_estimator.estimate(req)
    traffic = traffic_engine.evaluate(req, rival)
    probs = overtake_probability.predict_for_actions(req, rival, traffic, legal)
    outcomes = [rollout(req, a, probs[a], traffic) for a in legal]
    selected = ranker.select_highest_legal_net_value(outcomes)
    assert selected is not None
    # High counterattack + traffic risk in this scenario should discourage FULL_DEPLOY.
    assert selected.action != ActionType.FULL_DEPLOY

import json
from pathlib import Path
from app.schemas.telemetry import DecisionRequest
from app.schemas.common import ActionType
from app.engine import rule_engine

SCENARIOS_DIR = Path(__file__).resolve().parents[1] / "app" / "data" / "scenarios"


def load_scenario(name):
    with (SCENARIOS_DIR / f"{name}.json").open() as f:
        return DecisionRequest(**json.load(f))


def test_full_deploy_blocked_by_low_soc():
    req = load_scenario("low_soc_guard")
    legal = rule_engine.allowed_actions(req)
    assert ActionType.FULL_DEPLOY not in legal


def test_hold_always_legal():
    req = load_scenario("isolated_pass")
    legal = rule_engine.allowed_actions(req)
    assert ActionType.HOLD in legal


def test_full_deploy_legal_with_healthy_soc():
    req = load_scenario("isolated_pass")
    legal = rule_engine.allowed_actions(req)
    assert ActionType.FULL_DEPLOY in legal

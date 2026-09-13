import json
from pathlib import Path
from app.schemas.telemetry import DecisionRequest
from app.schemas.common import ActionType, DecisionStatus
from app.engine.decision_engine import decide

SCENARIOS_DIR = Path(__file__).resolve().parents[1] / "app" / "data" / "scenarios"


def load_scenario(name):
    with (SCENARIOS_DIR / f"{name}.json").open() as f:
        return json.load(f)


def test_stale_telemetry_forces_hold():
    data = load_scenario("isolated_pass")
    data["telemetry_age_ms"] = 999
    req = DecisionRequest(**data)
    response = decide(req)
    assert response.recommended_action == ActionType.HOLD
    assert response.status == DecisionStatus.SAFE_HOLD


def test_low_soc_never_recommends_full_deploy():
    data = load_scenario("low_soc_guard")
    req = DecisionRequest(**data)
    response = decide(req)
    assert response.recommended_action != ActionType.FULL_DEPLOY

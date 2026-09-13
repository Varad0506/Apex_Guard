import json
from pathlib import Path
from app.schemas.telemetry import DecisionRequest
from app.engine import opportunity_monitor

SCENARIOS_DIR = Path(__file__).resolve().parents[1] / "app" / "data" / "scenarios"


def load_scenario(name):
    with (SCENARIOS_DIR / f"{name}.json").open() as f:
        return DecisionRequest(**json.load(f))


def test_close_gap_high_opportunity():
    req = load_scenario("rear_drs_threat")
    score = opportunity_monitor.score(req)
    assert score >= 0.20


def test_score_bounded():
    req = load_scenario("isolated_pass")
    score = opportunity_monitor.score(req)
    assert 0.0 <= score <= 1.0

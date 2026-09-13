from app.engine.ers_2026 import assess, normal_ers_power_limit_kw, override_power_limit_kw
from app.schemas.common import ERSTacticalAction
from app.schemas.telemetry import DecisionRequest
from app.simulation.quantum_inspired import estimate
import json
from pathlib import Path


def load(name="isolated_pass"):
    path = Path(__file__).resolve().parents[1] / "app" / "data" / "scenarios" / f"{name}.json"
    return DecisionRequest(**json.loads(path.read_text()))


def test_power_limit_uses_2026_zone_caps():
    assert normal_ers_power_limit_kw(300, True) == 300.0
    assert normal_ers_power_limit_kw(300, False) == 250.0


def test_override_speed_curve():
    assert override_power_limit_kw(300) == 350.0
    assert override_power_limit_kw(355) == 0.0


def test_overtake_requires_detection_or_existing_arm():
    req = load()
    a = assess(req, ERSTacticalAction.OVERTAKE)
    assert not a.available
    req.track.detection_point_active = True
    req.target.gap_s = 0.8
    a = assess(req, ERSTacticalAction.OVERTAKE)
    assert a.available
    assert a.extra_overtake_energy_mj == 0.5


def test_quantum_inspired_encoding_is_normalized():
    result = estimate([
        {"passed": True, "counterattacked": False, "reserve_breached": False, "utility": 1.0},
        {"passed": False, "counterattacked": True, "reserve_breached": False, "utility": -1.0},
    ])
    assert result.success_probability == 0.5
    assert result.amplitude_success == 0.707107
    assert not result.quantum_hardware_used

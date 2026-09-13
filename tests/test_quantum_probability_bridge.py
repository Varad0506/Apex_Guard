import pytest


def test_quantum_bridge_uses_logistic_probabilities(monkeypatch):
    pytest.importorskip("qiskit")

    from app.models import quantum_probability_bridge as bridge
    from app.schemas.common import ActionType

    monkeypatch.setattr(
        bridge,
        "predict_for_actions",
        lambda **kwargs: {
            ActionType.HOLD: 0.713,
            ActionType.FULL_DEPLOY: 0.821,
        },
    )

    class DummyResult:
        estimate = 0.8125
        absolute_error = 0.0085
        oracle_calls = 63
        runtime_ms = 1.0

    monkeypatch.setattr(
        bridge,
        "simulate_quantum_amplitude",
        lambda **kwargs: DummyResult(),
    )

    result = bridge.estimate_quantum_from_logistic(
        req=None,
        rival=None,
        traffic=None,
        legal_actions=[ActionType.HOLD, ActionType.FULL_DEPLOY],
    )

    assert result["source"]["model"] == "ApexGuard calibrated Logistic Regression"
    assert result["source"]["probabilities_by_action"]["HOLD"] == pytest.approx(0.713)
    assert result["source"]["probabilities_by_action"]["FULL_DEPLOY"] == pytest.approx(0.821)
    assert result["quantum"]["estimates_by_action"]["FULL_DEPLOY"]["quantum_estimate"] == pytest.approx(0.8125)
    assert result["metadata"]["quantum_hardware_used"] is False
    assert result["metadata"]["production_path_changed"] is False

import json
from pathlib import Path

import pytest

from app.engine.traffic_engine import evaluate, evaluate_baseline, evaluate_with_source
from app.models import traffic_graph
from app.models.battle_graph import build_battle_graph
from app.models.rival_estimator import estimate
from app.schemas.telemetry import DecisionRequest


SCENARIO = Path(__file__).resolve().parents[1] / "app" / "data" / "scenarios" / "isolated_pass.json"


def _request():
    return DecisionRequest(**json.loads(SCENARIO.read_text()))


def test_battle_graph_shape_and_gnn_prediction():
    if not traffic_graph.model_is_available():
        pytest.skip("trained traffic_gat.pt artifact is not available")
    req = _request()
    rival = estimate(req)
    graph = build_battle_graph(req, rival)

    assert graph.node_features.shape == (5, 8)
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_features.shape[1] == 7
    assert graph.node_mask.shape == (5,)

    prediction = traffic_graph.predict(graph)
    assert set(prediction) == {"tow_strength", "counterattack_risk", "post_pass_traffic_risk"}
    assert all(0.0 <= value <= 1.0 for value in prediction.values())


def test_decision_traffic_engine_uses_gnn_when_available():
    req = _request()
    rival = estimate(req)
    traffic, source = evaluate_with_source(req, rival)

    if traffic_graph.model_is_available():
        assert source == "gnn"
    else:
        assert source == "baseline"

    assert 0.0 <= traffic.tow_strength <= 1.0
    assert 0.0 <= traffic.counterattack_risk <= 1.0
    assert 0.0 <= traffic.post_pass_traffic_risk <= 1.0


def test_baseline_remains_independent_from_gnn():
    req = _request()
    rival = estimate(req)
    baseline = evaluate_baseline(req, rival)
    assert 0.0 <= baseline.tow_strength <= 1.0
    assert 0.0 <= baseline.counterattack_risk <= 1.0
    assert 0.0 <= baseline.post_pass_traffic_risk <= 1.0

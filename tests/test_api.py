import json
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
SCENARIOS_DIR = Path(__file__).resolve().parents[1] / "app" / "data" / "scenarios"


def test_health():
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health_ready():
    resp = client.get("/v1/health/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


def test_list_scenarios():
    resp = client.get("/v1/scenarios")
    assert resp.status_code == 200
    assert len(resp.json()["scenarios"]) == 3


def test_decide_end_to_end_for_all_scenarios():
    for path in SCENARIOS_DIR.glob("*.json"):
        with path.open() as f:
            payload = json.load(f)
        resp = client.post("/v1/decide", json=payload)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["request_id"] == payload["request_id"]
        assert "recommended_action" in body
        assert "candidates" in body


def test_simulate_endpoint_returns_valid_outcome():
    with (SCENARIOS_DIR / "rear_drs_threat.json").open() as f:
        payload = json.load(f)
    resp = client.post("/v1/simulate", json={
        "state": payload,
        "action": "PARTIAL_DEPLOY",
        "horizon_s": 10.0,
        "n_paths": 15,
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["action"] == "PARTIAL_DEPLOY"
    assert 0.0 <= body["pass_probability"] <= 1.0
    assert 0.0 <= body["confidence"] <= 1.0


def test_audit_lookup_after_decide():
    with (SCENARIOS_DIR / "isolated_pass.json").open() as f:
        payload = json.load(f)
    decide_resp = client.post("/v1/decide", json=payload)
    audit_id = decide_resp.json()["audit_id"]
    audit_resp = client.get(f"/v1/audits/{audit_id}")
    assert audit_resp.status_code == 200
    assert audit_resp.json()["response"]["audit_id"] == audit_id

import json
import time
import uuid
from pathlib import Path
from typing import Optional
from app.schemas.telemetry import DecisionRequest
from app.schemas.decision import DecisionResponse

AUDIT_LOG_PATH = Path(__file__).resolve().parents[1] / "data" / "artifacts" / "audit_log.jsonl"
AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def new_audit_id(request_id: str) -> str:
    return f"ag-{request_id}-{uuid.uuid4().hex[:8]}"


def log_decision(req: DecisionRequest, response: DecisionResponse, reason: Optional[str] = None) -> None:
    record = {
        "logged_at_ms": int(time.time() * 1000),
        "request": req.model_dump(mode="json"),
        "response": response.model_dump(mode="json"),
        "reason": reason,
    }
    with AUDIT_LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


def get_audit_by_id(audit_id: str) -> Optional[dict]:
    if not AUDIT_LOG_PATH.exists():
        return None
    with AUDIT_LOG_PATH.open("r") as f:
        for line in f:
            record = json.loads(line)
            if record["response"]["audit_id"] == audit_id:
                return record
    return None

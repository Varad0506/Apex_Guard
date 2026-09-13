"""Experimental quantum-estimation endpoints.

These endpoints are intentionally separate from the production decision path.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.engine import rule_engine, safety
from app.engine.traffic_engine import evaluate_with_source as evaluate_traffic
from app.models import rival_estimator
from app.models.quantum_probability_bridge import estimate_quantum_from_logistic
from app.schemas.telemetry import DecisionRequest

router = APIRouter(tags=["quantum"])


class QuantumEstimateRequest(BaseModel):
    state: DecisionRequest
    epsilon: float = Field(
        default=0.05,
        gt=0.0,
        le=0.5,
        description="Target additive estimation precision for the toy experiment.",
    )
    evaluation_qubits: int = Field(
        default=6,
        ge=2,
        le=8,
        description="QAE evaluation-register size used by the toy experiment.",
    )


@router.post("/v1/quantum/estimate")
def quantum_estimate(payload: QuantumEstimateRequest) -> dict:
    """Estimate the real ApexGuard Logistic probabilities with the
    experimental Qiskit amplitude-estimation pathway.

    This endpoint reuses the same rival/traffic/action legality inputs as
    /v1/decide, but it never changes or calls the production final-action
    selection path.
    """
    req = payload.state

    if req.telemetry_age_ms > safety.TELEMETRY_STALE_MS:
        raise HTTPException(
            status_code=400,
            detail="telemetry is stale for the quantum research endpoint",
        )

    legal_actions = rule_engine.allowed_actions(req)
    if not legal_actions:
        raise HTTPException(
            status_code=400,
            detail="no legal action is available",
        )

    try:
        rival = rival_estimator.estimate(req)
        traffic, traffic_model = evaluate_traffic(req, rival)

        result = estimate_quantum_from_logistic(
            req=req,
            rival=rival,
            traffic=traffic,
            legal_actions=legal_actions,
            epsilon=payload.epsilon,
            evaluation_qubits=payload.evaluation_qubits,
        )

        result["traffic_model"] = traffic_model
        result["request_id"] = req.request_id
        return result

    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Quantum dependencies are not installed. "
                "Install the backend requirements including Qiskit."
            ),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"quantum estimation failed: {exc}",
        ) from exc

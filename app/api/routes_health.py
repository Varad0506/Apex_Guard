from fastapi import APIRouter
from app.models import policy_adapter, overtake_probability

router = APIRouter(tags=["health"])


@router.get("/v1/health")
def health():
    """Liveness: is the process running?"""
    return {"status": "ok"}


@router.get("/v1/health/ready")
def health_ready():
    """Readiness: are scenarios/models loaded and safe to serve?"""
    return {
        "status": "ready",
        "policy_available": policy_adapter.policy_is_available(),
        "overtake_model": "trained-v1" if overtake_probability.model_is_available() else "heuristic-v0",
        "traffic_engine": "deterministic-v0",
    }

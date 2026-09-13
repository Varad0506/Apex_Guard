from fastapi import APIRouter
from app.schemas.telemetry import DecisionRequest
from app.schemas.decision import DecisionResponse
from app.engine.decision_engine import decide

router = APIRouter(tags=["decision"])


@router.post("/v1/decide", response_model=DecisionResponse)
def post_decide(request: DecisionRequest) -> DecisionResponse:
    return decide(request)

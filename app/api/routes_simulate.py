from fastapi import APIRouter
from pydantic import BaseModel
from app.schemas.telemetry import DecisionRequest
from app.schemas.common import ActionType, ERSTacticalAction
from app.engine.traffic_engine import evaluate as evaluate_traffic
from app.models import rival_estimator
from app.simulation.engine import rollout_verified, _run_verified_paths, _ACTION_BASE_TIME_GAIN
from app.engine.ers_2026 import assess_all, assess
from app.simulation.quantum_inspired import estimate as quantum_estimate

router = APIRouter(tags=["simulate"])


class SimulateRequest(BaseModel):
    state: DecisionRequest
    action: ActionType
    horizon_s: float = 12.0
    n_paths: int = 30


class SimulateResponse(BaseModel):
    action: ActionType
    pass_probability: float
    confidence: float
    immediate_time_delta_s: float
    projected_soc_pct: float
    expected_net_value: float


@router.post("/v1/simulate", response_model=SimulateResponse)
def post_simulate(payload: SimulateRequest) -> SimulateResponse:
    """Run a full what-if Monte Carlo rollout for a single action, outside
    the normal /v1/decide ranking flow. Useful for exploring the decision
    space interactively (e.g. a frontend slider over horizon_s or n_paths)
    without going through the whole pipeline each time."""
    req = payload.state
    rival = rival_estimator.estimate(req)
    traffic = evaluate_traffic(req, rival)
    outcome = rollout_verified(
        req, payload.action, traffic, rival,
        horizon_s=payload.horizon_s, n_paths=payload.n_paths,
    )
    return SimulateResponse(
        action=outcome.action,
        pass_probability=outcome.pass_probability,
        confidence=outcome.confidence,
        immediate_time_delta_s=outcome.immediate_time_delta_s,
        projected_soc_pct=outcome.projected_soc_pct,
        expected_net_value=outcome.expected_net_value,
    )


class ERSTacticalRequest(BaseModel):
    state: DecisionRequest


class ERSTacticalResponse(BaseModel):
    actions: list[dict]
    advisory_only: bool = True
    regulation_basis: str = "2026 F1 ERS terminology/rule-inspired advisory model; verify against the latest FIA Sporting/Technical Regulations before race deployment."


@router.post("/v1/ers/tactical", response_model=ERSTacticalResponse)
def ers_tactical(payload: ERSTacticalRequest) -> ERSTacticalResponse:
    assessments = assess_all(payload.state)
    return ERSTacticalResponse(actions=[a.__dict__ for a in assessments])


class QuantumInspiredRequest(BaseModel):
    state: DecisionRequest
    action: ActionType
    horizon_s: float = 8.0
    n_paths: int = 60


class QuantumInspiredResponse(BaseModel):
    action: ActionType
    classical_monte_carlo: dict
    quantum_inspired: dict
    note: str


@router.post("/v1/simulate/quantum-inspired", response_model=QuantumInspiredResponse)
def quantum_inspired_simulate(payload: QuantumInspiredRequest) -> QuantumInspiredResponse:
    req = payload.state
    rival = rival_estimator.estimate(req)
    traffic = evaluate_traffic(req, rival)
    outcome = rollout_verified(
        req, payload.action, traffic, rival,
        horizon_s=payload.horizon_s, n_paths=payload.n_paths,
    )
    paths = _run_verified_paths(
        req, payload.action, traffic, rival,
        horizon_s=payload.horizon_s, dt_s=0.5, n_paths=payload.n_paths,
    )
    # Use the actual verifier paths for the experimental probability encoding.
    # The seed is deterministic for a given request/action, so the UI can replay
    # the same distribution.
    rows = []
    for path in paths:
        if path.passed:
            gain_fraction = max(0.0, (payload.horizon_s - path.pass_tick_s) / payload.horizon_s)
            time_gain = _ACTION_BASE_TIME_GAIN[payload.action] * (0.5 + 0.5 * gain_fraction)
        else:
            time_gain = min(0.0, _ACTION_BASE_TIME_GAIN[payload.action]) * 0.3
        utility = (
            (1.0 if path.passed else 0.0) * time_gain
            - path.budget_kj_used / 3500.0
            - traffic.counterattack_risk * 0.15
            - (0.18 if path.counterattacked else 0.0)
            - (0.6 if path.reserve_breached else 0.0)
        )
        rows.append({
            "passed": path.passed,
            "counterattacked": path.counterattacked,
            "reserve_breached": path.reserve_breached,
            "utility": utility,
        })
    q = quantum_estimate(rows)
    return QuantumInspiredResponse(
        action=outcome.action,
        classical_monte_carlo={
            "sample_count": len(paths),
            "pass_probability": outcome.pass_probability,
            "counterattack_probability": outcome.counterattack_rate,
            "reserve_breach_probability": outcome.reserve_breach_rate,
            "expected_utility": outcome.expected_net_value,
        },
        quantum_inspired=q.__dict__,
        note="Experimental quantum-inspired visualization only: it encodes the actual classical Monte Carlo path distribution as amplitude-like sqrt(probability) values. No quantum hardware or QAE speedup is claimed.",
    )

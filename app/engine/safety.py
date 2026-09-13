"""Fail-safe contract. Every branch here must emit a human-readable reason
so auditability is automatic, not bolted on later."""

import time
from app.schemas.common import ActionType, RiskLevel, DecisionStatus, CandidateStatus
from app.schemas.decision import DecisionResponse, TrafficSummary
from app.schemas.telemetry import DecisionRequest
from app.engine import audit

TELEMETRY_STALE_MS = 500
OPPORTUNITY_THRESHOLD = 0.20
MINIMUM_VALUE = -0.15  # net value below this is treated as "not worth it"


def safe_hold(req: DecisionRequest, reason: str, start_time: float) -> DecisionResponse:
    latency_ms = (time.perf_counter() - start_time) * 1000
    response = DecisionResponse(
        request_id=req.request_id,
        status=DecisionStatus.SAFE_HOLD,
        policy_proposal=None,
        recommended_action=ActionType.HOLD,
        recommendation_confidence=1.0,
        overtake_success_probability=0.0,
        counterattack_risk=0.0,
        risk_level=RiskLevel.LOW,
        rule_compliant=True,
        fallback_used=True,
        latency_ms=round(latency_ms, 2),
        explanation=f"Safe hold: {reason}",
        traffic_summary=TrafficSummary(
            target_gap_s=req.target.gap_s,
            rear_gap_s=req.traffic.rear_gap_s,
            tow_strength=0.0,
            post_pass_traffic_risk=0.0,
        ),
        candidates=[],
        audit_id=audit.new_audit_id(req.request_id),
    )
    audit.log_decision(req, response, reason=reason)
    return response


def low_opportunity_hold(req: DecisionRequest, opportunity_score: float, start_time: float) -> DecisionResponse:
    return safe_hold(
        req,
        f"No credible overtake opportunity (score={opportunity_score:.2f} < {OPPORTUNITY_THRESHOLD})",
        start_time,
    )

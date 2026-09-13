from typing import List, Optional
from pydantic import BaseModel, Field
from .common import ActionType, RiskLevel, DecisionStatus, CandidateStatus


class CandidateOutcome(BaseModel):
    action: ActionType
    legal: bool
    pass_probability: float
    immediate_time_delta_s: float
    projected_soc_pct: float
    expected_net_value: float
    status: CandidateStatus
    reason: str


class TrafficSummary(BaseModel):
    target_gap_s: float
    rear_gap_s: float
    tow_strength: float
    post_pass_traffic_risk: float


class DecisionResponse(BaseModel):
    request_id: str
    status: DecisionStatus
    policy_proposal: Optional[ActionType] = None
    recommended_action: ActionType
    recommendation_confidence: float
    overtake_success_probability: float
    counterattack_risk: float
    risk_level: RiskLevel
    rule_compliant: bool
    fallback_used: bool
    latency_ms: float
    traffic_model: str = "baseline"
    opponent_belief: dict = Field(default_factory=dict)
    overtake_intelligence: dict = Field(default_factory=dict)
    explanation: str
    traffic_summary: TrafficSummary
    candidates: List[CandidateOutcome]
    audit_id: str

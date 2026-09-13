from pydantic import BaseModel, Field
from .common import DecisionMode


class EgoState(BaseModel):
    soc_pct: float = Field(..., ge=0, le=100, description="Ego state-of-charge percent")
    speed_kph: float = Field(..., ge=0)
    throttle_pct: float = Field(..., ge=0, le=100)
    brake_pct: float = Field(..., ge=0, le=100)
    tyre_grip_estimate: float = Field(..., ge=0, le=1)
    laps_remaining: int = Field(..., ge=0)
    current_mode: str


class TargetState(BaseModel):
    gap_s: float = Field(..., ge=0)
    relative_speed_kph: float
    stint_age_laps: int = Field(..., ge=0)
    recent_sector_delta_s: float


class TrafficState(BaseModel):
    rear_gap_s: float = Field(..., ge=0)
    cars_within_3s: int = Field(..., ge=0)
    post_pass_traffic_gap_s: float = Field(..., ge=0)


class TrackState(BaseModel):
    track_id: str
    segment_type: str
    drs_available: bool
    straight_remaining_m: float = Field(..., ge=0)
    braking_zone_m: float = Field(..., ge=0)
    overtake_difficulty: float = Field(..., ge=0, le=1)
    # 2026 ERS tactical context. These are advisory-state flags, not direct
    # car-control commands. They default safely for existing scenarios.
    detection_point_active: bool = False
    overtake_mode_available: bool = False
    ers_key_acceleration_zone: bool = False


class RulesState(BaseModel):
    deployment_budget_remaining_kj: float = Field(..., ge=0)
    minimum_reserve_soc_pct: float = Field(..., ge=0, le=100)
    full_deploy_allowed: bool


class DecisionRequest(BaseModel):
    request_id: str
    battle_id: str | None = None
    timestamp_ms: int
    telemetry_age_ms: int = Field(..., ge=0)
    ego: EgoState
    target: TargetState
    traffic: TrafficState
    track: TrackState
    rules: RulesState
    decision_mode: DecisionMode = DecisionMode.VERIFIED

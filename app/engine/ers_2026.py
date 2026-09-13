"""2026-F1-inspired ERS tactical advisory layer.

This module deliberately models *decision eligibility* rather than direct
vehicle control. It is based on the published 2026 terminology/rules: Boost
is driver-operated, Overtake Mode is conditional on the one-second detection
criterion, and current FIA refinements distinguish 350 kW key acceleration /
overtaking zones from 250 kW elsewhere. Exact team maps and homologated SECU
logic are outside this prototype.
"""
from dataclasses import dataclass
from typing import Optional

from app.schemas.common import ERSTacticalAction
from app.schemas.telemetry import DecisionRequest

MAX_ERS_KW = 350.0
RACE_BOOST_EXTRA_KW = 150.0
KEY_ZONE_ERS_KW = 350.0
OTHER_ZONE_ERS_KW = 250.0
OVERTAKE_DETECTION_GAP_S = 1.0
EXTRA_OVERTAKE_ENERGY_MJ = 0.5
MAX_SOC_SWING_MJ = 4.0


@dataclass(frozen=True)
class ERSTacticalAssessment:
    action: ERSTacticalAction
    available: bool
    advisory_only: bool
    reason: str
    max_ers_power_kw: float
    boost_extra_power_kw: float
    overtake_mode_available: bool
    overtake_detection_eligible: bool
    extra_overtake_energy_mj: float
    reserve_headroom_pct: float


def normal_ers_power_limit_kw(speed_kph: float, key_acceleration_zone: bool = False) -> float:
    """Return a conservative current-2026 race advisory cap.

    The FIA's 2026 refinements retain 350 kW in key acceleration/overtaking
    zones and limit deployment to 250 kW elsewhere. The underlying technical
    regulations also impose speed-dependent limits, so we additionally cap the
    result using the published speed curve.
    """
    speed = max(0.0, float(speed_kph))
    if speed >= 345.0:
        speed_cap = 0.0
    elif speed >= 340.0:
        speed_cap = 6900.0 - 20.0 * speed
    else:
        speed_cap = 1800.0 - 5.0 * speed
    zone_cap = KEY_ZONE_ERS_KW if key_acceleration_zone else OTHER_ZONE_ERS_KW
    return round(max(0.0, min(MAX_ERS_KW, zone_cap, speed_cap)), 1)


def override_power_limit_kw(speed_kph: float) -> float:
    """Published technical-regulation override ceiling, before team/race logic."""
    speed = max(0.0, float(speed_kph))
    if speed >= 355.0:
        return 0.0
    return round(max(0.0, min(MAX_ERS_KW, 7100.0 - 20.0 * speed)), 1)


def assess(req: DecisionRequest, action: ERSTacticalAction) -> ERSTacticalAssessment:
    ego = req.ego
    track = req.track
    reserve = req.rules.minimum_reserve_soc_pct
    headroom = max(0.0, ego.soc_pct - reserve)
    fresh = req.telemetry_age_ms <= 500
    power = normal_ers_power_limit_kw(ego.speed_kph, track.ers_key_acceleration_zone)

    if action == ERSTacticalAction.HOLD:
        return ERSTacticalAssessment(action, True, True, "No ERS deployment requested.", power, 0.0, track.overtake_mode_available, False, 0.0, headroom)

    if not fresh:
        return ERSTacticalAssessment(action, False, True, "Telemetry is stale; aggressive ERS advice is blocked.", power, 0.0, False, False, 0.0, headroom)

    if action == ERSTacticalAction.RECHARGE:
        ok = ego.soc_pct < 99.5
        reason = "Recharge strategy available; actual harvesting depends on track/engine map." if ok else "Energy store is effectively full."
        return ERSTacticalAssessment(action, ok, True, reason, power, 0.0, track.overtake_mode_available, False, 0.0, headroom)

    if action == ERSTacticalAction.BOOST:
        ok = headroom > 0.0 and req.rules.deployment_budget_remaining_kj > 0.0 and power > 0.0
        reason = "Boost advisory is feasible with available charge and ERS power." if ok else "Insufficient reserve headroom, budget, or permitted ERS power."
        return ERSTacticalAssessment(action, ok, True, reason, power, RACE_BOOST_EXTRA_KW if ok else 0.0, track.overtake_mode_available, False, 0.0, headroom)

    # Overtake Mode is only armed after the one-second detection criterion.
    eligible = track.detection_point_active and req.target.gap_s <= OVERTAKE_DETECTION_GAP_S
    available = bool(track.overtake_mode_available or eligible) and headroom > 0.0 and power > 0.0
    if available:
        reason = "Overtake Mode advisory is available/armed under the supplied detection state."
    elif not eligible and not track.overtake_mode_available:
        reason = "Overtake Mode is not armed: the supplied state is not within the one-second detection condition."
    else:
        reason = "Overtake Mode blocked by energy or ERS constraints."
    return ERSTacticalAssessment(
        action, available, True, reason, power, RACE_BOOST_EXTRA_KW if available else 0.0,
        bool(track.overtake_mode_available), eligible, EXTRA_OVERTAKE_ENERGY_MJ if available else 0.0, headroom,
    )


def assess_all(req: DecisionRequest) -> list[ERSTacticalAssessment]:
    return [assess(req, action) for action in ERSTacticalAction]

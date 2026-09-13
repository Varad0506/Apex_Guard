"""Pass-probability model. Phase 2 upgrade: if a trained, calibrated
classifier artifact exists (produced by scripts/train_overtake_model.py),
load and use it. If it's missing or fails to load, fall back to the
Phase 1 explainable logistic-style heuristic -- this is a "Model missing"
fail-safe branch, not a hard failure, per the fail-safe contract."""

import math
from pathlib import Path
from typing import Dict, List, Optional

import joblib

from app.schemas.common import ActionType
from app.schemas.telemetry import DecisionRequest
from app.models.rival_estimator import RivalEstimate
from app.engine.traffic_engine import TrafficAssessment

# Per-action boost to pass likelihood from extra energy deployed. Used only
# by the heuristic fallback path.
_ACTION_BOOST = {
    ActionType.HARVEST: -1.2,
    ActionType.HOLD: -0.2,
    ActionType.PARTIAL_DEPLOY: 0.9,
    ActionType.FULL_DEPLOY: 1.6,
}

_ACTION_ENERGY_LEVEL = {
    ActionType.HARVEST: -1,
    ActionType.HOLD: 0,
    ActionType.PARTIAL_DEPLOY: 1,
    ActionType.FULL_DEPLOY: 2,
}

_MODEL_PATH = Path(__file__).resolve().parents[1] / "data" / "artifacts" / "overtake_model.joblib"

_model_bundle: Optional[dict] = None
_model_load_attempted = False


def _try_load_model() -> Optional[dict]:
    global _model_bundle, _model_load_attempted
    if _model_load_attempted:
        return _model_bundle
    _model_load_attempted = True
    if _MODEL_PATH.exists():
        try:
            _model_bundle = joblib.load(_MODEL_PATH)
        except Exception:
            # Corrupt/incompatible artifact -> treat exactly like "missing".
            _model_bundle = None
    return _model_bundle


def model_is_available() -> bool:
    return _try_load_model() is not None


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _heuristic_predict(
    req: DecisionRequest,
    rival: RivalEstimate,
    traffic: TrafficAssessment,
    legal_actions: List[ActionType],
) -> Dict[ActionType, float]:
    target = req.target
    track = req.track

    gap_term = -1.4 * target.gap_s
    closing_term = 0.03 * target.relative_speed_kph
    drs_term = 0.9 if track.drs_available else 0.0
    difficulty_term = -1.1 * track.overtake_difficulty
    grip_term = 0.6 * (req.ego.tyre_grip_estimate - rival.tyre_grip_estimate)
    straight_term = 0.4 * min(1.0, track.straight_remaining_m / 400.0)
    tow_term = 0.5 * traffic.tow_strength

    base = -0.2 + gap_term + closing_term + drs_term + difficulty_term + grip_term + straight_term + tow_term

    probabilities = {}
    for action in legal_actions:
        logit = base + _ACTION_BOOST.get(action, 0.0)
        probabilities[action] = round(_sigmoid(logit), 3)
    return probabilities


def _model_predict(
    bundle: dict,
    req: DecisionRequest,
    rival: RivalEstimate,
    traffic: TrafficAssessment,
    legal_actions: List[ActionType],
) -> Dict[ActionType, float]:
    import pandas as pd  # local import: only needed on the trained-model path

    pipeline = bundle["pipeline"]
    feature_columns = bundle["feature_columns"]

    rows = []
    for action in legal_actions:
        rows.append({
            "target_gap_s": req.target.gap_s,
            "relative_speed_kph": req.target.relative_speed_kph,
            "drs_available": int(req.track.drs_available),
            "straight_remaining_m": req.track.straight_remaining_m,
            "braking_zone_m": req.track.braking_zone_m,
            "track_overtake_difficulty": req.track.overtake_difficulty,
            "ego_tyre_grip_estimate": req.ego.tyre_grip_estimate,
            "target_tyre_grip_estimate": rival.tyre_grip_estimate,
            "target_tyre_uncertainty": rival.tyre_uncertainty,
            "laps_remaining": req.ego.laps_remaining,
            "rear_gap_s": req.traffic.rear_gap_s,
            "rear_drs_risk": max(0.0, 1.0 - req.traffic.rear_gap_s / 1.5),
            "cars_within_3s": req.traffic.cars_within_3s,
            "post_pass_traffic_gap_s": req.traffic.post_pass_traffic_gap_s,
            "recent_sector_delta_s": req.target.recent_sector_delta_s,
            "action_energy_level": _ACTION_ENERGY_LEVEL[action],
        })

    frame = pd.DataFrame(rows)[feature_columns]
    proba = pipeline.predict_proba(frame)[:, 1]

    return {action: round(float(p), 3) for action, p in zip(legal_actions, proba)}


def predict_for_actions(
    req: DecisionRequest,
    rival: RivalEstimate,
    traffic: TrafficAssessment,
    legal_actions: List[ActionType],
) -> Dict[ActionType, float]:
    bundle = _try_load_model()
    if bundle is not None:
        try:
            return _model_predict(bundle, req, rival, traffic, legal_actions)
        except Exception:
            # Any inference-time failure degrades to the heuristic rather
            # than raising -- matches "Model missing -> Degraded verified
            # mode", never a 500 to the frontend.
            pass
    return _heuristic_predict(req, rival, traffic, legal_actions)


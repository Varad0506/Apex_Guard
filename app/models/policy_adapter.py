"""Proposes candidate actions to roll out. Phase 4 plugs a trained PPO
policy in here (scripts/train_ppo.py + app/rl/apexguard_env.py). Until a
trained artifact exists -- and always, as a fallback if loading fails --
this proposes None, which decision_engine.py interprets as "evaluate every
legal action" instead of trusting a single proposal.

Key design rule, unchanged by adding PPO: the verifier (rollout_fast /
rollout_verified) -- not this module -- selects the action that reaches
the frontend. PPO's job is only to narrow the candidate set; a bad or
missing PPO policy degrades to "evaluate everything," never to a wrong
action reaching the car."""

from pathlib import Path
from typing import List, Optional

import numpy as np

from app.schemas.common import ActionType
from app.schemas.telemetry import DecisionRequest
from app.models import opponent_belief, rival_estimator, overtake_confidence

_PPO_MODEL_PATH = Path(__file__).resolve().parents[1] / "data" / "artifacts" / "ppo_policy.zip"
_ppo_model = None
_ppo_load_attempted = False

# Must match app/rl/apexguard_env.py's ACTIONS ordering exactly.
_ACTIONS = [ActionType.HARVEST, ActionType.HOLD, ActionType.PARTIAL_DEPLOY, ActionType.FULL_DEPLOY]


def _try_load_ppo():
    global _ppo_model, _ppo_load_attempted
    if _ppo_load_attempted:
        return _ppo_model
    _ppo_load_attempted = True
    if not _PPO_MODEL_PATH.exists():
        return None
    try:
        from stable_baselines3 import PPO  # heavy import; only pay for it if the artifact exists
        _ppo_model = PPO.load(str(_PPO_MODEL_PATH))
    except Exception:
        # Missing stable-baselines3, corrupt artifact, version mismatch --
        # all treated identically to "no policy available".
        _ppo_model = None
    return _ppo_model


def _observation_from_request(req: DecisionRequest) -> np.ndarray:
    """Build the same 27-dim observation vector used by ApexGuardEnv.

    The first 11 values are normalized live telemetry. The next 13 values
    are the temporal opponent-belief features. The final 3 values are the calibrated logistic confidence, normalized action entropy, and corridor-open predicate, so a trained policy sees the
    same information distribution at inference time as it saw during PPO
    training.
    """
    ego, target, traffic, track, rules = req.ego, req.target, req.traffic, req.track, req.rules
    rival = rival_estimator.estimate(req)
    battle_id = req.battle_id or f"live-{req.request_id}"
    belief = opponent_belief.update_temporal(req, rival, battle_id)

    raw = np.array([
        target.gap_s / 2.0,
        target.relative_speed_kph / 30.0,
        ego.soc_pct / 100.0,
        (ego.soc_pct - rules.minimum_reserve_soc_pct) / 100.0,
        rules.deployment_budget_remaining_kj / 2200.0,
        track.overtake_difficulty,
        rival.defensive_likelihood,
        traffic.rear_gap_s / 3.0,
        traffic.cars_within_3s / 5.0,
        1.0 if (track.drs_available and track.segment_type == "DRS_STRAIGHT") else 0.0,
        ego.laps_remaining / 55.0,
    ], dtype=np.float32)

    belief_obs = np.array([
        belief.soc_belief.get("LOW", 0.0),
        belief.soc_belief.get("MEDIUM", 0.0),
        belief.soc_belief.get("HIGH", 0.0),
        belief.override_belief.get("AVAILABLE", 0.0),
        belief.override_belief.get("UNAVAILABLE", 0.0),
        belief.tactical_belief.get("HARVESTING", 0.0),
        belief.tactical_belief.get("DEFENDING", 0.0),
        belief.tactical_belief.get("ATTACKING", 0.0),
        belief.tactical_belief.get("CONSERVING", 0.0),
        belief.aero_belief.get("LOW_DRAG", 0.0),
        belief.aero_belief.get("NORMAL", 0.0),
        belief.counter_harvest_trap_probability,
        belief.confidence,
    ], dtype=np.float32)

    legal_actions = [a for a in _ACTIONS if a in (
        ActionType.HARVEST, ActionType.HOLD, ActionType.PARTIAL_DEPLOY, ActionType.FULL_DEPLOY
    )]
    from app.engine.traffic_engine import evaluate_baseline
    traffic_assessment = evaluate_baseline(req, rival, belief)
    intelligence = overtake_confidence.assess(req, rival, traffic_assessment, legal_actions, belief)
    confidence_obs = np.array([
        intelligence["confidence"],
        intelligence["normalized_entropy"],
        intelligence["predicates"]["corridor_open"],
    ], dtype=np.float32)

    obs = np.concatenate([raw, belief_obs, confidence_obs])
    assert obs.shape == (27,), f"PPO observation mismatch: {obs.shape} != (27,)"
    return np.clip(obs, -5.0, 5.0)


def policy_is_available() -> bool:
    return _try_load_ppo() is not None


def propose(req: DecisionRequest, legal_actions: List[ActionType]) -> Optional[ActionType]:
    """Returns a single proposed action from the trained PPO policy, or
    None if no policy is available -- the safe fallback that makes
    decision_engine.py evaluate every legal action instead."""
    model = _try_load_ppo()
    if model is None:
        return None
    try:
        obs = _observation_from_request(req)
        action_idx, _ = model.predict(obs, deterministic=True)
        proposed = _ACTIONS[int(action_idx)]
        return proposed if proposed in legal_actions else None
    except Exception:
        # Any inference-time failure degrades to "no proposal" rather than
        # raising -- same fail-safe pattern as overtake_probability.py.
        return None

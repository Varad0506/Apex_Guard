"""Temporal opponent hidden-state belief model.

The rival's battery SOC, Override availability and active-aero state are not
observed directly.  ApexGuard therefore keeps a probabilistic belief over
those hidden states and updates it as new telemetry arrives.

The implementation is deliberately lightweight HMM-style filtering:

    prior_t = T @ belief_{t-1}
    belief_t ∝ prior_t * P(observation_t | state_t)

The observation model is the existing deterministic telemetry-to-likelihood
model.  This gives the backend temporal memory without pretending that hidden
telemetry is directly measurable.
"""

from dataclasses import dataclass
from math import exp
from typing import Dict, Optional

from app.schemas.telemetry import DecisionRequest
from app.models.rival_estimator import RivalEstimate
from app.models import rival_pattern_model


def _sigmoid(x: float) -> float:
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + exp(-x))


def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
    m = max(scores.values())
    exps = {k: exp(max(-20.0, min(20.0, v - m))) for k, v in scores.items()}
    total = sum(exps.values()) or 1.0
    return {k: v / total for k, v in exps.items()}


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


@dataclass
class OpponentBeliefState:
    soc_belief: Dict[str, float]
    override_belief: Dict[str, float]
    tactical_belief: Dict[str, float]
    aero_belief: Dict[str, float]
    counter_harvest_trap_probability: float
    confidence: float
    update_count: int = 0
    temporal: bool = False
    racer_pattern: Dict[str, object] = None

    @property
    def override_available_probability(self) -> float:
        return self.override_belief.get("AVAILABLE", 0.0)

    @property
    def low_drag_probability(self) -> float:
        return self.aero_belief.get("LOW_DRAG", 0.0)

    @property
    def harvesting_probability(self) -> float:
        return self.tactical_belief.get("HARVESTING", 0.0)

    @property
    def defensive_probability(self) -> float:
        return self.tactical_belief.get("DEFENDING", 0.0)

    def summary(self) -> dict:
        return {
            "soc_belief": {k: round(v, 3) for k, v in self.soc_belief.items()},
            "override_belief": {k: round(v, 3) for k, v in self.override_belief.items()},
            "tactical_belief": {k: round(v, 3) for k, v in self.tactical_belief.items()},
            "aero_belief": {k: round(v, 3) for k, v in self.aero_belief.items()},
            "counter_harvest_trap_probability": round(self.counter_harvest_trap_probability, 3),
            "confidence": round(self.confidence, 3),
            "update_count": self.update_count,
            "temporal": self.temporal,
            "racer_pattern": self.racer_pattern or {
                "model": "deterministic-fallback",
                "lstm_available": False,
                "hmm_transition_available": False,
                "history_window": 0,
            },
        }


def _observation_belief(req: DecisionRequest, rival: RivalEstimate) -> OpponentBeliefState:
    """Build the instantaneous observation likelihood from current telemetry."""
    target = req.target
    traffic = req.traffic
    track = req.track

    degradation_signal = _clamp(
        0.5 + (-target.recent_sector_delta_s * 2.0) + target.stint_age_laps * 0.012
    )
    fresh_pace_signal = _clamp(
        0.5 - target.recent_sector_delta_s * 1.5 - target.stint_age_laps * 0.008
    )

    low_soc_score = 1.15 * degradation_signal + 0.25 * _clamp(target.gap_s / 2.0)
    high_soc_score = 1.05 * fresh_pace_signal + 0.15 * (1.0 - _clamp(target.gap_s / 2.0))
    medium_soc_score = 0.55 + 0.35 * rival.tyre_uncertainty
    soc_belief = _softmax({
        "LOW": low_soc_score,
        "MEDIUM": medium_soc_score,
        "HIGH": high_soc_score,
    })

    battle_pressure = _clamp(
        0.55 * _clamp((1.5 - target.gap_s) / 1.5)
        + 0.25 * _clamp(target.relative_speed_kph / 20.0)
        + 0.20 * _clamp(traffic.rear_gap_s / 2.0)
    )
    override_score = 1.15 * soc_belief["HIGH"] + 0.65 * battle_pressure
    no_override_score = 0.95 * soc_belief["LOW"] + 0.20 * rival.tyre_uncertainty
    override_belief = _softmax({
        "AVAILABLE": override_score,
        "UNAVAILABLE": no_override_score,
    })

    speed_concession = _sigmoid(-target.relative_speed_kph / 5.0)
    healthy_target = _clamp(0.65 * rival.tyre_grip_estimate + 0.35 * fresh_pace_signal)
    close_battle = _clamp((1.4 - target.gap_s) / 1.4)

    harvesting_score = (
        1.15 * speed_concession
        + 0.85 * healthy_target
        + 0.30 * (1.0 - degradation_signal)
    )
    defending_score = 1.0 * rival.defensive_likelihood + 0.75 * close_battle + 0.25 * battle_pressure
    attacking_score = (
        0.9 * fresh_pace_signal
        + 0.55 * _clamp(target.relative_speed_kph / 15.0)
        + 0.35 * battle_pressure
    )
    conserving_score = 0.55 * speed_concession + 0.45 * soc_belief["HIGH"]
    tactical_belief = _softmax({
        "HARVESTING": harvesting_score,
        "DEFENDING": defending_score,
        "ATTACKING": attacking_score,
        "CONSERVING": conserving_score,
    })

    aero_opportunity = 1.0 if track.segment_type in {"STRAIGHT", "DRS_STRAIGHT"} else 0.25
    low_drag_score = 1.0 * speed_concession + 0.9 * healthy_target + 0.6 * aero_opportunity
    normal_aero_score = 0.8 * degradation_signal + 0.35 * (1.0 - aero_opportunity)
    aero_belief = _softmax({
        "LOW_DRAG": low_drag_score,
        "NORMAL": normal_aero_score,
    })

    trap = _clamp(
        tactical_belief["HARVESTING"]
        * aero_belief["LOW_DRAG"]
        * (0.55 + 0.45 * override_belief["AVAILABLE"])
    )

    max_tactical = max(tactical_belief.values())
    max_soc = max(soc_belief.values())
    confidence = _clamp(
        0.35 + 0.35 * max_tactical + 0.30 * max_soc - 0.20 * rival.tyre_uncertainty
    )

    return OpponentBeliefState(
        soc_belief=soc_belief,
        override_belief=override_belief,
        tactical_belief=tactical_belief,
        aero_belief=aero_belief,
        counter_harvest_trap_probability=trap,
        confidence=confidence,
    )


def infer(req: DecisionRequest, rival: RivalEstimate) -> OpponentBeliefState:
    """Backward-compatible stateless inference for callers that need one tick."""
    return _observation_belief(req, rival)


def _predict_distribution(previous: Dict[str, float], persistence: float) -> Dict[str, float]:
    """Simple sticky Markov transition: most mass remains in the same state."""
    states = list(previous)
    n = len(states)
    if n == 1:
        return dict(previous)
    switch = (1.0 - persistence) / (n - 1)
    return {
        state: persistence * previous[state]
        + sum(switch * previous[other] for other in states if other != state)
        for state in states
    }


def _bayes_update(prior: Dict[str, float], likelihood: Dict[str, float]) -> Dict[str, float]:
    posterior = {k: max(1e-9, prior[k]) * max(1e-9, likelihood[k]) for k in prior}
    total = sum(posterior.values()) or 1.0
    return {k: v / total for k, v in posterior.items()}


class OpponentBeliefFilter:
    """Lightweight HMM filter for a single ego-vs-target battle stream."""

    def __init__(self, prior: Optional[OpponentBeliefState] = None, update_count: int = 0):
        self.state = prior
        self.update_count = update_count
        self.racer_history = rival_pattern_model.RivalHistoryWindow()

    def update(self, req: DecisionRequest, rival: RivalEstimate) -> OpponentBeliefState:
        observation = _observation_belief(req, rival)
        if self.state is None:
            self.update_count = 1
            self.state = observation
            self.state.update_count = self.update_count
            self.state.temporal = False
            return self.state

        # Persist an observable telemetry window for the optional learned
        # racer-pattern layer.  The target car's hidden ERS state is never
        # passed in; only public/derived observables are used.
        self.racer_history.push(
            gap_s=req.target.gap_s,
            relative_speed_kph=req.target.relative_speed_kph,
            sector_kind=req.track.segment_type,
            dt_s=0.5,
            throttle=0.0,
            brake=0.0,
            speed_kph=req.ego.speed_kph,
            drs=1.0 if req.track.drs_available else 0.0,
            soc_pct=50.0,
            tyre_grip=rival.tyre_grip_estimate,
            position_delta=0.0,
        )

        # Persistence is deliberately high: hidden opponent state should not
        # flip because of one noisy sector/gap observation.
        soc_prior = _predict_distribution(self.state.soc_belief, 0.94)
        override_prior = _predict_distribution(self.state.override_belief, 0.92)
        tactical_prior = _predict_distribution(self.state.tactical_belief, 0.80)
        aero_prior = _predict_distribution(self.state.aero_belief, 0.86)

        soc = _bayes_update(soc_prior, observation.soc_belief)
        override = _bayes_update(override_prior, observation.override_belief)
        tactical = _bayes_update(tactical_prior, observation.tactical_belief)
        aero = _bayes_update(aero_prior, observation.aero_belief)

        # Optional learned temporal racer-pattern prediction.  It is advisory:
        # the deterministic HMM/observation posterior remains in the blend,
        # and an absent/broken model simply returns None.
        racer_pattern = rival_pattern_model.predict_next_tactical(
            self.racer_history,
            tactical,
        )
        if racer_pattern is not None:
            tactical = _normalize({
                state: 0.65 * racer_pattern[state] + 0.35 * tactical[state]
                for state in tactical
            })

        # The trap is a joint posterior, so it is recomputed from the filtered
        # states rather than averaged from independent per-tick trap scores.
        trap = _clamp(
            tactical["HARVESTING"]
            * aero["LOW_DRAG"]
            * (0.55 + 0.45 * override["AVAILABLE"])
        )
        max_tactical = max(tactical.values())
        max_soc = max(soc.values())
        confidence = _clamp(
            0.35 + 0.35 * max_tactical + 0.30 * max_soc - 0.20 * rival.tyre_uncertainty
        )

        self.update_count += 1
        self.state = OpponentBeliefState(
            soc_belief=soc,
            override_belief=override,
            tactical_belief=tactical,
            aero_belief=aero,
            counter_harvest_trap_probability=trap,
            confidence=confidence,
            update_count=self.update_count,
            temporal=True,
            racer_pattern={
                "model": rival_pattern_model.active_model_kind() or "deterministic-fallback",
                "lstm_available": rival_pattern_model.model_is_available(),
                "hmm_transition_available": rival_pattern_model.hmm_is_available(),
                "history_window": len(self.racer_history.as_list()),
            },
        )
        return self.state


def _normalize(dist: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, v) for v in dist.values()) or 1.0
    return {k: max(0.0, v) / total for k, v in dist.items()}


def forecast_step(
    belief: OpponentBeliefState,
    gap_s: float,
    relative_speed_kph: float,
    sector_kind: str,
    dt_s: float,
    rng=None,
) -> OpponentBeliefState:
    """Predict/update a hidden opponent belief one simulator tick ahead.

    VERIFIED rollouts do not receive future real telemetry, so this is a
    predictive HMM step: sticky hidden-state transitions are combined with
    simulated observations (gap, closing speed and sector type).  A sampled
    hidden state is therefore allowed to change during the rollout instead of
    being frozen for all 12 seconds.
    """
    dt_scale = max(0.25, min(2.0, dt_s / 0.5))
    persistence = {
        "soc": 0.965 ** dt_scale,
        "override": 0.94 ** dt_scale,
        "tactical": 0.88 ** dt_scale,
        "aero": 0.92 ** dt_scale,
    }
    soc_prior = _predict_distribution(belief.soc_belief, persistence["soc"])
    override_prior = _predict_distribution(belief.override_belief, persistence["override"])
    tactical_prior = _predict_distribution(belief.tactical_belief, persistence["tactical"])
    aero_prior = _predict_distribution(belief.aero_belief, persistence["aero"])

    close = _clamp((1.6 - gap_s) / 1.6)
    slowing = _sigmoid(-relative_speed_kph / 4.0)
    straight = 1.0 if sector_kind in {"STRAIGHT", "DRS_STRAIGHT"} else 0.15

    # Forecast observation likelihoods. These are intentionally broad: future
    # simulator observations should nudge the belief, not deterministically
    # reveal the hidden state.
    soc_like = _softmax({
        "LOW": 0.55 * slowing + 0.25 * (1.0 - close),
        "MEDIUM": 0.55 + 0.10 * close,
        "HIGH": 0.65 * (1.0 - slowing) + 0.25 * close,
    })
    tactical_like = _softmax({
        "HARVESTING": 1.05 * slowing + 0.45 * (1.0 - close),
        "DEFENDING": 0.90 * close + 0.35 * (1.0 - slowing),
        "ATTACKING": 0.80 * (1.0 - slowing) + 0.35 * close,
        "CONSERVING": 0.45 * slowing + 0.55 * (1.0 - close),
    })
    aero_like = _softmax({
        "LOW_DRAG": 0.95 * slowing + 0.80 * straight + 0.25 * close,
        "NORMAL": 0.75 * (1.0 - slowing) + 0.55 * (1.0 - straight),
    })

    # Override is latent but coupled to the tactical/energy hypothesis.
    override_like = _softmax({
        "AVAILABLE": 0.95 * soc_prior["HIGH"] + 0.50 * tactical_like["ATTACKING"] + 0.20 * close,
        "UNAVAILABLE": 0.95 * soc_prior["LOW"] + 0.45 * tactical_like["HARVESTING"],
    })

    soc = _bayes_update(soc_prior, soc_like)
    tactical = _bayes_update(tactical_prior, tactical_like)
    aero = _bayes_update(aero_prior, aero_like)
    override = _bayes_update(override_prior, override_like)

    trap = _clamp(
        tactical["HARVESTING"]
        * aero["LOW_DRAG"]
        * (0.55 + 0.45 * override["AVAILABLE"])
    )
    confidence = _clamp(
        0.30
        + 0.32 * max(tactical.values())
        + 0.28 * max(soc.values())
        + 0.10 * max(aero.values())
    )

    return OpponentBeliefState(
        soc_belief=_normalize(soc),
        override_belief=_normalize(override),
        tactical_belief=_normalize(tactical),
        aero_belief=_normalize(aero),
        counter_harvest_trap_probability=trap,
        confidence=confidence,
        update_count=belief.update_count + 1,
        temporal=True,
    )


# In-process battle memory. The API gets a battle_id so separate battles do
# not contaminate one another. This is appropriate for the current prototype;
# production deployment should move this state to Redis/a durable stream store.
_FILTERS: Dict[str, OpponentBeliefFilter] = {}
_MAX_FILTERS = 256


def update_temporal(
    req: DecisionRequest,
    rival: RivalEstimate,
    battle_id: Optional[str] = None,
) -> OpponentBeliefState:
    """Update the belief using the previous tick for this battle when present.

    If no battle_id is supplied, inference remains stateless to preserve safe
    backward compatibility with existing one-shot clients.
    """
    if not battle_id:
        return infer(req, rival)

    belief_filter = _FILTERS.get(battle_id)
    if belief_filter is None:
        if len(_FILTERS) >= _MAX_FILTERS:
            oldest_key = next(iter(_FILTERS))
            _FILTERS.pop(oldest_key, None)
        belief_filter = OpponentBeliefFilter()
        _FILTERS[battle_id] = belief_filter
    return belief_filter.update(req, rival)


def reset_temporal(battle_id: Optional[str] = None) -> None:
    """Reset one battle stream, or all in-process streams when omitted."""
    if battle_id is None:
        _FILTERS.clear()
    else:
        _FILTERS.pop(battle_id, None)

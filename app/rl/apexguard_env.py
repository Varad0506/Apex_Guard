"""Phase 4: a Gymnasium environment for training PPO to propose energy
deployment actions, built directly on top of the same energy_model /
traffic_model physics the VERIFIED Monte Carlo verifier uses
(app/simulation/engine.py). This matters: PPO should learn to propose good
actions in the same reality the verifier checks it against, not a
different, simplified toy environment that happens to share a name.

Episode structure: one episode = one overtake attempt window. At each
decision tick (every `decision_dt_s` seconds, default 1.0s -- coarser than
the physics tick `dt_s` so the agent doesn't have to re-decide every 0.5s),
the agent picks one of the 4 actions. The environment steps the underlying
physics forward by `decision_dt_s`, then returns a reward for that tick.
Episode ends when: the car passes (or gets passed / times out), SOC hits
zero, or the horizon elapses.

Domain randomization: reset() samples a fresh scenario (gap, closing speed,
SOC, rules, track segment, rival tyre/defensive state) from wide but
plausible ranges, plus optionally a synthetic track profile's segment
sequence -- so the trained policy generalizes across track/traffic
conditions instead of memorizing one scenario.
"""

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from app.schemas.common import ActionType, DecisionMode
from app.schemas.telemetry import DecisionRequest
from app.models import opponent_belief, rival_estimator, overtake_confidence
from app.simulation import energy_model, traffic_model
from app.simulation.powertrain_modes import PowertrainModeTracker, throttle_brake_proxy
from app.simulation.tracks import Sector, list_profile_ids, load_profile

ACTIONS = [ActionType.HARVEST, ActionType.HOLD, ActionType.PARTIAL_DEPLOY, ActionType.FULL_DEPLOY]
ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}

# Observation vector layout -- keep this in sync with _observation().
OBS_DIM = 27


@dataclass
class EpisodeState:
    gap_s: float
    relative_speed_kph: float
    soc_pct: float
    laps_remaining: int
    minimum_reserve_soc_pct: float
    deployment_budget_remaining_kj: float
    overtake_difficulty: float
    defensive_likelihood: float
    energy_recovery_factor: float
    rear_gap_s: float
    post_pass_traffic_gap_s: float
    cars_within_3s: int
    target_recent_sector_delta_s: float
    target_stint_age_laps: int
    passed: bool = False
    counterattacked: bool = False
    reserve_breached: bool = False
    t_s: float = 0.0


class ApexGuardEnv(gym.Env):
    """One overtake-attempt episode per reset(). Discrete(4) action space
    matching ActionType. Continuous Box observation space."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        horizon_s: float = 12.0,
        physics_dt_s: float = 0.5,
        decision_dt_s: float = 1.0,
        use_synthetic_tracks: bool = True,
        tight_margin_prob: float = 0.35,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.horizon_s = horizon_s
        self.physics_dt_s = physics_dt_s
        self.decision_dt_s = decision_dt_s
        self.use_synthetic_tracks = use_synthetic_tracks
        # Fraction of episodes deliberately sampled with SOC close to the
        # reserve boundary. Was previously an incidental side effect of
        # uniform sampling (soc<min_reserve+5 -> nudge up) with no control
        # over how often it actually happened; raised to a controllable
        # design parameter after the stress test showed a 4.1%
        # illegal-proposal rate concentrated in exactly these states.
        self.tight_margin_prob = tight_margin_prob

        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(OBS_DIM,), dtype=np.float32
        )

        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._sectors: list[Sector] = []
        self._state: Optional[EpisodeState] = None

        self._track_ids = list_profile_ids() if use_synthetic_tracks else []

    # ------------------------------------------------------------------
    # Domain randomization
    # ------------------------------------------------------------------
    def _sample_sectors(self) -> list[Sector]:
        if self.use_synthetic_tracks and self._track_ids:
            profile = load_profile(self._rng.choice(self._track_ids))
            if profile.sectors:
                return profile.sectors
        # Fallback: generic repeating pattern if no track profiles are
        # available (e.g. scripts/tracks.py hasn't been run yet).
        return [
            Sector(kind="STRAIGHT", length_m=450.0, has_drs=True),
            Sector(kind="BRAKING_ZONE", length_m=130.0),
            Sector(kind="CORNER", length_m=150.0),
        ] * 4

    def _sample_episode_state(self) -> EpisodeState:
        r = self._rng
        min_reserve = r.uniform(10.0, 20.0)

        # Deliberately oversample states with a tight SOC-to-reserve margin.
        # Previously this only happened as a side effect of uniform SOC
        # sampling occasionally landing close to the reserve (soc<min_reserve+5
        # -> nudge up), which meant tight-margin states were underrepresented
        # relative to how often they matter -- exactly the states where the
        # 4.1% illegal-proposal rate concentrated in the stress test. Now a
        # fixed fraction of episodes (tight_margin_prob) is *always* sampled
        # right above the boundary, regardless of what uniform sampling would
        # have produced.
        if r.random() < self.tight_margin_prob:
            # Tight margin: SOC sits within 0-10% of the reserve, never below it at reset, so reserve breaches represent the policy's
            # actual decision to spend through the hard reserve rather than
            # an impossible initial-state violation.
            soc = min_reserve + r.uniform(0.5, 10.0)
            soc = max(min_reserve + 0.5, soc)
        else:
            soc = r.uniform(15.0, 90.0)
            if soc < min_reserve + 5.0:
                soc = min_reserve + r.uniform(5.0, 15.0)

        return EpisodeState(
            gap_s=r.uniform(0.15, 2.0),
            relative_speed_kph=r.uniform(-5.0, 30.0),
            soc_pct=soc,
            laps_remaining=r.randint(1, 55),
            minimum_reserve_soc_pct=min_reserve,
            deployment_budget_remaining_kj=r.uniform(400.0, 2200.0),
            overtake_difficulty=r.uniform(0.2, 0.9),
            defensive_likelihood=r.uniform(0.1, 0.9),
            energy_recovery_factor=r.uniform(0.4, 0.7),
            rear_gap_s=r.uniform(0.2, 3.0),
            post_pass_traffic_gap_s=r.uniform(0.3, 2.5),
            cars_within_3s=r.randint(0, 5),
            target_recent_sector_delta_s=r.uniform(-0.12, 0.20),
            target_stint_age_laps=r.randint(2, 28),
        )

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)
            self._np_rng = np.random.default_rng(seed)
        self._sectors = self._sample_sectors()
        self._state = self._sample_episode_state()
        self._battle_id = f"ppo-env-{id(self)}-{self._state.t_s}-{self._rng.random()}"
        opponent_belief.reset_temporal(self._battle_id)
        self._powertrain_tracker = PowertrainModeTracker()  # fresh per episode, matches self._state
        return self._observation(), {}

    def step(self, action_idx: int):
        assert self._state is not None, "call reset() before step()"
        action = ACTIONS[int(action_idx)]
        s = self._state

        # Action-aware reserve feasibility signal. PPO sees the continuous
        # reserve margin in the observation, but this extra term teaches the
        # discrete policy that an action can be tactically attractive yet
        # physically unaffordable. The live legality mask/verifier remains
        # authoritative outside the training environment.
        reserve_margin_before = s.soc_pct - s.minimum_reserve_soc_pct
        required_headroom = {
            ActionType.HARVEST: 0.0,
            ActionType.HOLD: 0.0,
            ActionType.PARTIAL_DEPLOY: 1.5,
            ActionType.FULL_DEPLOY: 3.0,
        }[action]
        reward = 0.0
        if reserve_margin_before < required_headroom:
            deficit = required_headroom - reserve_margin_before
            reward -= min(0.18, 0.04 + 0.025 * deficit)

        n_physics_ticks = max(1, int(round(self.decision_dt_s / self.physics_dt_s)))

        for _ in range(n_physics_ticks):
            sector = self._sector_at(s.t_s)

            throttle_pct, brake_pct = throttle_brake_proxy(sector.kind, action)
            powertrain_state = self._powertrain_tracker.step(
                s.soc_pct, s.minimum_reserve_soc_pct, throttle_pct, brake_pct, action,
            )
            e_result = energy_model.step(
                s.soc_pct, action, self.physics_dt_s, sector, s.energy_recovery_factor,
                deploy_rate_multiplier=powertrain_state.deploy_rate_multiplier,
                harvest_rate_multiplier=powertrain_state.harvest_rate_multiplier,
            )
            s.soc_pct = e_result.soc_pct
            s.deployment_budget_remaining_kj = max(0.0, s.deployment_budget_remaining_kj - e_result.budget_kj_used)
            reserve_margin = s.soc_pct - s.minimum_reserve_soc_pct
            if reserve_margin < 0.0:
                s.reserve_breached = True
                # Strong shaping: once below the hard reserve, every physics
                # tick is costly. This makes reserve protection learnable
                # before the terminal reward is reached.
                reward -= 0.20 + min(0.30, abs(reserve_margin) / 20.0)
            elif reserve_margin < 8.0:
                # Soft warning zone: discourage actions that push the car
                # toward the hard reserve even before a violation occurs.
                reward -= 0.025 * (8.0 - reserve_margin) / 8.0

            if not s.passed:
                # Fold overtake_difficulty into the rival's effective
                # closing resistance so harder tracks need more commitment.
                t_result = traffic_model.step(
                    s.gap_s, s.relative_speed_kph, action, sector, self.physics_dt_s,
                    self._rng, min(0.95, s.defensive_likelihood + s.overtake_difficulty * 0.2),
                )
                s.gap_s = t_result.gap_s
                s.relative_speed_kph = t_result.relative_speed_kph
                if t_result.passed:
                    s.passed = True

            s.t_s += self.physics_dt_s

        terminated = False
        truncated = False

        if s.passed and not s.counterattacked:
            # Check for a counterattack once, right after passing, using the
            # remaining horizon.
            ticks_remaining = int(round((self.horizon_s - s.t_s) / self.physics_dt_s))
            s.counterattacked = traffic_model.post_pass_counterattack_check(
                self._rng, s.defensive_likelihood, s.post_pass_traffic_gap_s,
                max(0, ticks_remaining), self.physics_dt_s,
            )
            terminated = True  # episode ends once the pass outcome (incl. counterattack) is resolved

        if s.t_s >= self.horizon_s:
            truncated = True

        reward += self._terminal_reward(s, action) if (terminated or truncated) else self._shaping_reward(action)

        return self._observation(), reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    # Reward shaping
    # ------------------------------------------------------------------
    def _reserve_shaping_penalty(self, reserve_margin: float) -> float:
        """Progressive reserve penalty without a large discontinuity.

        The policy should learn a smooth preference for margin, rather than
        discovering that every near-reserve state is catastrophically bad.
        This keeps rewards numerically well behaved while still making an
        actual reserve breach clearly worse than a clean tactical loss.
        """
        if reserve_margin >= 15.0:
            return 0.0
        if reserve_margin >= 10.0:
            return 0.01 * (15.0 - reserve_margin) / 5.0
        if reserve_margin >= 5.0:
            return 0.02 + 0.03 * (10.0 - reserve_margin) / 5.0
        if reserve_margin >= 0.0:
            return 0.05 + 0.05 * (5.0 - reserve_margin) / 5.0
        # Breaches are serious, but let the terminal penalty carry most of
        # the signal so individual physics ticks do not dominate learning.
        return 0.10 + min(0.15, abs(reserve_margin) / 20.0)

    def _shaping_reward(self, action: ActionType) -> float:
        """Dense, smoothly scaled reward aligned with verifier priorities.

        Reward hierarchy:
          1. stay legal / preserve reserve,
          2. avoid wasting deployment budget,
          3. still make useful progress toward the overtake.
        """
        s = self._state
        assert s is not None

        cost = {
            ActionType.HARVEST: 0.002,
            ActionType.HOLD: 0.000,
            ActionType.PARTIAL_DEPLOY: -0.010,
            ActionType.FULL_DEPLOY: -0.022,
        }
        reward = cost[action]

        reserve_margin = s.soc_pct - s.minimum_reserve_soc_pct
        reward -= self._reserve_shaping_penalty(reserve_margin)

        # FULL_DEPLOY is only valuable when the car has enough reserve
        # headroom to support it. Keep this penalty modest because the live
        # legality mask/verifier remains authoritative after PPO proposes.
        if action == ActionType.FULL_DEPLOY:
            if reserve_margin < 22.0:
                reward -= 0.05 + 0.08 * max(0.0, (22.0 - reserve_margin) / 22.0)
            if s.deployment_budget_remaining_kj < 260.0:
                reward -= 0.06
        elif action == ActionType.PARTIAL_DEPLOY and s.deployment_budget_remaining_kj < 120.0:
            reward -= 0.04

        return reward

    def _terminal_reward(self, s: EpisodeState, action: ActionType) -> float:
        if s.passed:
            base = {
                ActionType.HARVEST: -0.08,
                ActionType.HOLD: 0.08,
                ActionType.PARTIAL_DEPLOY: 0.75,
                ActionType.FULL_DEPLOY: 0.90,
            }[action]
            reward = base
            if s.counterattacked:
                reward -= 0.45
        else:
            # Missing the pass is undesirable, but should not overwhelm all
            # other information in the trajectory.
            reward = -0.12

        # Reserve violation is the strongest terminal negative signal, but is
        # deliberately finite so PPO can still distinguish *how* bad the
        # preceding action sequence was.
        if s.reserve_breached:
            reserve_deficit = max(0.0, s.minimum_reserve_soc_pct - s.soc_pct)
            reward -= 1.50 + min(0.75, reserve_deficit / 8.0)

        headroom = max(0.0, s.soc_pct - s.minimum_reserve_soc_pct)
        if action == ActionType.FULL_DEPLOY and headroom < 15.0:
            reward -= 0.20

        # Small normalized terminal margin bonus. This is intentionally much
        # smaller than the pass/counterattack terms so reserve is a constraint,
        # not the only objective.
        reward += 0.08 * min(1.0, headroom / 40.0)

        return reward

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _sector_at(self, elapsed_s: float) -> Sector:
        t = 0.0
        idx = 0
        n = len(self._sectors)
        speed_by_kind = {"STRAIGHT": 90.0, "BRAKING_ZONE": 50.0, "CORNER": 40.0}
        guard = 0
        while guard < 500:
            sector = self._sectors[idx % n]
            duration = sector.length_m / speed_by_kind.get(sector.kind, 80.0)
            if t + duration > elapsed_s:
                return sector
            t += duration
            idx += 1
            guard += 1
        return self._sectors[-1]

    def _belief_observation(self):
        s = self._state
        sector = self._sector_at(s.t_s)
        req = DecisionRequest.model_validate({
            "request_id": f"ppo-observation-{id(self)}-{int(s.t_s * 1000)}",
            "battle_id": getattr(self, "_battle_id", None),
            "timestamp_ms": int(s.t_s * 1000),
            "telemetry_age_ms": 50,
            "ego": {
                "soc_pct": s.soc_pct, "speed_kph": 290.0, "throttle_pct": 95.0,
                "brake_pct": 0.0, "tyre_grip_estimate": 0.82,
                "laps_remaining": s.laps_remaining, "current_mode": "HOLD",
            },
            "target": {
                "gap_s": max(0.0, s.gap_s),
                "relative_speed_kph": s.relative_speed_kph,
                "stint_age_laps": s.target_stint_age_laps,
                "recent_sector_delta_s": s.target_recent_sector_delta_s,
            },
            "traffic": {
                "rear_gap_s": s.rear_gap_s, "cars_within_3s": s.cars_within_3s,
                "post_pass_traffic_gap_s": s.post_pass_traffic_gap_s,
            },
            "track": {
                "track_id": "ppo-training", "segment_type": sector.kind,
                "drs_available": sector.has_drs, "straight_remaining_m": 400.0,
                "braking_zone_m": 130.0, "overtake_difficulty": s.overtake_difficulty,
            },
            "rules": {
                "deployment_budget_remaining_kj": s.deployment_budget_remaining_kj,
                "minimum_reserve_soc_pct": s.minimum_reserve_soc_pct,
                "full_deploy_allowed": s.soc_pct >= s.minimum_reserve_soc_pct + 22.0,
            },
            "decision_mode": DecisionMode.VERIFIED,
        })
        rival = rival_estimator.estimate(req)
        belief = opponent_belief.update_temporal(req, rival, self._battle_id)
        return np.array([
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

    def _confidence_observation(self):
        s = self._state
        sector = self._sector_at(s.t_s)
        req = DecisionRequest.model_validate({
            "request_id": f"ppo-confidence-{id(self)}-{int(s.t_s * 1000)}",
            "battle_id": getattr(self, "_battle_id", None),
            "timestamp_ms": int(s.t_s * 1000),
            "telemetry_age_ms": 50,
            "ego": {
                "soc_pct": s.soc_pct, "speed_kph": 290.0, "throttle_pct": 95.0,
                "brake_pct": 0.0, "tyre_grip_estimate": 0.82,
                "laps_remaining": s.laps_remaining, "current_mode": "HOLD",
            },
            "target": {
                "gap_s": max(0.0, s.gap_s), "relative_speed_kph": s.relative_speed_kph,
                "stint_age_laps": s.target_stint_age_laps,
                "recent_sector_delta_s": s.target_recent_sector_delta_s,
            },
            "traffic": {
                "rear_gap_s": s.rear_gap_s, "cars_within_3s": s.cars_within_3s,
                "post_pass_traffic_gap_s": s.post_pass_traffic_gap_s,
            },
            "track": {
                "track_id": "ppo-training", "segment_type": sector.kind,
                "drs_available": sector.has_drs, "straight_remaining_m": 400.0,
                "braking_zone_m": 130.0, "overtake_difficulty": s.overtake_difficulty,
            },
            "rules": {
                "deployment_budget_remaining_kj": s.deployment_budget_remaining_kj,
                "minimum_reserve_soc_pct": s.minimum_reserve_soc_pct,
                "full_deploy_allowed": s.soc_pct >= s.minimum_reserve_soc_pct + 22.0,
            },
            "decision_mode": DecisionMode.VERIFIED,
        })
        rival = rival_estimator.estimate(req)
        traffic = __import__("app.engine.traffic_engine", fromlist=["evaluate_baseline"]).evaluate_baseline(req, rival)
        legal = [a for a in ACTIONS if s.soc_pct >= s.minimum_reserve_soc_pct or a in (ActionType.HARVEST, ActionType.HOLD)]
        intelligence = overtake_confidence.assess(req, rival, traffic, legal or [ActionType.HOLD])
        predicates = intelligence["predicates"]
        return np.array([
            intelligence["confidence"],
            intelligence["normalized_entropy"],
            predicates["corridor_open"],
        ], dtype=np.float32)

    def _observation(self) -> np.ndarray:
        s = self._state
        sector = self._sector_at(s.t_s)
        belief_obs = self._belief_observation()
        confidence_obs = self._confidence_observation()
        raw = np.array([
            s.gap_s / 2.0,
            s.relative_speed_kph / 30.0,
            s.soc_pct / 100.0,
            (s.soc_pct - s.minimum_reserve_soc_pct) / 100.0,
            s.deployment_budget_remaining_kj / 2200.0,
            s.overtake_difficulty,
            s.defensive_likelihood,
            s.rear_gap_s / 3.0,
            s.cars_within_3s / 5.0,
            1.0 if (sector.kind == "STRAIGHT" and sector.has_drs) else 0.0,
            s.laps_remaining / 55.0,
        ], dtype=np.float32)
        obs = np.concatenate([raw, belief_obs, confidence_obs])
        assert obs.shape == (OBS_DIM,), f"PPO observation mismatch: {obs.shape} != {(OBS_DIM,)}"
        return np.clip(obs, -5.0, 5.0)


def make_env(**kwargs):
    """Factory for SB3's vectorized env wrappers, e.g.
    `make_vec_env(lambda: make_env(), n_envs=8)`."""
    return ApexGuardEnv(**kwargs)
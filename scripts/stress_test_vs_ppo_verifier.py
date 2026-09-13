"""Stress-tests the trained PPO policy against the verifier across hundreds
of randomized states and all 10 synthetic track profiles -- the 3 curated
demo scenarios are hand-picked and clear-cut, so they're a poor test of
whether PPO and the verifier actually disagree anywhere. This script builds
DecisionRequest instances the same way ApexGuardEnv.reset() randomizes
training episodes, then reports:

  - illegal_proposal_rate: how often PPO proposes an action the rule engine
    has already blocked (policy_adapter.propose() returns None because of
    this -- a legitimate fail-safe catch, but signals PPO under-learned a
    constraint)
  - divergence_rate: of the remaining *legal* proposals, how often the
    verifier picks something different from what PPO proposed
  - a breakdown of what the verifier substitutes, when it does override

Usage (from apexguard-backend/, with ppo_policy.zip already trained):
    python scripts/stress_test_ppo_vs_verifier.py --n 500
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from app.schemas.telemetry import (
    DecisionRequest, EgoState, TargetState, TrafficState, TrackState, RulesState,
)
from app.schemas.common import DecisionMode
from app.engine.decision_engine import decide
from app.engine import rule_engine
from app.models import policy_adapter, overtake_probability
from app.simulation.tracks import list_profile_ids, load_profile

ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "app" / "data" / "artifacts" / "ppo_vs_verifier_stress.json"


def random_request(rng: random.Random, request_id: str, track_ids: list[str]) -> DecisionRequest:
    """Builds a randomized DecisionRequest the same way
    ApexGuardEnv._sample_episode_state() randomizes training episodes, so
    PPO is being tested on the distribution it actually trained on."""
    soc = rng.uniform(15.0, 90.0)
    min_reserve = rng.uniform(10.0, 20.0)
    if soc < min_reserve + 5.0:
        soc = min_reserve + rng.uniform(5.0, 15.0)

    if track_ids:
        profile = load_profile(rng.choice(track_ids))
        overtake_difficulty = profile.overtake_difficulty
        drs_available = any(s.has_drs for s in profile.sectors)
    else:
        overtake_difficulty = rng.uniform(0.2, 0.9)
        drs_available = rng.random() < 0.6

    return DecisionRequest(
        request_id=request_id,
        timestamp_ms=0,
        telemetry_age_ms=rng.choice([20, 30, 40, 50]),
        ego=EgoState(
            soc_pct=soc,
            speed_kph=rng.uniform(180, 320),
            throttle_pct=rng.uniform(60, 100),
            brake_pct=0,
            tyre_grip_estimate=rng.uniform(0.5, 0.95),
            laps_remaining=rng.randint(1, 55),
            current_mode="HOLD",
        ),
        target=TargetState(
            gap_s=rng.uniform(0.15, 2.0),
            relative_speed_kph=rng.uniform(-5, 30),
            stint_age_laps=rng.randint(1, 40),
            recent_sector_delta_s=rng.gauss(0, 0.2),
        ),
        traffic=TrafficState(
            rear_gap_s=rng.uniform(0.2, 3.0),
            cars_within_3s=rng.randint(0, 5),
            post_pass_traffic_gap_s=rng.uniform(0.3, 2.5),
        ),
        track=TrackState(
            track_id="stress_test",
            segment_type="DRS_STRAIGHT" if drs_available else "CORNER",
            drs_available=drs_available,
            straight_remaining_m=rng.uniform(150, 900),
            braking_zone_m=rng.uniform(90, 160),
            overtake_difficulty=overtake_difficulty,
        ),
        rules=RulesState(
            deployment_budget_remaining_kj=rng.uniform(400, 2200),
            minimum_reserve_soc_pct=min_reserve,
            full_deploy_allowed=True,
        ),
        decision_mode=DecisionMode.VERIFIED,
    )


def run(n: int, seed: int):
    if not policy_adapter.policy_is_available():
        print("WARNING: no PPO policy loaded. Aborting -- nothing meaningful to stress-test.")
        return

    overtake_probability.model_is_available()
    track_ids = list_profile_ids()
    rng = random.Random(seed)

    illegal_proposals = 0
    legal_proposals = 0
    divergences = []
    override_reasons = Counter()

    for i in range(n):
        req = random_request(rng, request_id=f"stress-{i}", track_ids=track_ids)
        legal = rule_engine.allowed_actions(req)

        ppo_proposal_raw = policy_adapter._try_load_ppo()
        # Re-derive whether PPO's *raw* choice (before the legality filter
        # inside propose()) was illegal, since propose() already discards
        # illegal proposals as None.
        import numpy as np
        obs = policy_adapter._observation_from_request(req)
        action_idx, _ = ppo_proposal_raw.predict(obs, deterministic=True)
        raw_action = policy_adapter._ACTIONS[int(action_idx)]

        if raw_action not in legal:
            illegal_proposals += 1
            continue

        legal_proposals += 1
        response = decide(req)
        if raw_action.value != response.recommended_action.value:
            override_reasons[(raw_action.value, response.recommended_action.value)] += 1
            divergences.append({
                "request_id": req.request_id,
                "ppo_proposed": raw_action.value,
                "verifier_selected": response.recommended_action.value,
                "explanation": response.explanation,
                "counterattack_risk": response.counterattack_risk,
                "risk_level": response.risk_level.value,
            })

    print(f"n={n}")
    print(f"illegal_proposal_rate={illegal_proposals / n:.3f}  ({illegal_proposals}/{n})")
    print(f"legal_proposal_rate={legal_proposals / n:.3f}")
    if legal_proposals:
        print(f"divergence_rate_among_legal={len(divergences) / legal_proposals:.3f}  "
              f"({len(divergences)}/{legal_proposals})")

    if override_reasons:
        print("\nWhat the verifier substitutes, and how often:")
        for (proposed, selected), count in override_reasons.most_common():
            print(f"  PPO proposed {proposed:16s} -> verifier selected {selected:16s}  x{count}")

    if divergences:
        print(f"\nExample divergences (first 5):")
        for d in divergences[:5]:
            print(f"  {d['ppo_proposed']} -> {d['verifier_selected']}  "
                  f"(risk={d['risk_level']}, counterattack={d['counterattack_risk']:.2f})")
            print(f"    {d['explanation']}")

    with ARTIFACT_PATH.open("w") as f:
        json.dump({
            "n": n,
            "illegal_proposal_rate": illegal_proposals / n,
            "divergence_rate_among_legal": len(divergences) / legal_proposals if legal_proposals else None,
            "divergences": divergences,
        }, f, indent=2)
    print(f"\nWrote full results -> {ARTIFACT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    run(args.n, args.seed)
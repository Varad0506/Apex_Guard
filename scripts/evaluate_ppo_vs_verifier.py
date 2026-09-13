"""Runs /v1/decide across all curated scenarios with the trained PPO policy
live, and reports every case where the verifier overrode PPO's proposal --
this is the guide's headline ablation case (policy_proposal != recommended_action),
now produced by a real trained policy instead of a hand-authored example.

Also runs PPO "raw" (no verifier) against the same scenarios by directly
calling policy_adapter.propose(), so you can see side-by-side what PPO
alone would have done vs. what the verifier-gated pipeline actually
recommends.

Usage (from apexguard-backend/, with ppo_policy.zip already trained):
    python scripts/evaluate_ppo_vs_verifier.py
"""

import json
from pathlib import Path

from app.schemas.telemetry import DecisionRequest
from app.engine.decision_engine import decide
from app.engine import rule_engine
from app.models import policy_adapter, overtake_probability

SCENARIOS_DIR = Path(__file__).resolve().parents[1] / "app" / "data" / "scenarios"


def run():
    if not policy_adapter.policy_is_available():
        print("WARNING: no PPO policy loaded (policy_adapter.policy_is_available() is False).")
        print("Check app/data/artifacts/ppo_policy.zip exists and stable_baselines3 is installed.")
        print("Continuing anyway -- results below will show PPO proposing nothing (None) for every case.\n")

    overtake_probability.model_is_available()  # warm classifier too, for fair latency numbers

    print(f"{'scenario':22s} {'ppo_raw_proposal':18s} {'verified_recommendation':22s} {'overridden?':12s} {'net_value':10s}")
    print("-" * 95)

    divergences = []
    for path in sorted(SCENARIOS_DIR.glob("*.json")):
        with path.open() as f:
            data = json.load(f)
        req = DecisionRequest(**data)

        legal = rule_engine.allowed_actions(req)
        ppo_raw = policy_adapter.propose(req, legal)

        response = decide(req)
        overridden = (ppo_raw is not None) and (ppo_raw.value != response.recommended_action.value)

        selected_candidate = next(
            (c for c in response.candidates if c.action == response.recommended_action), None
        )
        net_value = selected_candidate.expected_net_value if selected_candidate else float("nan")

        print(f"{path.stem:22s} {str(ppo_raw):18s} {response.recommended_action.value:22s} "
              f"{'YES' if overridden else 'no':12s} {net_value:+.4f}")

        if overridden:
            divergences.append({
                "scenario": path.stem,
                "ppo_proposal": ppo_raw.value,
                "verified_recommendation": response.recommended_action.value,
                "explanation": response.explanation,
                "candidates": [c.model_dump() for c in response.candidates],
            })

    print()
    if divergences:
        print(f"Found {len(divergences)} case(s) where the verifier overrode PPO's proposal:\n")
        for d in divergences:
            print(f"  [{d['scenario']}] PPO proposed {d['ppo_proposal']}, "
                  f"verifier selected {d['verified_recommendation']}")
            print(f"    reason: {d['explanation']}\n")
    else:
        print("No divergences in this scenario set -- PPO's raw proposals matched the verifier's")
        print("choice every time. That's a legitimate finding too (means PPO learned something")
        print("consistent with the verifier's own risk model on these particular scenarios), but")
        print("it's worth testing against the 10 synthetic track profiles too, where PPO is more")
        print("likely to encounter conditions further from what it was trained on.")

    out_path = Path(__file__).resolve().parents[1] / "app" / "data" / "artifacts" / "ppo_vs_verifier.json"
    with out_path.open("w") as f:
        json.dump(divergences, f, indent=2)
    print(f"\nWrote divergence details -> {out_path}")


if __name__ == "__main__":
    run()

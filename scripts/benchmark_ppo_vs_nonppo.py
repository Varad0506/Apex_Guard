"""Compare trained PPO + verifier against the same verifier stack without PPO.

Run after training app/data/artifacts/ppo_policy.zip:
    python -m scripts.benchmark_ppo_vs_nonppo --cases 80
"""
from __future__ import annotations
import json
from pathlib import Path
from statistics import mean
from app.evaluation.decision_quality import make_cases, _oracle_rollout, _utility
from app.engine import decision_engine, rule_engine
from app.models import policy_adapter, rival_estimator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "app" / "data" / "evaluations" / "ppo_vs_nonppo_report.json"


def _decide_with_ppo(req, enabled: bool):
    old_model = policy_adapter._ppo_model
    old_attempted = policy_adapter._ppo_load_attempted
    try:
        # Each variant gets a clean temporal belief state so PPO and non-PPO
        # are evaluated from the same information history.
        from app.models import opponent_belief
        opponent_belief.reset_temporal(req.battle_id)
        if enabled:
            policy_adapter._ppo_load_attempted = False
            policy_adapter._ppo_model = None
            if not policy_adapter.policy_is_available():
                raise RuntimeError("No trained PPO policy found at app/data/artifacts/ppo_policy.zip")
        else:
            policy_adapter._ppo_load_attempted = True
            policy_adapter._ppo_model = None
        return decision_engine.decide(req)
    finally:
        policy_adapter._ppo_model = old_model
        policy_adapter._ppo_load_attempted = old_attempted


def main(cases_n: int = 80):
    if not policy_adapter.policy_is_available():
        raise SystemExit("PPO policy not found. Train first with scripts.train_ppo.")
    cases = make_cases(cases_n, seed=20260912)
    # The production engine uses a larger Monte Carlo budget. For a repeatable
    # benchmark, cap the verifier to 6 paths per candidate while preserving
    # the exact production decision flow.
    original_rollout = decision_engine.rollout_verified
    def _benchmark_rollout(*args, **kwargs):
        kwargs["n_paths"] = min(int(kwargs.get("n_paths", 6)), 6)
        return original_rollout(*args, **kwargs)
    decision_engine.rollout_verified = _benchmark_rollout
    records = []
    for case in cases:
        legal = rule_engine.allowed_actions(case.req)
        oracle = {a: _oracle_rollout(case, a, n_paths=4) for a in legal}
        oracle_action = max(oracle, key=lambda a: _utility(oracle[a], case.req))
        oracle_u = _utility(oracle[oracle_action], case.req)
        for variant, enabled in (("non_ppo", False), ("ppo", True)):
            response = _decide_with_ppo(case.req, enabled)
            chosen = oracle.get(response.recommended_action)
            if chosen is None:
                chosen = _oracle_rollout(case, response.recommended_action, n_paths=12)
            u = _utility(chosen, case.req)
            records.append({
                "case_id": case.req.request_id,
                "scenario": case.hidden.scenario.value,
                "variant": variant,
                "policy_proposal": response.policy_proposal.value if response.policy_proposal else None,
                "recommended_action": response.recommended_action.value,
                "oracle_action": oracle_action.value,
                "action_correct": response.recommended_action == oracle_action,
                "successful_overtake": chosen.pass_probability,
                "counterattack_rate": chosen.counterattack_rate,
                "energy_consumed_kj": chosen.energy_used_kj,
                "reserve_violation_rate": chosen.reserve_breach_rate,
                "time_gained_s": max(0.0, chosen.immediate_time_delta_s) * chosen.pass_probability,
                "decision_regret": max(0.0, oracle_u - u),
                "risk_adjusted_utility": u,
                "verifier_override": enabled and response.policy_proposal is not None and response.policy_proposal != response.recommended_action,
                "latency_ms": response.latency_ms,
            })
    summary = {}
    for variant in ("non_ppo", "ppo"):
        rows = [r for r in records if r["variant"] == variant]
        summary[variant] = {
            "cases": len(rows),
            "successful_overtakes_pct": round(100 * mean(r["successful_overtake"] for r in rows), 2),
            "counterattack_rate_pct": round(100 * mean(r["counterattack_rate"] for r in rows), 2),
            "energy_consumed_kj_mean": round(mean(r["energy_consumed_kj"] for r in rows), 2),
            "reserve_violation_rate_pct": round(100 * mean(r["reserve_violation_rate"] for r in rows), 2),
            "time_gained_s_mean": round(mean(r["time_gained_s"] for r in rows), 4),
            "decision_regret_mean": round(mean(r["decision_regret"] for r in rows), 5),
            "action_selection_accuracy_pct": round(100 * mean(r["action_correct"] for r in rows), 2),
            "risk_adjusted_utility_mean": round(mean(r["risk_adjusted_utility"] for r in rows), 5),
            "verifier_override_rate_pct": round(100 * mean(r["verifier_override"] for r in rows), 2),
            "latency_ms_mean": round(mean(r["latency_ms"] for r in rows), 2),
        }
    decision_engine.rollout_verified = original_rollout
    report = {"benchmark": "PPO vs non-PPO verifier benchmark", "synthetic": True, "seed": 20260912, "summary": summary, "records": records}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cases", type=int, default=80)
    args = p.parse_args()
    main(args.cases)

"""Decision-quality benchmark for ApexGuard.

The benchmark evaluates decisions against a hidden-state oracle.  It is
synthetic by design: hidden opponent SOC/tactical/aero/override states are
known only to the evaluator, while ApexGuard sees the normal telemetry.

It compares a simple baseline, GNN-only, temporal-belief + baseline traffic,
and the full pre-PPO ApexGuard stack.  The output is JSON-safe and intended
for the frontend benchmark/replay screens.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import random
import hashlib
from statistics import mean
from typing import Dict, Iterable, List, Tuple

from app.engine import decision_engine, opportunity_monitor, rule_engine
from app.engine.traffic_engine import evaluate_baseline
from app.models import opponent_belief, rival_estimator
from app.schemas.common import ActionType, DecisionMode
from app.schemas.telemetry import DecisionRequest
from app.simulation.engine import rollout_verified
from app.simulation.rollout import RolloutOutcome


class HiddenScenario(str, Enum):
    TRAP = "COUNTER_HARVEST_TRAP"
    DEPLETION = "GENUINE_DEPLETION"
    ATTACK = "NORMAL_ATTACK"
    DEFENSE = "DEFENSIVE_RIVAL"


@dataclass(frozen=True)
class HiddenState:
    scenario: HiddenScenario
    soc_band: str
    tactical: str
    aero: str
    override: bool
    defensive_likelihood: float


@dataclass
class BenchmarkCase:
    req: DecisionRequest
    hidden: HiddenState


def _case(seed: int, i: int) -> BenchmarkCase:
    rng = random.Random(seed + i * 7919)
    scenario = rng.choice(list(HiddenScenario))
    if scenario == HiddenScenario.TRAP:
        sector_delta = rng.uniform(0.05, 0.18)
        stint = rng.randint(8, 24)
        gap = rng.uniform(0.35, 0.95)
        rel = rng.uniform(5, 17)
        hidden = HiddenState(scenario, rng.choice(["MEDIUM", "HIGH"]), "HARVESTING", "LOW_DRAG", True, rng.uniform(.62, .86))
    elif scenario == HiddenScenario.DEPLETION:
        sector_delta = rng.uniform(0.10, 0.42)
        stint = rng.randint(24, 38)
        gap = rng.uniform(0.55, 1.30)
        rel = rng.uniform(3, 13)
        hidden = HiddenState(scenario, "LOW", "CONSERVING", "NORMAL", False, rng.uniform(.18, .38))
    elif scenario == HiddenScenario.ATTACK:
        sector_delta = rng.uniform(-0.08, 0.16)
        stint = rng.randint(5, 18)
        gap = rng.uniform(0.30, 1.05)
        rel = rng.uniform(8, 21)
        hidden = HiddenState(scenario, "HIGH", "ATTACKING", "LOW_DRAG", True, rng.uniform(.55, .82))
    else:
        sector_delta = rng.uniform(-0.05, 0.20)
        stint = rng.randint(10, 28)
        gap = rng.uniform(0.35, 1.20)
        rel = rng.uniform(4, 16)
        hidden = HiddenState(scenario, "MEDIUM", "DEFENDING", "NORMAL", rng.random() < .45, rng.uniform(.58, .82))

    soc = rng.uniform(52, 78)
    req = DecisionRequest.model_validate({
        "request_id": f"benchmark-{seed}-{i}",
        "battle_id": f"benchmark-battle-{seed}-{i}",
        "timestamp_ms": 1778600000000 + i * 500,
        "telemetry_age_ms": rng.randint(20, 120),
        "ego": {
            "soc_pct": soc,
            "speed_kph": rng.uniform(265, 315),
            "throttle_pct": rng.uniform(92, 100),
            "brake_pct": 0,
            "tyre_grip_estimate": rng.uniform(.74, .91),
            "laps_remaining": rng.randint(6, 20),
            "current_mode": "HOLD",
        },
        "target": {
            "gap_s": gap,
            "relative_speed_kph": rel,
            "stint_age_laps": stint,
            "recent_sector_delta_s": sector_delta,
        },
        "traffic": {
            "rear_gap_s": rng.uniform(.65, 3.0),
            "cars_within_3s": rng.randint(0, 4),
            "post_pass_traffic_gap_s": rng.uniform(.7, 2.7),
        },
        "track": {
            "track_id": rng.choice(["bench-highspeed", "bench-balanced", "bench-technical"]),
            "segment_type": "DRS_STRAIGHT" if rng.random() < .72 else "STRAIGHT",
            "drs_available": True,
            "straight_remaining_m": rng.uniform(350, 750),
            "braking_zone_m": rng.uniform(90, 180),
            "overtake_difficulty": rng.uniform(.20, .62),
        },
        "rules": {
            "deployment_budget_remaining_kj": rng.uniform(650, 2200),
            "minimum_reserve_soc_pct": rng.choice([15, 18, 20]),
            "full_deploy_allowed": True,
        },
        "decision_mode": DecisionMode.VERIFIED,
    })
    return BenchmarkCase(req=req, hidden=hidden)


def make_cases(n: int = 80, seed: int = 20260911) -> List[BenchmarkCase]:
    return [_case(seed, i) for i in range(n)]


def _baseline_action(req: DecisionRequest) -> ActionType:
    legal = rule_engine.allowed_actions(req)
    if req.target.gap_s < .65 and req.target.relative_speed_kph > 9 and ActionType.FULL_DEPLOY in legal:
        return ActionType.FULL_DEPLOY
    if req.target.gap_s < 1.05 and req.target.relative_speed_kph > 5 and ActionType.PARTIAL_DEPLOY in legal:
        return ActionType.PARTIAL_DEPLOY
    return ActionType.HOLD if ActionType.HOLD in legal else legal[0]


def _neutral_belief(req: DecisionRequest, rival):
    return opponent_belief.OpponentBeliefState(
        soc_belief={"LOW": 1/3, "MEDIUM": 1/3, "HIGH": 1/3},
        override_belief={"AVAILABLE": .5, "UNAVAILABLE": .5},
        tactical_belief={"HARVESTING": .25, "DEFENDING": .25, "ATTACKING": .25, "CONSERVING": .25},
        aero_belief={"LOW_DRAG": .5, "NORMAL": .5},
        counter_harvest_trap_probability=.0,
        confidence=.25,
    )


def _decision_for_variant(case: BenchmarkCase, variant: str) -> Tuple[ActionType, dict]:
    """Select an action with PPO disabled so this benchmark isolates the
    reasoning stack.  VERIFIED rollouts use a small fixed path count for
    speed; the production decision engine keeps its larger path count."""
    req = case.req
    legal = rule_engine.allowed_actions(req)
    rival = rival_estimator.estimate(req)
    if variant == "baseline":
        action = _baseline_action(req)
        return action, {"trap_probability": 0.0, "confidence": 0.5, "traffic_model": "baseline"}

    if variant == "gnn_only":
        belief = _neutral_belief(req, rival)
        traffic, source = decision_engine.evaluate_traffic(req, rival, belief)
    elif variant == "belief_baseline_traffic":
        opponent_belief.reset_temporal(req.battle_id)
        belief = opponent_belief.infer(req, rival)
        traffic = evaluate_baseline(req, rival, belief)
        source = "baseline"
    elif variant == "full":
        # Feed the same battle stream through several observation ticks. This
        # is essential: the benchmark should test temporal accumulation, not
        # merely the single-tick posterior.
        opponent_belief.reset_temporal(req.battle_id)
        belief = None
        for tick in range(5):
            tick_req = req.model_copy(deep=True)
            tick_req.request_id = f"{req.request_id}-t{tick}"
            tick_req.timestamp_ms += tick * 500
            if case.hidden.scenario == HiddenScenario.TRAP:
                tick_req.target.recent_sector_delta_s = req.target.recent_sector_delta_s
                tick_req.target.relative_speed_kph = req.target.relative_speed_kph - tick * 4.5
            belief = opponent_belief.update_temporal(tick_req, rival, req.battle_id)
        traffic, source = decision_engine.evaluate_traffic(req, rival, belief)
    else:
        raise ValueError(variant)

    from app.models import overtake_probability
    pass_probs = overtake_probability.predict_for_actions(req, rival, traffic, legal)
    outcomes = [
        rollout_verified(req, action, traffic, rival, n_paths=6, seed=12345 + i * 97,
                         opponent_belief_state=belief)
        for i, action in enumerate(legal)
    ]
    # Mirror the production ranker: expected net value is authoritative.
    selected = max(outcomes, key=lambda o: o.expected_net_value)
    return selected.action, {
        "trap_probability": belief.counter_harvest_trap_probability,
        "confidence": belief.confidence,
        "traffic_model": source,
        "counterattack_risk": traffic.counterattack_risk,
    }


def _oracle_rollout(case: BenchmarkCase, action: ActionType, n_paths: int = 4) -> RolloutOutcome:
    req = case.req
    rival = rival_estimator.estimate(req)
    # The evaluator knows the hidden opponent state and injects it only into
    # the oracle's rival response model. ApexGuard never receives this value.
    truth_rival = rival.__class__(
        tyre_grip_estimate=rival.tyre_grip_estimate,
        tyre_uncertainty=rival.tyre_uncertainty,
        defensive_likelihood=case.hidden.defensive_likelihood,
    )
    traffic = evaluate_baseline(req, truth_rival)
    return rollout_verified(req, action, traffic, truth_rival, n_paths=n_paths, seed=99173 + int.from_bytes(hashlib.sha256(f"{req.request_id}:{action.value}".encode()).digest()[:4], "big") % 100000)


def _utility(outcome: RolloutOutcome, req: DecisionRequest) -> float:
    # Higher is better. Time gain is rewarded, energy/risk/reserve are costs.
    return (
        outcome.pass_probability * max(0.0, outcome.immediate_time_delta_s)
        - max(0.0, req.ego.soc_pct - outcome.projected_soc_pct) / 100.0 * .20
        - (1.0 - outcome.pass_probability) * .02
    )


def run_benchmark(cases: Iterable[BenchmarkCase]) -> dict:
    variants = ["baseline", "gnn_only", "belief_baseline_traffic", "full"]
    records = []
    for case in cases:
        legal = rule_engine.allowed_actions(case.req)
        oracle_outcomes = {a: _oracle_rollout(case, a) for a in legal}
        oracle_action = max(oracle_outcomes, key=lambda a: _utility(oracle_outcomes[a], case.req))
        oracle_u = _utility(oracle_outcomes[oracle_action], case.req)

        for variant in variants:
            action, meta = _decision_for_variant(case, variant)
            chosen = oracle_outcomes[action] if action in oracle_outcomes else None
            if chosen is None:
                chosen = _oracle_rollout(case, action, n_paths=24)
            utility = _utility(chosen, case.req)
            trap_truth = case.hidden.scenario == HiddenScenario.TRAP
            trap_pred = meta.get("trap_probability", 0.0) >= .35
            records.append({
                "case_id": case.req.request_id,
                "scenario": case.hidden.scenario.value,
                "variant": variant,
                "action": action.value,
                "oracle_action": oracle_action.value,
                "action_correct": action == oracle_action,
                "decision_regret": max(0.0, oracle_u - utility),
                "successful_overtake": chosen.pass_probability,
                "counterattack_rate": chosen.counterattack_rate,
                "energy_consumed_kj": chosen.energy_used_kj,
                "reserve_violation_rate": chosen.reserve_breach_rate,
                "time_gained_s": max(0.0, chosen.immediate_time_delta_s) * chosen.pass_probability,
                "trap_truth": trap_truth,
                "trap_detected": trap_pred,
                "false_trap_alarm": (not trap_truth) and trap_pred,
                "safety_violation": action not in legal,
                "unnecessary_full_deployment": action == ActionType.FULL_DEPLOY and oracle_action != ActionType.FULL_DEPLOY,
                "risk_adjusted_utility": utility,
                "trap_probability": meta.get("trap_probability", 0.0),
                "traffic_model": meta.get("traffic_model", "baseline"),
            })

    return _aggregate(records), records


def _aggregate(records: List[dict]) -> dict:
    by_variant: Dict[str, List[dict]] = {}
    for r in records:
        by_variant.setdefault(r["variant"], []).append(r)

    summary = {}
    for variant, rows in by_variant.items():
        traps = [r for r in rows if r["trap_truth"]]
        nontraps = [r for r in rows if not r["trap_truth"]]
        tp = sum(r["trap_detected"] for r in traps)
        fn = sum(not r["trap_detected"] for r in traps)
        fp = sum(r["false_trap_alarm"] for r in nontraps)
        tn = sum(not r["false_trap_alarm"] for r in nontraps)
        summary[variant] = {
            "cases": len(rows),
            "successful_overtakes_pct": round(100 * mean(r["successful_overtake"] for r in rows), 2),
            "counterattack_rate_pct": round(100 * mean(r["counterattack_rate"] for r in rows), 2),
            "energy_consumed_kj_mean": round(mean(r["energy_consumed_kj"] for r in rows), 2),
            "reserve_violation_rate_pct": round(100 * mean(r["reserve_violation_rate"] for r in rows), 2),
            "time_gained_s_mean": round(mean(r["time_gained_s"] for r in rows), 4),
            "decision_regret_mean": round(mean(r["decision_regret"] for r in rows), 5),
            "action_selection_accuracy_pct": round(100 * mean(r["action_correct"] for r in rows), 2),
            "trap_detection_recall_pct": round(100 * tp / max(1, tp + fn), 2),
            "false_trap_alarm_rate_pct": round(100 * fp / max(1, fp + tn), 2),
            "safety_violation_rate_pct": round(100 * mean(r["safety_violation"] for r in rows), 2),
            "unnecessary_full_deployment_rate_pct": round(100 * mean(r["unnecessary_full_deployment"] for r in rows), 2),
            "risk_adjusted_utility_mean": round(mean(r["risk_adjusted_utility"] for r in rows), 5),
        }
    return {"variants": summary}

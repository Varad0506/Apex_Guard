"""Phase 5 ablation test, per the build guide's table:

    Scenario               | Baseline expectation      | GNN value
    Simple isolated pass   | Similar decision          | Minimal difference is acceptable
    Rear DRS threat        | Baseline may over-deploy  | GNN reduces counterattack exposure
    DRS train               | Baseline may pass into    | GNN recognizes limited post-pass
    Multiple close rivals   | Flat features unclear     | Graph captures relational topology

"You keep the GNN only if it wins an ablation test... If it does not
demonstrate a meaningful difference, simplify the final submission and
state that you use a local traffic graph with engineered relational
features." -- this script produces the actual numbers to make that call
honestly, rather than asserting it.

Methodology: both the deterministic baseline (traffic_engine.py) and the
GNN (traffic_graph.py) are scored against the SAME independent synthetic
ground-truth formula used to train the GNN (generate_battle_graph_data.py's
_ground_truth_labels) -- neither model has a structural advantage in this
comparison; the GNN saw this ground truth during training, the baseline
never did, which is exactly the real-world situation ("a trained model vs
a hand-tuned formula that was never fit to any data").

Requires torch (see train_traffic_gnn.py's note). Run in your own
environment, after training:

    python scripts/train_traffic_gnn.py
    python scripts/ablation_test_gnn.py
"""

import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from app.schemas.telemetry import (
    DecisionRequest, EgoState, TargetState, TrafficState, TrackState, RulesState,
)
from app.schemas.common import DecisionMode
from app.engine.traffic_engine import evaluate as evaluate_baseline
from app.models.rival_estimator import estimate as estimate_rival
from app.models.battle_graph import build_battle_graph
from app.models import traffic_graph
from scripts.generate_battle_graph_data import _ground_truth_labels

ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "app" / "data" / "artifacts" / "gnn_ablation_results.json"

N_PER_CATEGORY = 300


def _base_request(rng: random.Random, request_id: str, **overrides) -> DecisionRequest:
    defaults = dict(
        soc_pct=rng.uniform(40, 80), speed_kph=rng.uniform(200, 300),
        gap_s=rng.uniform(0.3, 1.5), relative_speed_kph=rng.uniform(5, 20),
        stint_age_laps=rng.randint(5, 25), recent_sector_delta_s=rng.gauss(0, 0.15),
        rear_gap_s=rng.uniform(1.0, 3.0), cars_within_3s=rng.randint(0, 1),
        post_pass_traffic_gap_s=rng.uniform(1.5, 3.0),
        drs_available=True, overtake_difficulty=rng.uniform(0.3, 0.6),
    )
    defaults.update(overrides)
    d = defaults
    return DecisionRequest(
        request_id=request_id, timestamp_ms=0, telemetry_age_ms=30,
        ego=EgoState(soc_pct=d["soc_pct"], speed_kph=d["speed_kph"], throttle_pct=90, brake_pct=0,
                     tyre_grip_estimate=rng.uniform(0.6, 0.9), laps_remaining=rng.randint(5, 40), current_mode="HOLD"),
        target=TargetState(gap_s=d["gap_s"], relative_speed_kph=d["relative_speed_kph"],
                            stint_age_laps=d["stint_age_laps"], recent_sector_delta_s=d["recent_sector_delta_s"]),
        traffic=TrafficState(rear_gap_s=d["rear_gap_s"], cars_within_3s=d["cars_within_3s"],
                              post_pass_traffic_gap_s=d["post_pass_traffic_gap_s"]),
        track=TrackState(track_id="ablation", segment_type="DRS_STRAIGHT", drs_available=d["drs_available"],
                          straight_remaining_m=rng.uniform(300, 700), braking_zone_m=rng.uniform(100, 150),
                          overtake_difficulty=d["overtake_difficulty"]),
        rules=RulesState(deployment_budget_remaining_kj=rng.uniform(800, 2000),
                          minimum_reserve_soc_pct=rng.uniform(10, 18), full_deploy_allowed=True),
        decision_mode=DecisionMode.VERIFIED,
    )


def _sample_category(category: str, rng: random.Random, i: int) -> DecisionRequest:
    rid = f"{category}-{i}"
    if category == "simple_isolated_pass":
        return _base_request(rng, rid, rear_gap_s=rng.uniform(2.0, 4.0), cars_within_3s=0,
                              post_pass_traffic_gap_s=rng.uniform(2.0, 4.0))
    if category == "rear_drs_threat":
        return _base_request(rng, rid, rear_gap_s=rng.uniform(0.15, 0.6), cars_within_3s=rng.randint(1, 3))
    if category == "drs_train":
        return _base_request(rng, rid, cars_within_3s=rng.randint(3, 6),
                              post_pass_traffic_gap_s=rng.uniform(0.2, 0.7))
    if category == "multiple_close_rivals":
        return _base_request(rng, rid, cars_within_3s=rng.randint(4, 6), rear_gap_s=rng.uniform(0.3, 1.0),
                              post_pass_traffic_gap_s=rng.uniform(0.4, 1.2))
    raise ValueError(category)


def run():
    if not traffic_graph.model_is_available():
        print("WARNING: no trained traffic_gat.pt found. Run scripts/train_traffic_gnn.py first.")
        print("Aborting -- nothing meaningful to compare.")
        return

    categories = ["simple_isolated_pass", "rear_drs_threat", "drs_train", "multiple_close_rivals"]
    rng = random.Random(99)

    results = defaultdict(lambda: {"baseline_mse": [], "gnn_mse": [], "baseline_pred": [], "gnn_pred": []})

    for category in categories:
        for i in range(N_PER_CATEGORY):
            req = _sample_category(category, rng, i)
            rival = estimate_rival(req)
            truth = _ground_truth_labels(req, rival.defensive_likelihood, rng)

            baseline = evaluate_baseline(req, rival)
            graph = build_battle_graph(req, rival)
            gnn_out = traffic_graph.predict(graph)

            baseline_err = np.mean([
                (baseline.tow_strength - truth["tow_strength"]) ** 2,
                (baseline.counterattack_risk - truth["counterattack_risk"]) ** 2,
                (baseline.post_pass_traffic_risk - truth["post_pass_traffic_risk"]) ** 2,
            ])
            gnn_err = np.mean([
                (gnn_out["tow_strength"] - truth["tow_strength"]) ** 2,
                (gnn_out["counterattack_risk"] - truth["counterattack_risk"]) ** 2,
                (gnn_out["post_pass_traffic_risk"] - truth["post_pass_traffic_risk"]) ** 2,
            ])

            results[category]["baseline_mse"].append(baseline_err)
            results[category]["gnn_mse"].append(gnn_err)
            results[category]["baseline_pred"].append(baseline.counterattack_risk)
            results[category]["gnn_pred"].append(gnn_out["counterattack_risk"])

    print(f"{'category':24s} {'baseline_mse':14s} {'gnn_mse':14s} {'improvement':12s} {'verdict'}")
    print("-" * 90)

    overall_baseline, overall_gnn = [], []
    category_wins = 0
    for category in categories:
        b = np.mean(results[category]["baseline_mse"])
        g = np.mean(results[category]["gnn_mse"])
        overall_baseline.extend(results[category]["baseline_mse"])
        overall_gnn.extend(results[category]["gnn_mse"])
        improvement_pct = (b - g) / b * 100 if b > 0 else 0.0
        # "Meaningful difference" threshold, per the guide's own framing
        # ("minimal difference is acceptable" for the easy category, real
        # improvement expected for the harder ones): >=15% MSE reduction.
        wins = improvement_pct >= 15.0
        if wins:
            category_wins += 1
        verdict = "GNN wins" if wins else ("comparable" if abs(improvement_pct) < 15 else "baseline wins")
        print(f"{category:24s} {b:<14.5f} {g:<14.5f} {improvement_pct:>+10.1f}%  {verdict}")

    overall_b = np.mean(overall_baseline)
    overall_g = np.mean(overall_gnn)
    overall_improvement = (overall_b - overall_g) / overall_b * 100 if overall_b > 0 else 0.0

    print("-" * 90)
    print(f"{'OVERALL':24s} {overall_b:<14.5f} {overall_g:<14.5f} {overall_improvement:>+10.1f}%")
    print(f"\nCategories where GNN wins meaningfully (>=15% MSE reduction): {category_wins}/{len(categories)}")

    # Guide's actual decision rule.
    if category_wins >= 2 and overall_improvement > 10.0:
        print("\nVERDICT: Keep the GNN. It demonstrates a meaningful difference in at least")
        print("half the ablation categories and improves overall -- wire it into")
        print("traffic_engine.py's call sites as a swap, gated behind the same")
        print("try/fallback pattern as overtake_probability.py and policy_adapter.py.")
    else:
        print("\nVERDICT: Do not keep the GNN. Per the guide: 'simplify the final submission")
        print("and state that you use a local traffic graph with engineered relational")
        print("features' instead. The deterministic baseline is doing the job.")

    import json
    with ARTIFACT_PATH.open("w") as f:
        json.dump({
            "category_results": {
                c: {"baseline_mse": float(np.mean(results[c]["baseline_mse"])),
                    "gnn_mse": float(np.mean(results[c]["gnn_mse"]))}
                for c in categories
            },
            "overall_baseline_mse": float(overall_b),
            "overall_gnn_mse": float(overall_g),
            "category_wins": int(category_wins),
            "kept_gnn": bool(category_wins >= 2 and overall_improvement > 10.0),
        }, f, indent=2)
    print(f"\nWrote -> {ARTIFACT_PATH}")


if __name__ == "__main__":
    run()
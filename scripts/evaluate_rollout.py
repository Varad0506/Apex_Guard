"""Phase 0/3 evaluation: runs both rollout engines (FAST analytical,
VERIFIED Monte Carlo) across the 3 curated demo scenarios and reports what
each recommends, its confidence, and its latency -- a first cut at the
guide's "Method | Pass success | Net race-time value | ... | p95 latency"
results table. This compares the two *rollout engines* we actually have;
it is not yet the full PPO-vs-verifier ablation from the guide, since PPO
doesn't exist yet (see README, Phase 4).
"""

import copy
import json
import time
from pathlib import Path
from statistics import mean

from app.schemas.telemetry import DecisionRequest
from app.schemas.common import DecisionMode
from app.engine.decision_engine import decide
from app.models import overtake_probability

SCENARIOS_DIR = Path(__file__).resolve().parents[1] / "app" / "data" / "scenarios"


def run():
    overtake_probability.model_is_available()  # warm the classifier before timing anything
    scenario_paths = sorted(SCENARIOS_DIR.glob("*.json"))
    rows = []

    for mode in (DecisionMode.FAST, DecisionMode.VERIFIED):
        for path in scenario_paths:
            with path.open() as f:
                data = json.load(f)
            data = copy.deepcopy(data)
            data["decision_mode"] = mode.value
            req = DecisionRequest(**data)

            latencies = []
            for _ in range(5):
                start = time.perf_counter()
                response = decide(req)
                latencies.append((time.perf_counter() - start) * 1000)

            rows.append({
                "mode": mode.value,
                "scenario": path.stem,
                "recommended_action": response.recommended_action.value,
                "confidence": response.recommendation_confidence,
                "pass_probability": response.overtake_success_probability,
                "counterattack_risk": response.counterattack_risk,
                "mean_latency_ms": round(mean(latencies), 2),
                "p95_latency_ms": round(sorted(latencies)[-1], 2),
                "fallback_used": response.fallback_used,
            })

    col_widths = {"mode": 9, "scenario": 20, "recommended_action": 16, "confidence": 10,
                  "pass_probability": 8, "counterattack_risk": 8, "mean_latency_ms": 10,
                  "p95_latency_ms": 10}
    header = "  ".join(k.ljust(w) for k, w in col_widths.items())
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(str(row[k]).ljust(w) for k, w in col_widths.items()))

    out_path = Path(__file__).resolve().parents[1] / "app" / "data" / "artifacts" / "rollout_evaluation.json"
    with out_path.open("w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    run()

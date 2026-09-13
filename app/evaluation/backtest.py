"""Historical replay backtesting for ApexGuard.

Backtesting here means replaying historical FastF1-derived frames through the
same decision stack used by the API and aggregating decision/energy/safety
metrics. Public FastF1 telemetry does not expose every hidden state, so this
module explicitly labels reconstruction limits and does not pretend to know
actual battery SOC or the true counterfactual action outcome.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from app.schemas.telemetry import DecisionRequest
from app.engine.decision_engine import decide


def _as_request(frame: dict[str, Any]) -> DecisionRequest:
    return DecisionRequest.model_validate(frame)


def _action_is_aggressive(action: str) -> bool:
    return action in {"PARTIAL_DEPLOY", "FULL_DEPLOY"}


def backtest_replay(payload: dict[str, Any], max_frames: int | None = None) -> dict[str, Any]:
    """Run the real decision engine over a replay frame stream.

    Metrics are deliberately separated into observable/replay-derived metrics
    and model decision metrics. Historical ground truth for a counterfactual
    ApexGuard action is not inferred from public telemetry.
    """
    frames = payload.get("frames", [])
    if max_frames:
        frames = frames[:max_frames]
    if not frames:
        raise ValueError("Replay contains no frames")

    decisions = []
    for frame in frames:
        req = _as_request(frame)
        response = decide(req)
        decisions.append(response)

    actions = [d.recommended_action.value for d in decisions]
    aggressive = sum(_action_is_aggressive(a) for a in actions)
    fallback = sum(bool(d.fallback_used) for d in decisions)
    compliant = sum(bool(d.rule_compliant) for d in decisions)
    trap_values = [float(d.opponent_belief.get("counter_harvest_trap_probability", 0.0)) for d in decisions]
    confidence = [float(d.recommendation_confidence) for d in decisions]
    pass_prob = [float(d.overtake_success_probability) for d in decisions]
    latencies = [float(d.latency_ms) for d in decisions]
    gaps = [float(d.traffic_summary.target_gap_s) for d in decisions]
    # The first decision may include model/artifact cold-start overhead. Keep it
    # visible, but report steady-state latency separately so the benchmark does
    # not confuse initialization cost with per-decision runtime.
    warmup = min(1, len(latencies))
    steady_latencies = latencies[warmup:] or latencies

    action_counts = {a: actions.count(a) for a in sorted(set(actions))}
    high_trap_aggressive = sum(
        1 for d, trap in zip(decisions, trap_values)
        if trap >= 0.66 and _action_is_aggressive(d.recommended_action.value)
    )

    result = {
        "backtest_version": "1.0",
        "source": payload.get("source", "unknown"),
        "year": payload.get("year"),
        "event": payload.get("event"),
        "session": payload.get("session"),
        "driver": payload.get("driver"),
        "target_driver": payload.get("target_driver"),
        "start_lap": payload.get("start_lap", payload.get("lap")),
        "end_lap": payload.get("end_lap"),
        "target_start_lap": payload.get("target_start_lap", payload.get("target_lap")),
        "target_end_lap": payload.get("target_end_lap"),
        "frames_evaluated": len(decisions),
        "duration_s": payload.get("duration_s"),
        "synthetic_fields": payload.get("synthetic_fields", []),
        "metrics": {
            "rule_compliance_pct": round(100.0 * compliant / len(decisions), 2),
            "aggressive_action_rate_pct": round(100.0 * aggressive / len(decisions), 2),
            "fallback_rate_pct": round(100.0 * fallback / len(decisions), 2),
            "high_trap_aggressive_rate_pct": round(100.0 * high_trap_aggressive / max(1, len(decisions)), 2),
            "mean_overtake_probability_pct": round(100.0 * statistics.mean(pass_prob), 2),
            "mean_decision_confidence_pct": round(100.0 * statistics.mean(confidence), 2),
            "mean_counter_harvest_trap_pct": round(100.0 * statistics.mean(trap_values), 2),
            "mean_gap_s": round(statistics.mean(gaps), 3),
            "mean_latency_ms": round(statistics.mean(latencies), 2),
            "cold_start_latency_ms": round(latencies[0], 2),
            "steady_state_mean_latency_ms": round(statistics.mean(steady_latencies), 2),
            "steady_state_p95_latency_ms": round(sorted(steady_latencies)[max(0, int(0.95 * len(steady_latencies)) - 1)], 2),
        },
        "action_counts": action_counts,
        "decision_timeline": [
            {
                "t_s": frame.get("t_s"),
                "request_id": response.request_id,
                "action": response.recommended_action.value,
                "policy_proposal": response.policy_proposal.value if response.policy_proposal else None,
                "overtake_probability": response.overtake_success_probability,
                "confidence": response.recommendation_confidence,
                "counter_harvest_trap_probability": response.opponent_belief.get("counter_harvest_trap_probability", 0.0),
                "risk_level": response.risk_level.value,
                "fallback_used": response.fallback_used,
                "latency_ms": response.latency_ms,
                "explanation": response.explanation,
            }
            for frame, response in zip(frames, decisions)
        ],
        "interpretation": {
            "historical_ground_truth_available": False,
            "note": "This backtest replays public historical telemetry through ApexGuard. Public telemetry does not expose all hidden battery, tyre, traffic and rule state needed to label counterfactual action success; those fields remain explicitly synthetic where applicable.",
        },
    }
    return result


def backtest_file(path: str | Path, output_path: str | Path | None = None, max_frames: int | None = None) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = backtest_replay(payload, max_frames=max_frames)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

# ApexGuard Quantum Amplitude Estimation Demo

This is an experimental research path. It does **not** modify the production `/v1/decide` pipeline.

## What is integrated

```text
OpenF1 / FastF1 telemetry
        |
        v
ApexGuard calibrated Logistic Regression
        |
        +----> production decision path (unchanged)
        |
        +----> /v1/quantum/estimate
                    |
                    v
             Qiskit Statevector
                    |
                    v
             toy QAE resolution model
```

The quantum module actually executes the amplitude-preparation circuit with `Statevector.from_instruction()` and extracts `P(|1>)`. The finite evaluation-register resolution and query-count comparison are explicitly simplified research models; no quantum hardware or quantum speedup is claimed.

## Install

```bash
pip install -r requirements.txt
```

## Toy benchmark

```bash
python -m scripts.benchmark_quantum_ae
python -m scripts.benchmark_quantum_ae --qubits 6
python -m scripts.benchmark_quantum_ae --qubits 8
```

The benchmark writes `app/data/evaluations/quantum_ae_report.json`.

## API

Start the backend normally, then use `POST /v1/quantum/estimate` from Swagger at `/docs`.

Example request:

```json
{
  "state": {
    "request_id": "quantum-demo-001",
    "battle_id": "battle-001",
    "timestamp_ms": 0,
    "telemetry_age_ms": 100,
    "ego": {
      "soc_pct": 65,
      "speed_kph": 310,
      "throttle_pct": 100,
      "brake_pct": 0,
      "tyre_grip_estimate": 0.9,
      "laps_remaining": 10,
      "current_mode": "ATTACK"
    },
    "target": {
      "gap_s": 0.7,
      "relative_speed_kph": 8,
      "stint_age_laps": 12,
      "recent_sector_delta_s": -0.1
    },
    "traffic": {
      "rear_gap_s": 2.5,
      "cars_within_3s": 1,
      "post_pass_traffic_gap_s": 2.0
    },
    "track": {
      "track_id": "demo",
      "segment_type": "STRAIGHT",
      "drs_available": true,
      "straight_remaining_m": 700,
      "braking_zone_m": 150,
      "overtake_difficulty": 0.3
    },
    "rules": {
      "deployment_budget_remaining_kj": 1000,
      "minimum_reserve_soc_pct": 20,
      "full_deploy_allowed": true
    },
    "decision_mode": "VERIFIED"
  },
  "epsilon": 0.05,
  "evaluation_qubits": 6
}
```

The endpoint obtains the **real calibrated Logistic Regression probabilities** for all currently legal actions and runs the experimental quantum estimator separately for each action. The result contains `quantum_hardware_used: false`, `production_path_changed: false`, and `research_only: true`.

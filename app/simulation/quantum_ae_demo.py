"""ApexGuard toy Quantum Amplitude Estimation experiment.

Research-only module. It does not modify the production decision path.
The amplitude-preparation circuit is actually executed with Qiskit's
Statevector simulator. The finite evaluation-register/QAE resolution and
query-scaling portions are explicitly labeled as a simplified model.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


@dataclass
class ClassicalResult:
    epsilon: float
    true_probability: float
    estimate: float
    absolute_error: float
    samples: int
    runtime_ms: float


@dataclass
class QuantumResult:
    epsilon: float
    true_probability: float
    estimate: float
    absolute_error: float
    oracle_calls: int
    evaluation_qubits: int
    runtime_ms: float
    method: str


def simplified_overtake_probability(
    gap_seconds: float,
    relative_speed_kmh: float,
    soc_percent: float,
    drs_available: bool,
    rival_defending: bool,
) -> float:
    """Small illustrative overtake-probability model."""
    score = 0.0
    score += max(0.0, 1.0 - gap_seconds / 2.0) * 0.35
    score += np.clip(relative_speed_kmh / 20.0, 0.0, 1.0) * 0.20
    score += np.clip(soc_percent / 100.0, 0.0, 1.0) * 0.20
    if drs_available:
        score += 0.15
    if rival_defending:
        score -= 0.15
    return float(np.clip(score, 0.01, 0.99))


def classical_required_samples(epsilon: float, confidence: float = 0.95) -> int:
    """Conservative distribution-free Hoeffding sample bound.

    Tighter Bernoulli-specific bounds exist; Hoeffding is used here because
    it makes the O(1/epsilon^2) scaling transparent.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    delta = 1.0 - confidence
    return int(math.ceil(math.log(2.0 / delta) / (2.0 * epsilon * epsilon)))


def quantum_query_budget(epsilon: float, confidence: float = 0.95) -> int:
    """Simplified theoretical O(1/epsilon) QAE query-budget model.

    This is not a hardware execution count; it is included only for the
    asymptotic scaling comparison.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    delta = 1.0 - confidence
    return int(math.ceil(math.log(1.0 / delta) / epsilon))


def build_toy_amplitude_circuit(probability: float) -> QuantumCircuit:
    """Prepare sqrt(1-p)|0> + sqrt(p)|1> with an Ry rotation."""
    probability = float(np.clip(probability, 0.0, 1.0))
    theta = 2.0 * math.asin(math.sqrt(probability))
    circuit = QuantumCircuit(1)
    circuit.ry(theta, 0)
    return circuit


def simulate_quantum_amplitude(
    true_probability: float,
    epsilon: float,
    evaluation_qubits: int = 4,
) -> QuantumResult:
    """Actually execute amplitude preparation using Statevector.

    Statevector produces the probability of |1>. We then separately model
    the finite QAE evaluation-register resolution; this is not a claim that
    the simplified grid itself is a full QAE algorithm.
    """
    if evaluation_qubits < 1:
        raise ValueError("evaluation_qubits must be >= 1")

    true_probability = float(np.clip(true_probability, 0.0, 1.0))
    start = time.perf_counter()

    circuit = build_toy_amplitude_circuit(true_probability)
    statevector = Statevector.from_instruction(circuit)
    probabilities = statevector.probabilities([0])
    measured_probability = float(probabilities[1])

    # Simplified amplitude-estimation resolution model.
    grid_size = 2 ** evaluation_qubits
    candidates = np.arange(grid_size)
    candidate_probabilities = np.sin(np.pi * candidates / grid_size) ** 2
    index = int(np.argmin(np.abs(candidate_probabilities - measured_probability)))
    qae_estimate = float(candidate_probabilities[index])

    # Simplified query accounting for the scaling demonstration only.
    oracle_calls = max(1, grid_size - 1)
    runtime_ms = (time.perf_counter() - start) * 1000.0

    return QuantumResult(
        epsilon=epsilon,
        true_probability=true_probability,
        estimate=qae_estimate,
        absolute_error=abs(qae_estimate - true_probability),
        oracle_calls=oracle_calls,
        evaluation_qubits=evaluation_qubits,
        runtime_ms=runtime_ms,
        method="Qiskit Statevector amplitude preparation + QAE resolution model",
    )


def run_classical_mc(true_probability: float, epsilon: float, seed: int = 42) -> ClassicalResult:
    """Run Bernoulli Monte Carlo using the conservative Hoeffding count."""
    samples = classical_required_samples(epsilon)
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    successes = rng.binomial(n=samples, p=true_probability)
    estimate = successes / samples
    runtime_ms = (time.perf_counter() - start) * 1000.0
    return ClassicalResult(
        epsilon=epsilon,
        true_probability=true_probability,
        estimate=float(estimate),
        absolute_error=abs(float(estimate) - true_probability),
        samples=samples,
        runtime_ms=runtime_ms,
    )


def run_scaling_experiment(
    true_probability: float = 0.62,
    epsilons: List[float] | None = None,
    evaluation_qubits: int = 4,
) -> Dict:
    """Run the classical-vs-quantum toy experiment."""
    if epsilons is None:
        epsilons = [0.20, 0.10, 0.05, 0.025]

    classical_results = []
    quantum_results = []
    scaling = []

    for index, epsilon in enumerate(epsilons):
        classical = run_classical_mc(true_probability, epsilon, seed=42 + index)
        quantum = simulate_quantum_amplitude(true_probability, epsilon, evaluation_qubits)
        classical_results.append(asdict(classical))
        quantum_results.append(asdict(quantum))
        cq = classical_required_samples(epsilon)
        qq = quantum_query_budget(epsilon)
        scaling.append({
            "epsilon": epsilon,
            "classical_O_1_over_epsilon_squared": cq,
            "quantum_O_1_over_epsilon": qq,
            "theoretical_query_ratio": cq / qq,
        })

    return {
        "experiment": "ApexGuard toy pass-probability estimation",
        "true_probability": true_probability,
        "evaluation_qubits": evaluation_qubits,
        "classical_results": classical_results,
        "quantum_results": quantum_results,
        "scaling": scaling,
        "interpretation": {
            "classical_complexity": "O(1/epsilon^2)",
            "quantum_complexity": "O(1/epsilon) up to confidence/log factors",
            "production_path_changed": False,
            "quantum_hardware_used": False,
            "warning": (
                "The quantum probability is obtained by executing the Qiskit "
                "Statevector circuit. The finite-resolution and query-scaling "
                "parts are a simplified QAE model; simulator runtime is not "
                "quantum hardware speedup."
            ),
        },
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run_scaling_experiment(), indent=2))

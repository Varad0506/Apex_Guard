"""
Bridge between ApexGuard's calibrated Logistic Regression model and the
experimental Quantum Amplitude Estimation module.

Research-only: this module never changes the production /v1/decide path.
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.engine.traffic_engine import TrafficAssessment
from app.models.overtake_probability import predict_for_actions
from app.models.rival_estimator import RivalEstimate
from app.schemas.common import ActionType
from app.schemas.telemetry import DecisionRequest
from app.simulation.quantum_ae_demo import simulate_quantum_amplitude


def estimate_quantum_from_logistic(
    req: DecisionRequest,
    rival: RivalEstimate,
    traffic: TrafficAssessment,
    legal_actions: List[ActionType],
    epsilon: float = 0.05,
    evaluation_qubits: int = 6,
) -> Dict[str, Any]:
    """Run the quantum experiment for every currently legal action.

    The input probability for each action comes from the same calibrated
    Logistic Regression pipeline used by the production decision engine.
    The quantum result is research metadata only.
    """
    if not legal_actions:
        raise ValueError("No legal action available for quantum estimation.")

    logistic_probabilities = predict_for_actions(
        req=req,
        rival=rival,
        traffic=traffic,
        legal_actions=legal_actions,
    )

    action_results: Dict[str, Any] = {}

    for action in legal_actions:
        probability = float(logistic_probabilities[action])

        quantum_result = simulate_quantum_amplitude(
            true_probability=probability,
            epsilon=epsilon,
            evaluation_qubits=evaluation_qubits,
        )

        action_results[action.value] = {
            "logistic_probability": probability,
            "quantum_estimate": quantum_result.estimate,
            "absolute_error": quantum_result.absolute_error,
            "epsilon": epsilon,
            "evaluation_qubits": evaluation_qubits,
            "oracle_calls": quantum_result.oracle_calls,
            "runtime_ms": quantum_result.runtime_ms,
        }

    return {
        "source": {
            "model": "ApexGuard calibrated Logistic Regression",
            "legal_actions": [a.value for a in legal_actions],
            "probabilities_by_action": {
                action.value: float(logistic_probabilities[action])
                for action in legal_actions
            },
        },
        "quantum": {
            "estimates_by_action": action_results,
            "method": (
                "Qiskit Statevector amplitude preparation "
                "+ QAE resolution model"
            ),
        },
        "complexity": {
            "classical_monte_carlo": "O(1/epsilon^2)",
            "quantum_amplitude_estimation": "O(1/epsilon)",
        },
        "metadata": {
            "quantum_hardware_used": False,
            "production_path_changed": False,
            "research_only": True,
        },
    }

import pytest


def test_quantum_statevector_executes():
    pytest.importorskip("qiskit")

    from app.simulation.quantum_ae_demo import build_toy_amplitude_circuit
    from qiskit.quantum_info import Statevector

    probability = 0.62
    circuit = build_toy_amplitude_circuit(probability)
    statevector = Statevector.from_instruction(circuit)
    probabilities = statevector.probabilities([0])

    assert len(probabilities) == 2
    assert float(probabilities[1]) == pytest.approx(probability, abs=1e-10)


def test_quantum_demo_returns_result():
    pytest.importorskip("qiskit")

    from app.simulation.quantum_ae_demo import simulate_quantum_amplitude

    result = simulate_quantum_amplitude(
        true_probability=0.62,
        epsilon=0.05,
        evaluation_qubits=4,
    )

    assert result.true_probability == 0.62
    assert 0.0 <= result.estimate <= 1.0
    assert result.absolute_error >= 0.0
    assert result.oracle_calls > 0
    assert result.evaluation_qubits == 4

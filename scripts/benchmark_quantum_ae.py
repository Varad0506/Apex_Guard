"""Run ApexGuard's toy Quantum Amplitude Estimation benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.simulation.quantum_ae_demo import run_scaling_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="ApexGuard Quantum Amplitude Estimation benchmark")
    parser.add_argument("--qubits", type=int, default=4, help="QAE evaluation qubits (2-8 recommended)")
    parser.add_argument("--probability", type=float, default=0.62, help="Toy overtake probability")
    args = parser.parse_args()

    report = run_scaling_experiment(
        true_probability=args.probability,
        evaluation_qubits=args.qubits,
    )

    output_dir = Path("app/data/evaluations")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "quantum_ae_report.json"
    output_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print("=" * 72)
    print("APEXGUARD QUANTUM AMPLITUDE ESTIMATION DEMO")
    print("=" * 72)
    print()
    print(f"Toy pass probability : {args.probability:.3f}")
    print(f"Evaluation qubits    : {args.qubits}")
    print()
    print(f"{'epsilon':>10}{'Classical':>18}{'Quantum':>18}{'Ratio':>14}")
    print("-" * 62)
    for row in report["scaling"]:
        print(
            f"{row['epsilon']:>10.3f}"
            f"{row['classical_O_1_over_epsilon_squared']:>18}"
            f"{row['quantum_O_1_over_epsilon']:>18}"
            f"{row['theoretical_query_ratio']:>14.2f}x"
        )
    print()
    print(f"Report written to: {output_file}")
    print()


if __name__ == "__main__":
    main()

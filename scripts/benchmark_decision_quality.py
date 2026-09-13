#!/usr/bin/env python3
"""Run ApexGuard decision-quality benchmark and write a frontend-ready report."""
import json
from pathlib import Path
from app.evaluation.decision_quality import make_cases, run_benchmark

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "app" / "data" / "evaluations"
OUT.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    cases = make_cases(n=80, seed=20260911)
    summary, records = run_benchmark(cases)
    report = {
        "benchmark": "ApexGuard Decision Quality Benchmark",
        "version": "pre-PPO-v1",
        "synthetic": True,
        "seed": 20260911,
        "cases": 80,
        "oracle": "hidden-state simulator",
        "variants": list(summary["variants"].keys()),
        "metrics": summary["variants"],
        "records": records,
    }
    path = OUT / "decision_quality_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {path}")

from app.evaluation.decision_quality import make_cases, run_benchmark


def test_benchmark_report_contains_required_metrics():
    summary, records = run_benchmark(make_cases(n=4, seed=7))
    required = {
        "successful_overtakes_pct", "counterattack_rate_pct", "energy_consumed_kj_mean",
        "reserve_violation_rate_pct", "time_gained_s_mean", "decision_regret_mean",
        "action_selection_accuracy_pct", "trap_detection_recall_pct",
        "false_trap_alarm_rate_pct", "safety_violation_rate_pct",
    }
    assert len(records) == 16
    for metrics in summary["variants"].values():
        assert required.issubset(metrics)


def test_benchmark_is_deterministic_for_same_seed():
    a, _ = run_benchmark(make_cases(n=3, seed=11))
    b, _ = run_benchmark(make_cases(n=3, seed=11))
    assert a == b

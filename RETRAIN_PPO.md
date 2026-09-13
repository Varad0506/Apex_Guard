# PPO Retraining + PPO vs Non-PPO Benchmark

This build changes the PPO observation from **24-D to 27-D** by adding three
confidence-aware logistic features:

1. calibrated Logistic Regression confidence,
2. normalized action-distribution entropy,
3. `corridor_open` probabilistic predicate.

Therefore the previously trained 24-D `ppo_policy.zip` is intentionally not
included. Retrain from scratch before running the PPO benchmark.

## Install

```bash
pip install -r requirements.txt
```

If your environment does not already provide them:

```bash
pip install "stable-baselines3[extra]" gymnasium
```

## One-command workflow

```bash
python -m scripts.retrain_ppo_and_benchmark --timesteps 50000 --cases 80
```

This performs:

- PPO training from scratch with the 27-D observation;
- 200-episode policy evaluation;
- the same synthetic hidden-state PPO-vs-non-PPO benchmark;
- report output to `app/data/evaluations/ppo_vs_nonppo_report.json`.

For a stronger final experiment, repeat with 100k or 200k timesteps and keep
the seed fixed when comparing variants.

## Important interpretation

The benchmark is synthetic and should be reported as a controlled decision-
quality experiment, not as real F1 race performance. PPO is proposal-only;
the legal mask, Monte Carlo verifier, and deterministic safety layer remain
authoritative.
